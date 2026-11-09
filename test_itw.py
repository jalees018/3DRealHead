"""Test-time inversion of the Gaussian-map generator on monocular in-the-wild videos (VHAP-tracked).

For every subject, a few frames of its video are used to personalise the trained generator:

1. The expression (SMIRK mesh offsets + texture gradients) and texture conditions of the first
   inversion frame are encoded once with the trained encoders.
2. For ``TOTAL_ITERS`` iterations a random inversion frame is rendered and compared to the video:
   the texture-encoder features, the FLAME shape/expression of each inversion frame and a per-subject
   colour transform are optimised; in the last phase the outermost decoder layers are fine-tuned too.
3. At ``RENDER_ITER`` the personalised avatar is exported (point cloud, per-frame images and,
   optionally, a video of the avatar re-rendered/driven with the remaining frames of the video).
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, utils
from tqdm import tqdm

from avatar.cameras import load_itw_cameras
from avatar.config import load_paths
from avatar.geometry import (FlameParams, gaussian_maps_to_3d, init_gaussians, load_canonical_position_map,
                             load_uv_layout, rotate_gaussians, split_sh, SH)
from avatar.losses import VGGLoss, uv_coordinate_colors, uv_tv_loss
from avatar.subjects import ITW_SUBJECTS
from avatar.utils import load_uv_mask, requires_grad, seed_everything, write_pointcloud
from flame import FLAMEModule
from gaussian_splatting import GaussianModel, render
from gaussian_splatting.loss_utils import ssim
from gaussian_splatting.sh_utils import eval_sh
from networks.generator import GaussianMapGeneratorInversion

VHAP_EXPORT_SUFFIX = "_whiteBg_staticOffset_maskBelowLine"

# Inversion frames of the paper's subjects (indices into the video). ``--num_inversion_frames n`` uses
# the first n entries; the first entry is the frame whose conditions are encoded.
INVERSION_FRAMES = dict(zip(ITW_SUBJECTS, [
    [0, 196, 437, 155, 370], [0, 219, 493, 134, 251], [0, 129, 499, 85, 322], [0, 59, 183, 91, 133],
    [10, 59, 139, 155, 248], [0, 136, 207, 79, 296], [107, 60, 399, 134, 278], [0, 104, 297, 54, 259],
    [0, 89, 319, 59, 289],
]))

BATCH = 2  # the generator is run on two copies of the conditions; only the first output is used
MAP_SIZE = 512
STYLE_DIM = 512
N_MLP = 6

TOTAL_ITERS = 2400
WARMUP_ITERS = 600  # nothing is optimised up to and including this iteration
DECODER_FINETUNE_ITER = 1800  # afterwards the outermost decoder layers are fine-tuned as well
RENDER_ITER = 2397  # iteration at which the results are exported
NUM_TRAIN_CAMERAS = 500  # per-subject cameras read from transforms_train.json for inversion
NUM_TEST_CAMERAS = 600  # per-subject cameras read for the driving video
TRAIN_CAMERA = 250  # fixed camera used while optimising

LR = 0.0025 * 0.15
LR_SMALL = 0.0025 * 0.1
L1_W, VGG_W, SSIM_W, OFFSET_REG_W = 5.0, 0.06, 0.2, 1.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paths.yaml", help="path config (see README)")
    parser.add_argument("--ckpt", required=True, help="trained generator checkpoint (from train.py)")
    parser.add_argument("--out_dir", default=None, help="defaults to <output_root>/test_itw")
    parser.add_argument("--subjects", nargs="+", default=ITW_SUBJECTS[:4], help="names of the videos to invert")
    parser.add_argument("--inversion_frames", type=int, nargs="+", default=None,
                        help="video frames used for inversion (required for videos not in the paper's set; "
                             "the first one is encoded)")
    parser.add_argument("--num_inversion_frames", type=int, default=3, choices=range(1, 6),
                        help="number of video frames used for the inversion")
    parser.add_argument("--render_video", action="store_true",
                        help="render a video over the last 600 video frames instead of 16 still images")
    parser.add_argument("--drive_expressions", action="store_true",
                        help="with --render_video: drive the avatar with the tracked expressions of the video")
    parser.add_argument("--frontal_only", action="store_true",
                        help="with --render_video --drive_expressions: do not rotate the head")
    parser.add_argument("--tv_uv_reg", action="store_true", help="add a total-variation loss on UV renderings")
    return parser.parse_args()


def load_uv_numpy(uv_dir):
    """Barycentric UV layout as float32 numpy arrays (used to rasterise meshes into UV maps)."""
    uvs = np.load(os.path.join(uv_dir, "uvs.npy"))
    w = 1 - uvs[0, ..., 0] - uvs[0, ..., 1]
    return (uvs.astype(np.float32), w[..., None].astype(np.float32),
            np.load(os.path.join(uv_dir, "vertex_ids.npy")), np.load(os.path.join(uv_dir, "verts_to_uv_index.npy")))


def mesh_to_uv_map(mesh_path, uv_np):
    """Rasterise the vertex positions of a FLAME mesh into a (H, W, 3) float tensor in UV space."""
    uv, w, verts_ids, verts_to_uv_index = uv_np
    v = trimesh.load(mesh_path, process=False).vertices[verts_to_uv_index][verts_ids]
    return torch.tensor(v[:, :, 0, :] * uv[0, ..., 0:1] + v[:, :, 1, :] * uv[0, ..., 1:] + v[:, :, 2, :] * w).float()


def to_batch(hwc, device):
    """(H, W, C) -> (BATCH, C, H, W) on ``device``."""
    return hwc.to(device).unsqueeze(0).repeat((BATCH, 1, 1, 1)).permute((0, 3, 1, 2))


def load_flame_params(path, device):
    params = np.load(path)
    expression = np.concatenate((params["expr"][0:1, 0:50], params["jaw_pose"][0:1], params["eyes_pose"][0:1]), axis=-1)
    pose = np.concatenate((params["rotation"][0:1], params["translation"][0:1]), axis=-1)
    return FlameParams(shape=torch.tensor(params["shape"][None][:, 0:100]).float().to(device),
                       expression=torch.tensor(expression).float().to(device),
                       scale=torch.ones((1, 1), dtype=torch.float32).to(device),
                       pose=torch.tensor(pose).float().to(device),
                       neck_pose=torch.tensor(params["neck_pose"][0:1]).float().to(device))


def load_rgba(path, device, background):
    """Video frame composited over ``background`` (3,), plus its (3, H, W) alpha."""
    rgba = cv2.imread(path, -1)
    image = transforms.ToTensor()(Image.fromarray(cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2RGB)))
    alpha = torch.tensor(rgba[:, :, 3:] / 255).repeat((1, 1, 3)).float().permute(2, 0, 1).to(device)
    return image.to(device) * alpha + background.unsqueeze(-1).unsqueeze(-1) * (1 - alpha), alpha


class Subject:
    """File locations of one in-the-wild subject."""

    def __init__(self, paths, name):
        self.name = name
        self.export_dir = os.path.join(paths.itw_vhap_export, name + VHAP_EXPORT_SUFFIX)
        self.frames_dir = os.path.join(paths.itw_frames, name, "images")
        self.mesh_dir = os.path.join(paths.itw_smirk_meshes, name)
        self.texture_dir = os.path.join(paths.itw_textures, name)

    def frame(self, f):
        return os.path.join(self.frames_dir, f"{f:05d}.png")

    def mesh(self, f):
        return os.path.join(self.mesh_dir, f"{f:05d}.obj")

    def flame(self, f):
        return os.path.join(self.export_dir, "flame_param", f"{f:05d}.npz")

    def texture(self, f, suffix):
        return os.path.join(self.texture_dir, f"tex_can_new_{f:05d}{suffix}")

    @property
    def transforms_file(self):
        return os.path.join(self.export_dir, "transforms_train.json")


class Conditioner:
    """Builds the generator's expression and texture conditions for a subject's frames."""

    def __init__(self, paths, device):
        self.device = device
        self.uv_np = load_uv_numpy(paths.uv_assets)
        mouth = cv2.imread(os.path.join(paths.uv_masks, "Background_Mouth.png"), -1)[:, :, :3] / 255
        self.mouth_mask = torch.tensor(mouth.astype(np.uint8))
        self.expression_mask = load_uv_mask(os.path.join(paths.uv_masks, "Background_Exp_Map_2.png"), BATCH).to(device)

    def expression(self, subject, f):
        """SMIRK mesh offsets from the subject's frame-0 mesh, plus the texture-gradient map of frame ``f``."""
        offsets = (to_batch(mesh_to_uv_map(subject.mesh(f), self.uv_np), self.device)
                   - to_batch(mesh_to_uv_map(subject.mesh(0), self.uv_np), self.device)) * self.expression_mask
        grad = self.mouth_mask * torch.tensor(np.load(subject.texture(f, "_grad.npy")).astype(np.float32))
        return torch.cat((offsets, to_batch(grad, self.device)), dim=1).to(self.device)

    def texture(self, subject, f):
        """Fitted UV texture of frame ``f`` in [-1, 1] with the mouth interior blanked."""
        rgb = cv2.cvtColor(cv2.imread(subject.texture(f, ".png"), -1)[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2RGB)
        tex = -1 + transforms.ToTensor()(np.asarray(rgb) * (1 - self.mouth_mask.cpu().numpy())).permute(1, 2, 0) * 2
        tex = to_batch(tex, self.device)
        return torch.cat((tex, 0.0 * tex), dim=1).to(self.device)


class ColorTransforms(torch.nn.Module):
    """Per-subject affine colour transform ``rgb @ M + b`` absorbing camera/lighting differences."""

    def __init__(self, num_subjects, device):
        super().__init__()
        self.col0 = torch.nn.Parameter(torch.tensor([[1, 0, 0]]).repeat(num_subjects, 1).float().to(device))
        self.col1 = torch.nn.Parameter(torch.tensor([[0, 1, 0]]).repeat(num_subjects, 1).float().to(device))
        self.col2 = torch.nn.Parameter(torch.tensor([[0, 0, 1]]).repeat(num_subjects, 1).float().to(device))
        self.bias = torch.nn.Parameter(torch.zeros(num_subjects, 3).float().to(device))

    def matrix(self, k):
        cols = (self.col0[k:k + 1].t(), self.col1[k:k + 1].t(), self.col2[k:k + 1].t())
        return torch.cat(cols, dim=1).unsqueeze(0).unsqueeze(0)

    def forward(self, image, k):
        rgb = image.permute(1, 2, 0).unsqueeze(-2)
        out = torch.matmul(rgb, self.matrix(k)[:, :, 0:3, 0:3]) + self.bias[k:k + 1].unsqueeze(0).unsqueeze(0)
        return out[:, :, 0, :].permute(2, 0, 1)

    def white_background(self, k):
        """Background colour that maps to white under the transform."""
        return ((1.0 - self.bias[k:k + 1]) @ torch.inverse(self.matrix(k)[0, 0]))[0]


def inversion_frames(args, name):
    frames = args.inversion_frames or INVERSION_FRAMES.get(name)
    if frames is None:
        raise ValueError(f"No default inversion frames for '{name}'; pass --inversion_frames")
    return frames[:args.num_inversion_frames]


def set_trainable(generator, iteration, texture_features, shape, expression):
    requires_grad(generator, False)
    if iteration > DECODER_FINETUNE_ITER:
        for i in range(2):
            requires_grad(generator.comb_convs[i], True)
            requires_grad(generator.convs1[-2 + i], True)
        requires_grad(generator.iwt, True)
        for i, feature in enumerate(texture_features):
            feature.requires_grad = i < 3
    else:
        for feature in texture_features:
            feature.requires_grad = True
    shape.requires_grad = True
    expression.requires_grad = True


def invert(args, paths, generator, optimizer, color, style_latents, encoded, device):
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(0)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    vgg = VGGLoss(device)
    can_loc, texel_mask = load_canonical_position_map(paths.position_map, device)
    init = init_gaussians(can_loc, texel_mask, device)
    uv_layout = load_uv_layout(paths.uv_assets, device)
    conditioner = Conditioner(paths, device)
    mouth_reg_mask = (torch.tensor(cv2.imread(os.path.join(paths.uv_masks, "Background_Mouth_2.png"), -1)[:, :, :1] / 255)
                      .unsqueeze(0).repeat((1, 1, 1, 59)).permute((0, 3, 1, 2)).float().to(device))
    flame_model = FLAMEModule(1, device, paths.flame_assets)
    gaussians = GaussianModel()

    xs, ys = torch.meshgrid(torch.linspace(-1, 1, steps=MAP_SIZE), torch.linspace(-1, 1, steps=MAP_SIZE), indexing="ij")
    uv_colors = uv_coordinate_colors(torch.stack([ys, xs], dim=-1), texel_mask, device)

    for k, name in enumerate(tqdm(args.subjects)):
        subject = Subject(paths, name)
        frames = inversion_frames(args, name)
        style = style_latents[k]
        expression_features = [f.detach() for f in encoded[k][0]]
        texture_features = [torch.nn.Parameter(f) for f in encoded[k][1]]

        # FLAME parameters of the inversion frames; shape and expression are refined.
        tracked = [load_flame_params(subject.flame(f), device) for f in frames]
        shape = torch.nn.Parameter(torch.stack([p.shape for p in tracked], dim=0))
        expression = torch.nn.Parameter(torch.stack([p.expression for p in tracked], dim=0))
        neck_pose = torch.stack([p.neck_pose for p in tracked], dim=0)
        pose = torch.stack([p.pose for p in tracked], dim=0)
        scale = torch.ones((1, 1), dtype=torch.float32).to(device)

        # The original indexed the first subject's parameter groups when setting learning rates, so only
        # the first inverted subject uses the smaller rate for its features/shape/expression.
        subject_lr = LR_SMALL if k == 0 else LR
        subject_groups = [{"params": texture_features}, {"params": shape}, {"params": expression}]
        for group in subject_groups:
            optimizer.add_param_group(group)

        requires_grad(generator, False)
        with torch.no_grad():
            init_maps, _ = generator(torch.full((BATCH, STYLE_DIM), 1 / np.sqrt(512), device=device),
                                     conditioner.expression(subject, frames[0]), conditioner.texture(subject, frames[0]),
                                     randomize_noise=False)

        train_cameras = load_itw_cameras([subject.transforms_file], NUM_TRAIN_CAMERAS)
        test_cameras = load_itw_cameras([subject.transforms_file], NUM_TEST_CAMERAS)

        def to_gaussians(g_maps, flame):
            points, displacement = gaussian_maps_to_3d(g_maps[0:1], texel_mask, init, flame_model, flame, uv_layout)
            features_dc, features_rest = split_sh(points["colors"])
            gaussians.modify(points["xyz"], points["opacity"], points["scaling"], points["rotation"],
                             features_dc, features_rest)
            return points, displacement, features_dc, features_rest

        pbar = tqdm(range(TOTAL_ITERS), leave=False)
        for it in pbar:
            frame = random.choice(frames)
            background = torch.tensor([random.random(), random.random(), random.random()], dtype=torch.float32).to(device)
            if it <= WARMUP_ITERS:
                continue  # the random draws above are kept so the sampled frame sequence is unchanged

            real_img, alpha = load_rgba(subject.frame(frame), device, background)
            set_trainable(generator, it, texture_features, shape, expression)
            for group in optimizer.param_groups:
                group["lr"] = LR
            for group in optimizer.param_groups[2:5]:  # colour transform: col1, col2 and bias
                group["lr"] = LR_SMALL
            for group in subject_groups:
                group["lr"] = subject_lr

            i = frames.index(frame)
            flame = FlameParams(shape[i][0:1], expression[i][0:1], scale[0:1], pose[i][0:1], neck_pose[i])

            optimizer.zero_grad(set_to_none=True)
            g_maps, _ = generator(style, expression_features, texture_features, randomize_noise=False,
                                  optimize_latent=True)
            points, displacement, features_dc, features_rest = to_gaussians(g_maps, flame)

            fake_img = render(train_cameras[TRAIN_CAMERA], gaussians, background)["render"]
            fake_img = torch.where(alpha >= 0.75, color(fake_img, k), fake_img)

            # Keep the geometry channels close to the un-personalised prediction, especially in the mouth.
            map_diff = init_maps[0:1, 0:11].detach() - g_maps[0:1, 0:11]
            map_reg = 5 * torch.mean(map_diff ** 2) + 1000 * torch.mean((map_diff * mouth_reg_mask[:, 0:11]) ** 2)
            l1_loss = torch.mean(torch.abs(fake_img - real_img)) * L1_W
            loss = (l1_loss
                    + vgg(-1 + 2 * fake_img.unsqueeze(0), -1 + 2 * real_img.unsqueeze(0)) * VGG_W * 2
                    + torch.mean(displacement ** 2) * OFFSET_REG_W
                    + (1.0 - ssim(fake_img, real_img)) * SSIM_W
                    + map_reg)
            if args.tv_uv_reg:
                uv_render = render(train_cameras[TRAIN_CAMERA], gaussians, torch.ones(3, device=device),
                                   override_color=uv_colors[0].detach(),
                                   override_opacity=torch.sigmoid(points["opacity"]))["render"]
                loss = loss + uv_tv_loss(uv_render.unsqueeze(0)) * 40
            loss.backward()
            optimizer.step()
            pbar.set_postfix(l1=l1_loss.item())

            if it == RENDER_ITER:
                export_results(args, subject, k, frames, generator, style, texture_features, conditioner, color,
                               gaussians, points, features_dc, features_rest, g_maps, to_gaussians,
                               train_cameras, test_cameras, device)


@torch.no_grad()
def export_results(args, subject, k, frames, generator, style, texture_features, conditioner, color, gaussians,
                   points, features_dc, features_rest, g_maps, to_gaussians, train_cameras, test_cameras, device):
    """Save a coloured point cloud and renderings (or a video) of the inverted avatar."""
    print("Colour transform:", color.matrix(k), color.bias[k:k + 1])
    prefix = os.path.join(args.out_dir, f"Test_{subject.name}")
    num_views = NUM_TEST_CAMERAS if args.render_video else 16
    video = (cv2.VideoWriter(prefix + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (512 * 2, 512))
             if args.render_video else None)

    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
    dirs = gaussians.get_xyz - train_cameras[0].camera_center.repeat(gaussians.get_features.shape[0], 1)
    rgb = torch.clamp_min(eval_sh(gaussians.active_sh_degree, shs_view, dirs / dirs.norm(dim=1, keepdim=True)) + 0.5, 0.0)
    write_pointcloud(prefix + "_.ply", gaussians.get_xyz.cpu().numpy().reshape(-1, 3),
                     (rgb.cpu().numpy().reshape(-1, 3) * 255).astype(np.uint8))

    background = color.white_background(k)
    white = torch.ones(3, dtype=torch.float32, device=device)
    # Driving frames are taken from the end of the video.
    driving_offset = len(glob.glob(os.path.join(subject.export_dir, "images", "*.png"))) - 601
    for view in range(num_views):
        driving_frame = driving_offset + view
        if not args.render_video:
            image_path = subject.frame(frames[0])
        elif args.drive_expressions:
            image_path = subject.frame(driving_frame)
        else:
            image_path = subject.frame(random.choice(frames))
        if not os.path.exists(image_path):
            continue
        real_img, _ = load_rgba(image_path, device, white)

        yaw = -0.96 + view * (2 * 0.96 / num_views)
        if args.render_video and args.drive_expressions:
            drive_maps, _ = generator(style, conditioner.expression(subject, driving_frame), texture_features,
                                      randomize_noise=False, optimize_latent=True, drive=True)
            points, _, features_dc, features_rest = to_gaussians(
                drive_maps, load_flame_params(subject.flame(driving_frame), device))
            if not args.frontal_only:
                rotate_gaussians(gaussians, points, features_dc, features_rest, yaw, device)
            camera = test_cameras[view]
        elif args.render_video:
            rotate_gaussians(gaussians, points, features_dc, features_rest, yaw, device)
            camera = train_cameras[0]
        else:
            camera = train_cameras[0]
        fake_img = color(render(camera, gaussians, background)["render"], k)

        color_map = F.interpolate(g_maps[0:1, SH][:, 0:3], size=real_img.shape[1:], mode="bilinear",
                                  align_corners=False)
        utils.save_image(torch.cat([real_img.unsqueeze(0), fake_img.unsqueeze(0).clamp(0, 1), color_map], dim=3),
                         os.path.join(args.out_dir, f"gaussian_maps_test_{subject.name}_{view:05d}.png"))
        if video is not None:
            frame = torch.cat([real_img, fake_img.clamp(0, 1)], dim=2)
            video.write(cv2.cvtColor((frame.permute((1, 2, 0)).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if video is not None:
        video.release()


def main():
    args = parse_args()
    paths = load_paths(args.config)
    args.out_dir = args.out_dir or os.path.join(paths.output_root, "test_itw")
    device = "cuda"

    seed_everything(0)
    generator = GaussianMapGeneratorInversion(inp_size=MAP_SIZE, inp_ch=3 + 3, out_ch=59, out_size=MAP_SIZE,
                                              style_dim=STYLE_DIM, n_mlp=N_MLP).to(device)
    print("load model:", args.ckpt)
    generator.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage)["g"], strict=False)

    # Encode the conditions of every subject with the trained (not yet fine-tuned) generator.
    conditioner = Conditioner(paths, device)
    style_latents, encoded = [], []
    with torch.no_grad():
        for name in args.subjects:
            subject = Subject(paths, name)
            frame = inversion_frames(args, name)[0]
            style_latents.append(generator.map_latent(torch.full((BATCH, STYLE_DIM), 1 / np.sqrt(512), device=device)))
            encoded.append(generator.encode_conditions(conditioner.expression(subject, frame),
                                                       conditioner.texture(subject, frame)))

    color = ColorTransforms(len(args.subjects), device)
    optimizer = optim.Adam([{"params": generator.parameters()}, {"params": color.col0}, {"params": color.col1},
                            {"params": color.col2}, {"params": color.bias}], lr=LR)
    print(f"Number of parameters: {sum(p.numel() for p in generator.parameters())}")
    invert(args, paths, generator, optimizer, color, style_latents, encoded, device)


if __name__ == "__main__":
    main()
