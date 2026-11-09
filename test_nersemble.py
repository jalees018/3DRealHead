"""Test-time inversion and animation on held-out NeRSemble subjects.

For every test subject the pretrained generator is conditioned on the subject's fitted UV texture
and on the expression map of an inversion frame. Two modes are supported:

* default: feed-forward prediction, then render the 16 calibrated views (or a turntable /
  expression-driven video with ``--render_video``).
* ``--optimize_latent``: per-subject inversion against ``--num_invert`` views of the inversion
  frame. A rigid head transform is optimised first, then the texture-encoder features and finally a
  few decoder layers. The optimised avatar is then rendered as above.

Note that the generator (and the optimiser state) is shared across subjects, so with
``--optimize_latent`` each subject starts from the weights fine-tuned on the previous one.
"""
import argparse
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch import optim
from torchvision import transforms, utils
from tqdm import tqdm

from avatar.cameras import NERSEMBLE_IMAGE_SIZE, load_nersemble_cameras, nersemble_camera_index
from avatar.config import load_paths
from avatar.geometry import (SH, FlameParams, gaussian_maps_to_3d, init_gaussians, load_canonical_position_map,
                             load_uv_layout, quaternion_multiply, rotate_gaussians, rotation_matrix_to_quaternion,
                             split_sh)
from avatar.losses import VGGLoss, uv_coordinate_colors, uv_tv_loss
from avatar.subjects import NERSEMBLE_CAMERAS, NERSEMBLE_SEQUENCES, NERSEMBLE_SUBJECTS
from avatar.utils import load_uv_mask, requires_grad, seed_everything, to_uint8_bgr, write_pointcloud
from flame import FLAMEModule
from gaussian_splatting import GaussianModel, render
from gaussian_splatting.loss_utils import ssim
from gaussian_splatting.sh_utils import eval_sh
from networks.generator import GaussianMapGeneratorInversion

PAIR = 2  # the generator is always run on a duplicated pair of conditioning maps
STYLE_VALUE = 1 / np.sqrt(512)  # constant style input (identity comes from the texture conditioning)
NEUTRAL_SEQUENCE, NEUTRAL_FRAME = "EXP-5-mouth", 0
INVERSION_CAMERA_SEQUENCE = NERSEMBLE_SEQUENCES.index(NEUTRAL_SEQUENCE)
FRONT_VIEW = 8
INVERSION_VIEWS = [0, 8, 15]
SEG_REMOVE_LABEL = 22

# Inversion schedule (per subject).
INVERSION_ITERS = 2800
EVAL_ITER = 2798  # evaluation happens after this step; the remaining step still updates the generator
RIGID_ONLY_ITERS = 400  # up to here only the rigid head transform is optimised
DECODER_FINETUNE_ITER = 1800  # after this, the last decoder layers are fine-tuned as well
LR = 0.0025 * 4 / 5
FINETUNE_LR = 0.0025 * 0.15
RIGID_FINETUNE_LR = 0.0025 * 0.1

L1_W, VGG_W, OFFSET_REG_W, SSIM_W, TV_UV_W = 5.0, 0.06 * 2, 1.0, 0.2, 40.0
TURNTABLE_YAW = 1.22  # turntable sweeps yaw in [-1.22, 1.22] rad


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paths.yaml", help="path config (see README)")
    parser.add_argument("--ckpt", required=True, help="generator checkpoint produced by train.py")
    parser.add_argument("--out_dir", default=None, help="defaults to <output_root>/test_nersemble")
    parser.add_argument("--subjects", type=int, nargs="+", default=NERSEMBLE_SUBJECTS[216:220],
                        help="NeRSemble subject ids to test (default: the held-out subjects of the paper)")
    parser.add_argument("--optimize_latent", action="store_true", help="run per-subject inversion")
    parser.add_argument("--num_invert", type=int, default=3, choices=[1, 3],
                        help="views used for inversion: 3 = random of (0, 8, 15), 1 = frontal view only")
    parser.add_argument("--tv_uv_reg", action="store_true", help="add a total-variation loss on UV renderings")
    parser.add_argument("--inversion_seq", default="EXP-9-jaw-2", help="sequence of the inversion frame")
    parser.add_argument("--inversion_frame", type=int, default=0, help="inversion frame number")
    parser.add_argument("--render_video", action="store_true",
                        help="render a video (turntable, or expression-driven with --drive_expressions)")
    parser.add_argument("--drive_expressions", action="store_true",
                        help="animate with the expressions of --driving_seq starting at --driving_frame")
    parser.add_argument("--frontal_only", action="store_true", help="do not rotate the head in driven videos")
    parser.add_argument("--driving_seq", default="EXP-5-mouth", choices=NERSEMBLE_SEQUENCES)
    parser.add_argument("--driving_frame", type=int, default=100, help="first driving frame")
    return parser.parse_args()


class NeRSembleTestData:
    """Loads images, FLAME parameters and generator conditioning maps of NeRSemble frames."""

    def __init__(self, paths, device):
        self.paths = paths
        self.device = device
        self.to_tensor = transforms.ToTensor()

        uvs = np.load(os.path.join(paths.uv_assets, "uvs.npy"))
        self.uv = uvs.astype(np.float32)
        self.w = (1 - uvs[0, ..., 0] - uvs[0, ..., 1])[..., None].astype(np.float32)
        self.verts_ids = np.load(os.path.join(paths.uv_assets, "vertex_ids.npy"))
        self.verts_to_uv_index = np.load(os.path.join(paths.uv_assets, "verts_to_uv_index.npy"))

        mouth_file = os.path.join(paths.uv_masks, "Background_Mouth.png")
        self.mouth_mask = torch.tensor((cv2.imread(mouth_file, -1)[:, :, :3] / 255).astype(np.uint8))
        self.expression_mask = load_uv_mask(os.path.join(paths.uv_masks, "Background_Exp_Map_2.png"), PAIR).to(device)

    def _export(self, subject, sequence, folder, name):
        return os.path.join(self.paths.nersemble_export, f"{subject:03d}", sequence, folder, name)

    def num_frames(self, subject, sequence):
        return len(glob(self._export(subject, sequence, "images", f"*_{FRONT_VIEW:02d}.png"))) - 1

    def _pair(self, x):
        """(H, W, C) -> (PAIR, C, H, W) on the device."""
        return x.to(self.device).unsqueeze(0).repeat((PAIR, 1, 1, 1)).permute((0, 3, 1, 2))

    def _mesh_uv_map(self, subject, sequence, frame):
        """SMIRK mesh vertices interpolated into UV space, (H, W, 3)."""
        mesh = trimesh.load(os.path.join(self.paths.nersemble_smirk_meshes, f"{subject:03d}", sequence,
                                         f"{frame:05d}.obj"), process=False)
        v = mesh.vertices[self.verts_to_uv_index][self.verts_ids]
        return torch.tensor(v[:, :, 0, :] * self.uv[0, ..., 0:1] + v[:, :, 1, :] * self.uv[0, ..., 1:]
                            + v[:, :, 2, :] * self.w).float()

    def expression_condition(self, subject, sequence, frame):
        """Expression map (offset from the neutral mesh) + texture-gradient map, (PAIR, 6, H, W)."""
        offsets = (self._pair(self._mesh_uv_map(subject, sequence, frame))
                   - self._pair(self._mesh_uv_map(subject, NEUTRAL_SEQUENCE, NEUTRAL_FRAME))) * self.expression_mask
        grad = np.load(os.path.join(self.paths.nersemble_test_textures, f"{subject:03d}", sequence,
                                    f"tex_can_new_{frame:05d}_grad.npy")).astype(np.float32)
        grad = self.mouth_mask * torch.tensor(grad)
        return torch.cat((offsets, self._pair(grad)), dim=1).to(self.device)

    def texture_condition(self, subject):
        """Fitted UV texture in [-1, 1] with the mouth interior blanked, (PAIR, 6, H, W)."""
        texture = cv2.imread(os.path.join(self.paths.nersemble_test_textures, f"{subject:03d}",
                                          f"tex_can_new_{subject:03d}.png"), -1)[:, :, :3]
        texture = cv2.cvtColor(texture.astype(np.uint8), cv2.COLOR_BGR2RGB) * (1 - self.mouth_mask.numpy())
        texture = self._pair(-1 + self.to_tensor(texture).permute(1, 2, 0) * 2)
        return torch.cat((texture, 0.0 * texture), dim=1).to(self.device)

    def flame_params(self, subject, sequence, frame):
        params = np.load(self._export(subject, sequence, "flame_param", f"{frame:05d}.npz"))

        def to_device(x):
            return torch.tensor(x).float().to(self.device)

        expression = np.concatenate((params["expr"][0:1, 0:50], params["jaw_pose"][0:1], params["eyes_pose"][0:1]),
                                    axis=-1)
        pose = np.concatenate((params["rotation"][0:1], params["translation"][0:1]), axis=-1)
        return FlameParams(to_device(params["shape"][None][:, 0:100]), to_device(expression),
                           torch.ones((1, 1), dtype=torch.float32).to(self.device), to_device(pose),
                           to_device(params["neck_pose"][0:1]))

    def image(self, subject, sequence, frame, view, background):
        """Colour-corrected foreground image composited over ``background`` (3,), shape (3, H, W)."""
        name = f"{frame:05d}_{view:02d}.png"
        image = cv2.imread(self._export(subject, sequence, "images", name), -1)[:, :, :3]
        alpha = cv2.imread(self._export(subject, sequence, "fg_masks", name))[:, :, :3]
        alpha[cv2.imread(self._export(subject, sequence, "segs", name), -1) == SEG_REMOVE_LABEL, :] = 0
        image = cv2.cvtColor((image * (alpha / 255)).astype(np.uint8), cv2.COLOR_BGR2RGB)

        color_correction = torch.tensor(np.load(os.path.join(
            self.paths.nersemble_color_correction, f"{subject:03d}", NERSEMBLE_CAMERAS[view] + ".npy"))).unsqueeze(0).float()
        image = (torch.matmul(self.to_tensor(image).permute(1, 2, 0).unsqueeze(-2), color_correction[:, 0:3, 0:3])
                 + color_correction[:, 0:3, 3].unsqueeze(0).unsqueeze(0))[:, :, 0, :].permute(2, 0, 1)
        alpha = torch.tensor(alpha / 255).float().permute(2, 0, 1).to(self.device)
        return image.to(self.device) * alpha + background.unsqueeze(-1).unsqueeze(-1) * (1 - alpha)


class Avatar:
    """Turns generator outputs into posed 3D Gaussians of one template layout."""

    def __init__(self, paths, device):
        can_loc, self.texel_mask = load_canonical_position_map(paths.position_map, device)
        self.init = init_gaussians(can_loc, self.texel_mask, device)
        self.uv_layout = load_uv_layout(paths.uv_assets, device)
        self.flame_model = FLAMEModule(1, device, paths.flame_assets)
        self.gaussians = GaussianModel()

    def points(self, g_maps, flame):
        return gaussian_maps_to_3d(g_maps[0:1], self.texel_mask, self.init, self.flame_model, flame, self.uv_layout)

    def set(self, points, rot=None, trans=None):
        """Write ``points`` into the Gaussian model, optionally applying a rigid transform."""
        features_dc, features_rest = split_sh(points["colors"])
        xyz, rotation = points["xyz"], points["rotation"]
        if rot is not None:
            quat = rotation_matrix_to_quaternion(rot.unsqueeze(0))
            xyz = torch.matmul(rot, xyz.t()).t() + trans
            rotation = quaternion_multiply(F.normalize(quat), rotation)
        self.gaussians.modify(xyz, points["opacity"], points["scaling"], rotation, features_dc, features_rest)
        return features_dc, features_rest

    def turntable(self, points, features_dc, features_rest, step, num_steps, device):
        yaw = -TURNTABLE_YAW + step * (2 * TURNTABLE_YAW / num_steps)
        rotate_gaussians(self.gaussians, points, features_dc, features_rest, yaw, device)


def rigid_transform(rot_x, rot_y, trans, k):
    """Rotation from two (unnormalised) basis vectors, and translation, of subject ``k``."""
    x, y = rot_x[k:k + 1], rot_y[k:k + 1]
    return torch.cat((x.t(), y.t(), torch.cross(x, y, dim=1).t()), dim=1), trans[k:k + 1]


def save_colored_pointcloud(path, gaussians, camera):
    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
    dirs = gaussians.get_xyz - camera.camera_center.repeat(gaussians.get_features.shape[0], 1)
    colors = torch.clamp_min(eval_sh(gaussians.active_sh_degree, shs_view, dirs / dirs.norm(dim=1, keepdim=True)) + 0.5,
                             0.0)
    write_pointcloud(path, gaussians.get_xyz.detach().cpu().numpy().reshape(-1, 3),
                     (colors.detach().cpu().numpy().reshape(-1, 3) * 255).astype(np.uint8))


def save_comparison(path, real, fake, g_maps):
    color_map = F.interpolate(g_maps[0:1, SH][:, 0:3], size=real.shape[1:], mode="bilinear", align_corners=False)
    utils.save_image(torch.cat([real.unsqueeze(0), fake.unsqueeze(0).clamp(0, 1), color_map], dim=3), path)


class Tester:
    def __init__(self, args, paths, generator, cameras, device):
        self.args = args
        self.subjects = args.subjects
        self.generator = generator
        self.cameras = cameras
        self.device = device
        self.data = NeRSembleTestData(paths, device)
        self.avatar = Avatar(paths, device)
        self.style = torch.full((PAIR, 512), STYLE_VALUE, device=device)
        self.white = torch.ones(3, dtype=torch.float32, device=device)
        self.video_dir = os.path.join(args.out_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

    def driving_camera(self, idx):
        return self.cameras[nersemble_camera_index(idx, NERSEMBLE_SEQUENCES.index(self.args.driving_seq),
                                                   FRONT_VIEW)]

    def driving_condition(self, subject, frame):
        return (self.data.expression_condition(subject, self.args.driving_seq, frame),
                self.data.flame_params(subject, self.args.driving_seq, frame))

    def open_video(self, idx, width):
        if not self.args.render_video:
            return None
        path = os.path.join(self.video_dir, f"{self.subjects[idx]:03d}.mp4")
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, NERSEMBLE_IMAGE_SIZE[1]))

    def finish(self, idx, video):
        subject = self.subjects[idx]
        save_colored_pointcloud(os.path.join(self.args.out_dir, f"{subject:03d}.ply"), self.avatar.gaussians,
                                self.cameras[nersemble_camera_index(idx, 0, FRONT_VIEW)])
        if video is not None:
            video.release()

    def output_path(self, idx, step):
        return os.path.join(self.args.out_dir, f"{self.subjects[idx]:03d}_{step:05d}.png")

    # ----------------------------------------------------------------------------------------------
    # Feed-forward prediction
    # ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def run_feedforward(self, idx):
        """Predict the avatar for the inversion frame and render views / a video."""
        args, avatar = self.args, self.avatar
        subject = self.subjects[idx]
        expression = self.data.expression_condition(subject, args.inversion_seq, args.inversion_frame)
        texture = self.data.texture_condition(subject)
        flame = self.data.flame_params(subject, args.inversion_seq, args.inversion_frame)

        g_maps, _ = self.generator(self.style, expression, texture, randomize_noise=False)
        points, _ = avatar.points(g_maps, flame)
        features_dc, features_rest = avatar.set(points)

        num_steps = (self.data.num_frames(subject, args.driving_seq)
                     if args.render_video else len(NERSEMBLE_CAMERAS))
        video = self.open_video(idx, NERSEMBLE_IMAGE_SIZE[0])
        for step in range(num_steps):
            view = FRONT_VIEW if args.render_video else step
            real = self.data.image(subject, args.inversion_seq, args.inversion_frame, view, self.white)
            if args.render_video:
                if args.drive_expressions:
                    expression, flame = self.driving_condition(subject, args.driving_frame + step)
                    g_driven, _ = self.generator(self.style, expression, texture, randomize_noise=False)
                    points, _ = avatar.points(g_driven, flame)
                    features_dc, features_rest = avatar.set(points)
                    if not args.frontal_only:
                        avatar.turntable(points, features_dc, features_rest, step, num_steps, self.device)
                else:
                    avatar.turntable(points, features_dc, features_rest, step, num_steps, self.device)
                camera = self.driving_camera(idx)
            else:
                camera = self.cameras[nersemble_camera_index(idx, 0, view)]
            fake = render(camera, avatar.gaussians, self.white)["render"]

            save_comparison(self.output_path(idx, step), real, fake, g_maps)
            if video is not None:
                video.write(to_uint8_bgr(fake.clamp(0, 1)))
        self.finish(idx, video)

    # ----------------------------------------------------------------------------------------------
    # Inversion
    # ----------------------------------------------------------------------------------------------
    def set_trainable(self, it, expression_feats, texture_feats, optimizer):
        """Inversion schedule: rigid transform -> texture-encoder features -> last decoder layers."""
        generator = self.generator
        requires_grad(generator, False)
        for p in expression_feats:
            p.requires_grad = False
        if it <= RIGID_ONLY_ITERS:
            for p in texture_feats:
                p.requires_grad = False
            return

        if it > DECODER_FINETUNE_ITER:
            for i in range(2):
                requires_grad(generator.comb_convs[i], True)
                requires_grad(generator.convs1[-2 + i], True)
            requires_grad(generator.iwt, True)
            for i, p in enumerate(texture_feats):
                p.requires_grad = i < 3
        else:
            for p in texture_feats:
                p.requires_grad = True

        for group in optimizer.param_groups:
            group["lr"] = FINETUNE_LR
        for group in optimizer.param_groups[1:4]:  # rot_x, rot_y, trans
            group["lr"] = RIGID_FINETUNE_LR

    def run_inversion(self, idx, optimizer, layer_latents, rigid, encoder_feats, vgg, mouth_mask_maps,
                      uv_colors):
        args, avatar, data, generator = self.args, self.avatar, self.data, self.generator
        subject = self.subjects[idx]
        rot_x, rot_y, trans = rigid

        expression_feats = torch.nn.ParameterList([torch.nn.Parameter(f) for f in encoder_feats[idx][0]])
        texture_feats = torch.nn.ParameterList([torch.nn.Parameter(f) for f in encoder_feats[idx][1]])
        optimizer.add_param_group({"params": expression_feats, "lr": LR})
        optimizer.add_param_group({"params": texture_feats, "lr": LR})
        requires_grad(generator, False)

        flame = data.flame_params(subject, args.inversion_seq, args.inversion_frame)
        with torch.no_grad():
            expression = data.expression_condition(subject, args.inversion_seq, args.inversion_frame)
            g_maps_init, _ = generator(self.style, expression, data.texture_condition(subject), randomize_noise=False)

        pbar = tqdm(range(INVERSION_ITERS))
        for it in pbar:
            view = random.choice(INVERSION_VIEWS) if args.num_invert == 3 else FRONT_VIEW
            background = torch.tensor([random.random(), random.random(), random.random()],
                                      dtype=torch.float32).to(self.device)
            real = data.image(subject, args.inversion_seq, args.inversion_frame, view, background)
            self.set_trainable(it, expression_feats, texture_feats, optimizer)

            optimizer.zero_grad(set_to_none=True)
            g_maps, _ = generator(layer_latents, expression_feats, texture_feats, randomize_noise=False,
                                  optimize_latent=True)
            points, displacement = avatar.points(g_maps, flame)
            rot, tr = rigid_transform(rot_x, rot_y, trans, idx)
            features_dc, features_rest = avatar.set(points, rot, tr)
            camera = self.cameras[nersemble_camera_index(idx, INVERSION_CAMERA_SEQUENCE, view)]
            fake = render(camera, avatar.gaussians, background)["render"]

            l1_loss = torch.mean(torch.abs(fake - real)) * L1_W
            loss = (l1_loss
                    + vgg(-1 + 2 * fake.unsqueeze(0), -1 + 2 * real.unsqueeze(0)) * VGG_W
                    + torch.mean(displacement ** 2) * OFFSET_REG_W
                    + (1.0 - ssim(fake, real)) * SSIM_W)
            # Keep the two basis vectors orthonormal.
            loss = loss + (100 * torch.mean(torch.abs(torch.sum(rot_x[idx:idx + 1] * rot_y[idx:idx + 1], dim=1)))
                           + 100 * torch.mean((torch.norm(rot_x[idx:idx + 1]) - 1) ** 2)
                           + 100 * torch.mean((torch.norm(rot_y[idx:idx + 1]) - 1) ** 2))
            # Stay close to the feed-forward prediction (geometry channels strongly inside the mouth).
            loss = loss + (10 * torch.mean(((g_maps_init[0:1] - g_maps[0:1]) * (1 - mouth_mask_maps)) ** 2)
                           + 100 * torch.mean(((g_maps_init[0:1, 0:11] - g_maps[0:1, 0:11])
                                               * mouth_mask_maps[:, 0:11]) ** 2))
            if args.tv_uv_reg:
                uv_render = render(camera, avatar.gaussians, self.white, override_color=uv_colors[0].detach(),
                                   override_opacity=torch.sigmoid(points["opacity"]))["render"]
                loss = loss + uv_tv_loss(uv_render.unsqueeze(0)) * TV_UV_W
            loss.backward()
            optimizer.step()
            pbar.set_postfix(l1=f"{l1_loss.item():.4f}")

            if it == EVAL_ITER:
                with torch.no_grad():
                    self.render_inverted(idx, layer_latents, texture_feats, points, features_dc,
                                         features_rest, g_maps)

    def render_inverted(self, idx, layer_latents, texture_feats, points, features_dc, features_rest, g_maps):
        args, avatar, data = self.args, self.avatar, self.data
        subject = self.subjects[idx]
        num_steps = data.num_frames(subject, args.driving_seq) if args.render_video else len(NERSEMBLE_CAMERAS)
        video = self.open_video(idx, 2 * NERSEMBLE_IMAGE_SIZE[0])
        ref_view = FRONT_VIEW
        for step in range(num_steps):
            driving_frame = args.driving_frame + step
            if not args.render_video:
                real = data.image(subject, args.inversion_seq, args.inversion_frame, step, self.white)
            elif args.drive_expressions:
                real = data.image(subject, args.driving_seq, driving_frame, FRONT_VIEW, self.white)
            else:
                if step % 10 == 0:
                    ref_view = random.choice(INVERSION_VIEWS) if args.num_invert == 3 else FRONT_VIEW
                real = data.image(subject, args.inversion_seq, args.inversion_frame, ref_view, self.white)

            if not args.render_video:
                camera = self.cameras[nersemble_camera_index(idx, 0, step)]
            else:
                # Driven / turntable renders use the predicted (un-transformed) Gaussians.
                if args.drive_expressions:
                    expression, flame = self.driving_condition(subject, driving_frame)
                    g_driven, _ = self.generator(layer_latents, expression, texture_feats, randomize_noise=False,
                                                 optimize_latent=True, drive=True)
                    points, _ = avatar.points(g_driven, flame)
                    features_dc, features_rest = avatar.set(points)
                    if not args.frontal_only:
                        avatar.turntable(points, features_dc, features_rest, step, num_steps, self.device)
                else:
                    avatar.turntable(points, features_dc, features_rest, step, num_steps, self.device)
                camera = self.driving_camera(idx)
            fake = render(camera, avatar.gaussians, self.white)["render"]

            save_comparison(self.output_path(idx, step), real, fake, g_maps)
            if video is not None:
                video.write(to_uint8_bgr(torch.cat([real, fake.clamp(0, 1)], dim=2)))
        self.finish(idx, video)

    def run(self, encoder_feats=None, layer_latents=None, optimizer=None, rigid=None):
        args, device = self.args, self.device
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        vgg = VGGLoss(device)

        mouth_file = os.path.join(self.data.paths.uv_masks, "Background_Mouth.png")
        mouth_mask_maps = (torch.tensor(cv2.imread(mouth_file, -1)[:, :, :1] / 255).unsqueeze(0)
                           .repeat((1, 1, 1, 59)).permute((0, 3, 1, 2)).float().to(device))
        xs, ys = torch.meshgrid(torch.linspace(-1, 1, steps=512), torch.linspace(-1, 1, steps=512), indexing="ij")
        uv_colors = uv_coordinate_colors(torch.stack([ys, xs], dim=-1), self.avatar.texel_mask, device)

        for idx in tqdm(range(len(self.subjects))):
            if args.optimize_latent:
                self.run_inversion(idx, optimizer, layer_latents, rigid, encoder_feats, vgg,
                                   mouth_mask_maps, uv_colors)
            else:
                requires_grad(self.generator, False)
                self.run_feedforward(idx)


def main():
    args = parse_args()
    paths = load_paths(args.config)
    args.out_dir = args.out_dir or os.path.join(paths.output_root, "test_nersemble")
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"

    seed_everything(0)
    cameras = load_nersemble_cameras(paths.nersemble_export, subjects=args.subjects)
    generator = GaussianMapGeneratorInversion(inp_size=512, inp_ch=3 + 3, out_ch=59, out_size=512, style_dim=512,
                                              n_mlp=6).to(device)
    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g"], strict=False)
    print(f"Number of parameters: {sum(p.numel() for p in generator.parameters())}")

    tester = Tester(args, paths, generator, cameras, device)
    if not args.optimize_latent:
        tester.run()
        return

    # Inversion starts from the encoder features of each subject's neutral frame and texture.
    subjects = args.subjects
    with torch.no_grad():
        layer_latents = generator.map_latent(torch.full((PAIR, 512), STYLE_VALUE, device=device))
        encoder_feats = [generator.encode_conditions(
            tester.data.expression_condition(subject, NEUTRAL_SEQUENCE, NEUTRAL_FRAME),
            tester.data.texture_condition(subject)) for subject in subjects]

    rot_x = torch.nn.Parameter(torch.tensor([[1, 0, 0]]).repeat(len(subjects), 1).float().to(device))
    rot_y = torch.nn.Parameter(torch.tensor([[0, 1, 0]]).repeat(len(subjects), 1).float().to(device))
    trans = torch.nn.Parameter(torch.zeros(len(subjects), 3).float().to(device))
    optimizer = optim.Adam([
        {"params": generator.parameters()},
        {"params": rot_x, "lr": LR},
        {"params": rot_y, "lr": LR},
        {"params": trans, "lr": LR * 0.2},
    ], lr=LR)
    tester.run(encoder_feats, layer_latents, optimizer, (rot_x, rot_y, trans))


if __name__ == "__main__":
    main()
