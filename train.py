"""Train the StyleUNet Gaussian-map generator on multi-view NeRSemble data.

The generator maps (expression UV map, texture-gradient map) and (fitted UV texture) to a 59-channel
UV map of 3D Gaussians. The Gaussians are anchored on the tracked FLAME mesh, posed with LBS and
rendered with 3DGS into the training cameras. Training is photometric (L1 + SSIM + VGG + mouth L1)
and adds a DINO-based projected discriminator after ``--gan_start`` iterations.
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from tqdm import tqdm

from avatar.cameras import NERSEMBLE_IMAGE_SIZE, load_nersemble_cameras, nersemble_camera_index
from avatar.config import load_paths
from avatar.dataset import NeRSembleDataset
from avatar.geometry import (FlameParams, gaussian_maps_to_3d, init_gaussians, load_canonical_position_map,
                             load_uv_layout, rotate_gaussians, split_sh, SH)
from avatar.identity import clip_identity_embeddings
from avatar.losses import VGGLoss, hinge_d_loss, hinge_g_loss
from avatar.subjects import NERSEMBLE_SUBJECTS
from avatar.utils import infinite_loader, load_uv_mask, requires_grad, seed_everything, to_uint8_bgr
from flame import FLAMEModule
from gaussian_splatting import GaussianModel, render
from gaussian_splatting.loss_utils import ssim
from networks.discriminator import ProjectedDiscriminator
from networks.generator import GaussianMapGenerator

GEOMETRY_WARMUP_ITERS = 4000  # Gaussians stay on the mesh surface with identity rotation before this
VGG_START_ITER = 100_000
MOUTH_LOSS_START_ITER = 150_000
LR_MILESTONES = ((320_000, 5), (400_000, 10), (480_000, 15))  # (iteration, divisor of the base lr)
BASE_LR = 0.0025 * 0.15
CROP_SIZE = 540  # random crop fed to the discriminator
SAMPLE_EVERY, CKPT_EVERY, VIDEO_EVERY = 1000, 10_000, 25_000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paths.yaml", help="path config (see README)")
    parser.add_argument("--out_dir", default=None, help="defaults to <output_root>/train")
    parser.add_argument("--ckpt", default=None, help="checkpoint to resume from (gaussian_maps_<iter>.pt)")
    parser.add_argument("--iter", type=int, default=8_000_000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num_train", type=int, default=200, help="number of NeRSemble training subjects")
    parser.add_argument("--lr", type=float, default=0.0025 * 0.25, help="initial generator learning rate")
    parser.add_argument("--gan_start", type=int, default=600_000, help="iteration at which the GAN loss starts")
    parser.add_argument("--d_every", type=int, default=10, help="discriminator update interval")
    parser.add_argument("--l1_w", type=float, default=5.0)
    parser.add_argument("--vgg_w", type=float, default=0.06)
    parser.add_argument("--ssim_w", type=float, default=0.2)
    parser.add_argument("--offset_reg_w", type=float, default=1.0, help="L2 penalty on predicted displacements")
    parser.add_argument("--gan_w", type=float, default=1.0)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=6)
    parser.add_argument("--map_size", type=int, default=512, help="UV resolution of input and output maps")
    return parser.parse_args()


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def render_turntable(path, cameras, gaussians, points, features_dc, features_rest, device, num_views=100):
    background = torch.zeros(3, dtype=torch.float32, device=device)
    video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, NERSEMBLE_IMAGE_SIZE)
    for view in range(num_views):
        yaw = -(np.pi / 2) + view * (np.pi / num_views)
        rotate_gaussians(gaussians, points, features_dc, features_rest, yaw, device)
        video.write(to_uint8_bgr(render(cameras[8], gaussians, background)["render"]))
    video.release()


def train(args, paths, loader, generator, discriminator, g_optim, d_optim, cameras, identity_latents, device):
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    sample_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    torch.manual_seed(0)
    loader = infinite_loader(loader)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    vgg = VGGLoss(device)
    discriminator.train()

    can_loc, texel_mask = load_canonical_position_map(paths.position_map, device)
    init = init_gaussians(can_loc, texel_mask, device)
    uv_layout = load_uv_layout(paths.uv_assets, device)
    expression_mask = load_uv_mask(os.path.join(paths.uv_masks, "Background_Exp_Map_2.png"), args.batch)
    flame_model = FLAMEModule(1, device, paths.flame_assets)
    gaussians = GaussianModel()

    # The generator's style input is a constant; identity enters through the texture conditioning.
    style = torch.full_like(identity_latents[0], 1 / np.sqrt(512))
    background = torch.zeros(3, dtype=torch.float32, device=device)

    gen_losses, disc_losses, gan_steps = [], [], []
    d_loss_val = float("nan")
    lr_reset_iter = args.start_iter if args.start_iter != 0 else args.gan_start

    pbar = tqdm(range(args.iter), initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    for idx in pbar:
        i = idx + args.start_iter
        use_gan = i >= args.gan_start

        requires_grad(generator, False)
        requires_grad(discriminator, use_gan)
        discriminator.dino.requires_grad_(False)

        if i > args.iter:
            print("Done!")
            break

        if i == lr_reset_iter:
            set_lr(g_optim, BASE_LR)
            set_lr(d_optim, BASE_LR)
        for milestone, divisor in LR_MILESTONES:
            if i == milestone:
                set_lr(g_optim, BASE_LR / divisor)

        (real_img, expression_map, texture, view, subject_idx, _, mouth_mask, alpha, texture_grad,
         shape, expression, scale, pose, neck_pose, sequence_idx) = next(loader)

        # Random crop offsets for the discriminator; drawn every iteration to keep the RNG stream fixed.
        crop_y = np.random.randint(260, size=args.batch)
        crop_x = np.random.randint(9, size=args.batch)

        flame = FlameParams(shape[:, 0, :].to(device), expression[:, 0, :].to(device), scale[:, 0, :].to(device),
                            pose[:, 0, :].to(device), neck_pose[:, 0, :].to(device))
        alpha = alpha.float().to(device).permute((0, 3, 1, 2)) / 255
        real_img = real_img.to(device) * alpha + background.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * (1 - alpha)
        mouth_mask = mouth_mask.permute((0, 3, 1, 2)).to(device)

        cond_expression = torch.cat((expression_map.permute((0, 3, 1, 2)) * expression_mask,
                                     texture_grad.permute((0, 3, 1, 2))), dim=1).to(device)
        cond_texture = texture.permute((0, 3, 1, 2))
        cond_texture = torch.cat((cond_texture, 0.0 * cond_texture), dim=1).to(device)

        def to_gaussians(g_maps, b):
            flame_b = FlameParams(*(p[b:b + 1] for p in flame))
            return gaussian_maps_to_3d(g_maps[b:b + 1], texel_mask, init, flame_model, flame_b, uv_layout,
                                       freeze_geometry=i < GEOMETRY_WARMUP_ITERS)

        def render_view(points, b):
            features_dc, features_rest = split_sh(points["colors"])
            gaussians.modify(points["xyz"], points["opacity"], points["scaling"], points["rotation"],
                             features_dc, features_rest)
            camera = cameras[nersemble_camera_index(subject_idx[b], sequence_idx[b], view[b])]
            return render(camera, gaussians, background)["render"], features_dc, features_rest

        def disc(img, b):
            latent = identity_latents[subject_idx[b]][0:2].detach()
            return discriminator(-1 + 2 * torch.cat((img, img), dim=0), latent)

        if real_img.mean() < 0.025:  # skip (almost) empty frames
            continue

        if use_gan and i % args.d_every == 0:
            g_maps, _ = generator(style, cond_expression, cond_texture, randomize_noise=False)
            points, _ = to_gaussians(g_maps, 0)
            fake_img, _, _ = render_view(points, 0)
            fake_img = TF.crop(fake_img, crop_y[0], crop_x[0], CROP_SIZE, CROP_SIZE)
            real_crop = TF.crop(real_img[0:1], crop_y[0], crop_x[0], CROP_SIZE, CROP_SIZE)

            fake_pred, _ = disc(fake_img.unsqueeze(0), 0)
            real_pred, _ = disc(real_crop, 0)
            d_loss = hinge_d_loss(real_pred, fake_pred)
            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()
            d_loss_val = d_loss.item()

        requires_grad(generator, True)
        requires_grad(discriminator, False)
        discriminator.dino.requires_grad_(False)

        g_maps, _ = generator(style, cond_expression, cond_texture, randomize_noise=False)
        total_loss = 0
        g_optim.zero_grad(set_to_none=True)
        for b in range(args.batch):
            points, displacement = to_gaussians(g_maps, b)
            fake_img, features_dc, features_rest = render_view(points, b)

            target, target_mouth = real_img, mouth_mask
            if use_gan:
                target = TF.crop(real_img, crop_y[b], crop_x[b], CROP_SIZE, CROP_SIZE)
                target_mouth = TF.crop(mouth_mask, crop_y[b], crop_x[b], CROP_SIZE, CROP_SIZE)
                fake_img = TF.crop(fake_img, crop_y[b], crop_x[b], CROP_SIZE, CROP_SIZE)

                fake_pred, fake_feat = disc(fake_img.unsqueeze(0), b)
                _, real_feat = disc(target[b:b + 1], b)
                feat_loss = sum(torch.mean(torch.abs(fake_feat[k] - real_feat[k].detach())) for k in "0123")
                g_loss = hinge_g_loss(fake_pred) * args.gan_w + feat_loss * args.gan_w * 2

            l1_loss = torch.mean(torch.abs(fake_img - target[b])) * args.l1_w
            if i < VGG_START_ITER:
                vgg_loss = torch.zeros((), device=device)
            else:
                vgg_loss = vgg(-1 + 2 * fake_img.unsqueeze(0), -1 + 2 * target[b:b + 1]) * args.vgg_w * 2
            reg_loss = torch.mean(displacement ** 2) * args.offset_reg_w
            ssim_loss = (1.0 - ssim(fake_img, target[b])) * args.ssim_w

            total_loss = total_loss + l1_loss + vgg_loss + reg_loss + ssim_loss
            if use_gan:
                total_loss = total_loss + g_loss * 0.01
            if i >= MOUTH_LOSS_START_ITER and mouth_mask[b].sum() > 0:
                mouth_l1 = torch.mean(torch.abs(fake_img * target_mouth[b] - target[b] * target_mouth[b]))
                total_loss = total_loss + mouth_l1 * args.l1_w * 50
        total_loss.backward()
        g_optim.step()

        description = f"l1: {l1_loss.item():.4f}; vgg: {vgg_loss.item():.4f}"
        if use_gan:
            description += f"; g_loss: {g_loss.item():.4f}; d_loss: {d_loss_val:.4f}"
        pbar.set_description(description)

        if i % SAMPLE_EVERY == 0:
            with torch.no_grad():
                color_map = F.interpolate(g_maps[b:b + 1, SH][:, 0:3].detach(), size=target.shape[2:],
                                          mode="bilinear", align_corners=False)
                utils.save_image(torch.cat([target[b:b + 1], fake_img.unsqueeze(0), color_map], dim=3),
                                 os.path.join(sample_dir, f"gaussian_maps_{i:06d}.png"), nrow=int(args.batch ** 0.5))
            if use_gan and i % 3000 == 0:
                gen_losses.append(g_loss.item())
                disc_losses.append(d_loss_val)
                gan_steps.append(i - args.gan_start)
                plt.figure()
                plt.plot(gan_steps, gen_losses, "r", label="gen_loss")
                plt.plot(gan_steps, disc_losses, "g", label="discr_loss")
                plt.legend()
                plt.savefig(os.path.join(sample_dir, f"gaussian_maps_{i:06d}_plot.png"))
                plt.close()

        if i % CKPT_EVERY == 0:
            torch.save({"g": generator.state_dict(), "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(), "args": args},
                       os.path.join(ckpt_dir, f"gaussian_maps_{i:06d}.pt"))

        if i % VIDEO_EVERY == 0:
            with torch.no_grad():
                render_turntable(os.path.join(sample_dir, f"turntable_{i:06d}.mp4"), cameras, gaussians, points,
                                 features_dc, features_rest, device)


def main():
    args = parse_args()
    paths = load_paths(args.config)
    args.out_dir = args.out_dir or os.path.join(paths.output_root, "train")
    device = "cuda"

    # Seeding and construction order are kept fixed so runs reproduce the original model initialisation.
    seed_everything(0)
    cameras = load_nersemble_cameras(paths.nersemble_export)
    dataset = NeRSembleDataset(paths, args.num_train,
                               frame_counts_file=os.path.join("assets", "List_Indices_Exp_3_VHAP.npy"),
                               probability_file=os.path.join("assets", "Probability_Sampling_Exp_3_VHAP.json"),
                               uv_dir=paths.uv_assets,
                               mouth_mask_file=os.path.join(paths.uv_masks, "Background_Mouth.png"))

    generator = GaussianMapGenerator(inp_size=args.map_size, inp_ch=3 + 3, out_ch=59, out_size=args.map_size,
                                     style_dim=args.style_dim, n_mlp=args.n_mlp).to(device)
    discriminator = ProjectedDiscriminator(0, diffaug=False, p_crop=0.75).to(device)

    g_reg_ratio = 4 / 5
    d_reg_ratio = 10 / 11
    g_optim = optim.Adam([{"params": generator.parameters()}], lr=args.lr * g_reg_ratio)
    d_optim = optim.Adam(discriminator.parameters(), lr=0.0025 * d_reg_ratio)

    args.start_iter = 0
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            args.start_iter = int(os.path.splitext(os.path.basename(args.ckpt))[0].split("_")[-1])
        except ValueError:
            pass
        generator.load_state_dict(ckpt["g"], strict=False)
        g_optim.load_state_dict(ckpt["g_optim"])

    loader = data.DataLoader(dataset, batch_size=args.batch, sampler=data.RandomSampler(dataset), drop_last=True)

    clip_images = [os.path.join(paths.nersemble_clip_images, f"{subject:03d}", "EXP-9-jaw-2", "images",
                                "00000_cam_222200037.png") for subject in NERSEMBLE_SUBJECTS[:args.num_train]]
    identity_latents = clip_identity_embeddings(clip_images, device, args.batch, args.style_dim)

    train(args, paths, loader, generator, discriminator, g_optim, d_optim, cameras, identity_latents, device)


if __name__ == "__main__":
    main()
