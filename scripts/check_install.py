"""Check the installation without any dataset.

Loads a generator checkpoint and FLAME, predicts Gaussians for synthetic neutral conditioning maps, places
them on a neutral FLAME head, renders a frontal view to ``--out`` and runs one backward pass through the
renderer (exercising all CUDA extensions).

    python scripts/check_install.py --ckpt checkpoints/gaussian_maps.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
from torchvision import utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avatar.cameras import _camera_from_c2w  # noqa: E402
from avatar.config import load_paths  # noqa: E402
from avatar.geometry import (FlameParams, gaussian_maps_to_3d, init_gaussians, load_canonical_position_map,  # noqa: E402
                             load_uv_layout, split_sh)
from flame import FLAMEModule  # noqa: E402
from gaussian_splatting import GaussianModel, render  # noqa: E402
from networks.generator import GaussianMapGeneratorInversion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="outputs/check_install.png")
    args = parser.parse_args()
    paths = load_paths(args.config)
    device = "cuda"
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name()}")

    generator = GaussianMapGeneratorInversion(inp_size=512, inp_ch=6, out_ch=59, out_size=512, style_dim=512,
                                              n_mlp=6).to(device)
    state = torch.load(args.ckpt, map_location="cpu")["g"]
    missing, unexpected = generator.load_state_dict(state, strict=False)
    print(f"checkpoint: {len(state)} tensors, missing {len(missing)}, unexpected {len(unexpected)}")
    assert not missing and not unexpected, (missing, unexpected)

    can_loc, texel_mask = load_canonical_position_map(paths.position_map, device)
    init = init_gaussians(can_loc, texel_mask, device)
    uv_layout = load_uv_layout(paths.uv_assets, device)
    flame_model = FLAMEModule(1, device, paths.flame_assets)
    zeros = lambda *shape: torch.zeros(shape, device=device)  # noqa: E731
    flame = FlameParams(zeros(1, 100), zeros(1, 59), torch.ones((1, 1), device=device), zeros(1, 6), zeros(1, 3))

    # Neutral expression (zero offsets and gradients) and a uniform grey texture in [-1, 1].
    expression = zeros(2, 6, 512, 512)
    texture = torch.cat((torch.full((2, 3, 512, 512), 0.2, device=device), zeros(2, 3, 512, 512)), dim=1)
    style = torch.full((2, 512), 1 / np.sqrt(512), device=device)
    g_maps, _ = generator(style, expression, texture, randomize_noise=False)
    points, displacement = gaussian_maps_to_3d(g_maps[0:1], texel_mask, init, flame_model, flame, uv_layout)
    print(f"Gaussians: {points['xyz'].shape[0]}, xyz range {points['xyz'].min(0).values.tolist()} .. "
          f"{points['xyz'].max(0).values.tolist()}")

    gaussians = GaussianModel()
    gaussians.modify(points["xyz"], points["opacity"], points["scaling"], points["rotation"], *split_sh(points["colors"]))
    c2w = np.eye(4)
    c2w[2, 3] = 1.0  # 1 m in front of the head, looking down -z (OpenGL convention)
    camera = _camera_from_c2w(c2w, 0.22, 0.32, (550, 802))
    image = render(camera, gaussians, torch.ones(3, device=device))["render"]
    print(f"render: {tuple(image.shape)}, mean {image.mean().item():.3f}, "
          f"foreground pixels {(image.mean(0) < 0.99).float().mean().item():.1%}")

    (image.mean() + displacement.pow(2).mean()).backward()
    grad = sum(p.grad.abs().sum().item() for p in generator.parameters() if p.grad is not None)
    print(f"backward ok, generator grad L1 {grad:.3e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    utils.save_image(image.clamp(0, 1), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
