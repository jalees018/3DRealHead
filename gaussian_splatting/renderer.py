#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from gaussian_splatting.gaussian_model import GaussianModel


def render(camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, override_opacity=None):
    """Rasterise the Gaussians in ``pc`` from ``camera``. ``bg_color`` must be on the GPU."""
    # Zero tensor used to make PyTorch return gradients of the 2D (screen-space) means.
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    shs, colors_precomp = (pc.get_features, None) if override_color is None else (None, override_color)
    opacity = pc.get_opacity if override_opacity is None else override_opacity

    rendered_image, radii = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}
