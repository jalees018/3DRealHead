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

import numpy as np
import torch


def get_world_to_view(R, t):
    """World-to-camera matrix from a (transposed) rotation and a translation."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # Round trip through camera-to-world as in the reference implementation (bit-identical results).
    C2W = np.linalg.inv(Rt)
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_projection_matrix(znear, zfar, fov_x, fov_y):
    tan_half_fov_y = math.tan(fov_y / 2)
    tan_half_fov_x = math.tan(fov_x / 2)

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class Camera:
    """Pinhole camera in the convention expected by the 3DGS rasterizer.

    ``R`` is the transposed world-to-camera rotation (as stored by the CUDA code's glm convention)
    and ``T`` the world-to-camera translation.
    """

    znear = 0.01
    zfar = 100.0

    def __init__(self, R, T, fov_x, fov_y, width, height):
        self.R = R
        self.T = T
        self.FoVx = fov_x
        self.FoVy = fov_y
        self.image_width = width
        self.image_height = height

        self.world_view_transform = torch.tensor(get_world_to_view(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = get_projection_matrix(self.znear, self.zfar, fov_x, fov_y).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
