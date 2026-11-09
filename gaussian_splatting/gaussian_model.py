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

import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class GaussianModel:
    """Container for one set of 3D Gaussians.

    Unlike vanilla 3DGS, the Gaussians are not optimised directly: the network predicts them
    every iteration and writes them in with :meth:`modify`. Parameters are stored pre-activation
    (log-scales, opacity logits, unnormalised quaternions). ``active_sh_degree`` stays 0, so only
    the DC colour term is rasterised.
    """

    def __init__(self, sh_degree=3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

    def modify(self, xyz, opacity, scaling, rotation, features_dc, features_rest):
        self._xyz = xyz
        self._opacity = opacity
        self._scaling = scaling
        self._rotation = rotation
        self._features_dc = features_dc
        self._features_rest = features_rest

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)
