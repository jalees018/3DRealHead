# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Projected discriminator adapted from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".

A frozen DINO ViT-S/16 extracts multi-level features; small trainable 1D-conv heads
predict real/fake logits from each level. The features are also returned so they can
be used for a feature-matching loss.
"""
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import Normalize, RandomCrop

from networks.diffaug import DiffAugment
from networks.vit_utils import forward_vit, make_vit_backbone


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class FullyConnectedLayer(nn.Module):
    """Linear layer with learning-rate multiplier (used for class conditioning only)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 lr_multiplier: float = 1.0, weight_init: float = 1.0, bias_init: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is None:
            return x.matmul(w.t())
        b = b.to(x.dtype)
        if self.bias_gain != 1:
            b = b * self.bias_gain
        return torch.addmm(b.unsqueeze(0), x, w.t())


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    """Batch norm computed over groups of `virtual_bs` samples."""

    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(channels, channels, kernel_size=kernel_size,
                       padding=kernel_size // 2, padding_mode='circular'),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9)),
        )
        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        out = self.cls(self.main(x))
        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out


class DINO(nn.Module):
    """Frozen DINO ViT-S/16 returning token features of 4 blocks plus the patch embedding."""

    def __init__(self, hooks: list = [2, 5, 8, 11], hook_patch: bool = True):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)

        self.model = make_vit_backbone(
            timm.create_model('vit_small_patch16_224_dino', pretrained=True),
            patch_size=[16, 16], hooks=hooks, hook_patch=hook_patch,
        )
        self.model = self.model.eval().requires_grad_(False)

        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, x: torch.Tensor) -> dict:
        """x in [0, 1]; returns a dict of activations keyed '0'..'4'."""
        x = F.interpolate(x, self.img_resolution, mode='area')
        x = self.norm(x)
        return forward_vit(self.model, x)


class ProjectedDiscriminator(nn.Module):
    """DINO-feature projected discriminator.

    Args:
        c_dim: conditioning dimension (0 = unconditional).
        diffaug: apply DiffAugment (color, translation, cutout) to the input.
        p_crop: probability of taking a random DINO-resolution crop of larger inputs.
    """

    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop

        self.dino = DINO()
        heads = [(str(i), DiscHead(self.dino.embed_dim, c_dim)) for i in range(self.dino.n_hooks)]
        self.heads = nn.ModuleDict(heads)

    def train(self, mode: bool = True):
        # The DINO backbone always stays in eval mode.
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """x in [-1, 1]. Returns (logits (B, N), DINO features dict)."""
        if self.diffaug:
            x = DiffAugment(x, policy='color,translation,cutout')

        x = x.add(1).div(2)
        if x.size(-1) > self.dino.img_resolution and np.random.random() < self.p_crop:
            x = RandomCrop(self.dino.img_resolution)(x)

        features = self.dino(x)
        logits = [head(features[k], c).view(x.size(0), -1) for k, head in self.heads.items()]
        return torch.cat(logits, dim=1), features
