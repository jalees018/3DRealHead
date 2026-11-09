# MIT License
#
# Copyright (c) 2021 Intel ISL (Intel Intelligent Systems Lab)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Based on code from https://github.com/isl-org/DPT

"""Flexible configuration and feature extraction of timm VisionTransformers."""

import math
import types
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddReadout(nn.Module):
    def __init__(self, start_index: int = 1):
        super().__init__()
        self.start_index = start_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1).contiguous()


# Filled by forward hooks registered in `make_vit_backbone`.
activations = {}


def get_activation(name: str) -> Callable:
    def hook(model, input, output):
        activations[name] = output
    return hook


def forward_vit(pretrained: nn.Module, x: torch.Tensor) -> dict:
    pretrained.model.forward_flex(x)
    return {k: pretrained.rearrange(v) for k, v in activations.items()}


def _resize_pos_embed(self, posemb: torch.Tensor, gs_h: int, gs_w: int) -> torch.Tensor:
    posemb_tok, posemb_grid = posemb[:, :self.start_index], posemb[0, self.start_index:]
    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    return torch.cat([posemb_tok, posemb_grid], dim=1)


def forward_flex(self, x: torch.Tensor) -> torch.Tensor:
    """ViT forward pass with position embeddings resized to the input resolution."""
    B, C, H, W = x.size()
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    pos_embed = self._resize_pos_embed(self.pos_embed, H // self.patch_size[1], W // self.patch_size[0])

    cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + pos_embed
    x = self.pos_drop(x)
    for blk in self.blocks:
        x = blk(x)
    return self.norm(x)


def make_vit_backbone(model: nn.Module, patch_size: list = [16, 16], hooks: list = [2, 5, 8, 11],
                      hook_patch: bool = True, start_index: int = 1) -> nn.Module:
    assert len(hooks) == 4

    pretrained = nn.Module()
    pretrained.model = model
    for i, block_idx in enumerate(hooks):
        pretrained.model.blocks[block_idx].register_forward_hook(get_activation(str(i)))
    if hook_patch:
        pretrained.model.pos_drop.register_forward_hook(get_activation('4'))

    pretrained.rearrange = nn.Sequential(AddReadout(start_index), Transpose(1, 2))
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = patch_size

    # Inject methods so interpolated position embeddings work without modifying timm.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
    return pretrained
