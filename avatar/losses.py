import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class VGGLoss(nn.Module):
    """L1 distance between VGG19 features of the prediction and the target."""

    def __init__(self, device, n_layers=5):
        super().__init__()
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer
        for param in self.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)
        return loss


def hinge_d_loss(real_logits, fake_logits):
    return F.relu(torch.ones_like(fake_logits) + fake_logits).mean() + F.relu(torch.ones_like(real_logits) - real_logits).mean()


def hinge_g_loss(fake_logits):
    return (-fake_logits).mean()


def uv_coordinate_colors(uv_coords, texel_mask, device):
    """Per-Gaussian colours encoding their UV coordinate (u, v, -1), for UV renderings."""
    uv_colors = uv_coords[texel_mask.cpu()].unsqueeze(0)  # (1, N, 2)
    return torch.cat([uv_colors.to(device), -torch.ones((1, uv_colors.shape[1], 1), device=device)], dim=-1)


def uv_tv_loss(uv_rendering):
    """Total variation of a UV-coordinate rendering (alpha-blended over a white background)."""
    mask_uv = 1 - (uv_rendering[:, [2]] + 1) / 2
    uv_no_blend = (uv_rendering - (1 - mask_uv)) / mask_uv  # undo alpha blending; background becomes inf
    background = mask_uv == 0

    mask_y = (background[:, :, 1:] | background[:, :, :-1]).repeat(1, 3, 1, 1)
    mask_x = (background[:, :, :, 1:] | background[:, :, :, :-1]).repeat(1, 3, 1, 1)
    diff_y = uv_no_blend[:, :, 1:] - uv_no_blend[:, :, :-1]
    diff_x = uv_no_blend[:, :, :, 1:] - uv_no_blend[:, :, :, :-1]
    return diff_y[~mask_y].abs().mean() + diff_x[~mask_x].abs().mean()
