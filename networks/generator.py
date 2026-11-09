"""Dual-branch StyleUNet that predicts UV-space Gaussian parameter maps.

The network encodes two conditioning images (expression/pose UV map and texture UV map)
with two separate convolutional encoders, fuses them at every resolution and decodes them
with StyleGAN2-style modulated convolutions into a map of per-texel Gaussian parameters.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from networks.ops import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        self.pad = ((p + 1) // 2 + factor - 1, p // 2)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor
        self.pad = ((p + 1) // 2, p // 2)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer("kernel", kernel)
        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, input):
        return F.conv2d(input, self.weight * self.scale, bias=self.bias,
                        stride=self.stride, padding=self.padding)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0.0, lr_mul=1.0, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            return fused_leaky_relu(out, self.bias * self.lr_mul)
        return F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim,
                 demodulate=True, upsample=False, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        # Grouped convolution applies a different modulated kernel to every sample.
        input = input.view(1, batch * in_channel, height, width)
        if self.upsample:
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = self.blur(out.view(batch, self.out_channel, height, width))
        else:
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False,
                 blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
        layers = []
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            layers.append(Blur(blur_kernel, pad=((p + 1) // 2, p // 2)))
            stride, padding = 2, 0
        else:
            stride, padding = 1, kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=padding,
                                  stride=stride, bias=bias and not activate))
        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))
        super().__init__(*layers)


def get_haar_wavelet():
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h
    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self):
        super().__init__()
        ll, lh, hl, hh = get_haar_wavelet()
        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)
        return torch.cat((ll, lh, hl, hh), 1)


class InverseHaarTransform(nn.Module):
    def __init__(self):
        super().__init__()
        ll, lh, hl, hh = get_haar_wavelet()
        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))
        return ll + lh + hl + hh


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

    def forward(self, input):
        return self.conv2(self.conv1(input))


class FromRGB(nn.Module):
    """Downsamples the conditioning image and projects it to features (added to `skip`)."""

    def __init__(self, out_channel, in_channel):
        super().__init__()
        self.downsample = Downsample([1, 3, 3, 1])
        self.conv = ConvLayer(in_channel, out_channel, 1)

    def forward(self, input, skip=None):
        input = self.downsample(input)
        out = self.conv(input)
        if skip is not None:
            out = out + skip
        return input, out


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False,
                 blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim,
                                    upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        return self.activate(out)


class ToRGB(nn.Module):
    """Modulated 1x1 projection to Haar-wavelet output channels with an upsampled skip."""

    def __init__(self, in_channel, style_dim, out_channel):
        super().__init__()
        self.upsample = Upsample([1, 3, 3, 1])
        self.iwt = InverseHaarTransform()
        self.dwt = HaarTransform()
        self.conv = ModulatedConv2d(in_channel, out_channel, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style) + self.bias
        if skip is not None:
            skip = self.dwt(self.upsample(self.iwt(skip)))
            out = out + skip
        return out


class GaussianMapGenerator(nn.Module):
    """Dual-encoder StyleUNet mapping two UV conditioning images to a Gaussian parameter map.

    Args:
        inp_size: resolution of the conditioning images.
        inp_ch: channels of each conditioning image.
        out_ch: channels of the output map (per-texel Gaussian parameters).
        out_size: output resolution.
        style_dim: dimension of the style/identity latent.
        n_mlp: number of layers in the mapping network.
    """

    def __init__(self, inp_size, inp_ch, out_ch, out_size, style_dim, n_mlp, middle_size=8, c_dim=0,
                 channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01):
        super().__init__()
        self.inp_size = inp_size
        self.style_dim = style_dim
        self.middle_log_size = int(math.log(middle_size, 2))

        layers = [PixelNorm(),
                  EqualLinear(style_dim + c_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu")]
        for _ in range(n_mlp - 1):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))
        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.log_size = int(math.log(out_size, 2)) - 1

        # Encoder for the first conditioning image (expression / pose UV map).
        in_channel = self.channels[inp_size // 2]
        self.from_rgbs = nn.ModuleList()
        self.cond_convs = nn.ModuleList()
        self.comb_convs = nn.ModuleList()
        self.comb_convs.append(ConvLayer(in_channel * 2, in_channel, 3))
        self.conv_in = ConvLayer(inp_ch, in_channel, 3, downsample=True)
        for i in range(int(math.log(inp_size, 2)) - 2, self.middle_log_size - 1, -1):
            out_channel = self.channels[2 ** i]
            self.from_rgbs.append(FromRGB(in_channel, inp_ch))
            self.cond_convs.append(ConvBlock(in_channel, out_channel))
            if i > self.middle_log_size:
                self.comb_convs.append(ConvLayer(out_channel * 2, out_channel, 3))
            else:
                self.comb_convs.append(ConvLayer(out_channel, out_channel, 3))
            in_channel = out_channel

        # Encoder for the second conditioning image (texture UV map).
        self.comb_convs2 = nn.ModuleList()
        in_channel = self.channels[inp_size // 2]
        self.from_rgbs2 = nn.ModuleList()
        self.cond_convs2 = nn.ModuleList()
        self.comb_convs2.append(ConvLayer(in_channel * 2, in_channel, 3))
        self.conv_in2 = ConvLayer(inp_ch, in_channel, 3, downsample=True)
        for i in range(int(math.log(inp_size, 2)) - 2, self.middle_log_size - 1, -1):
            out_channel = self.channels[2 ** i]
            self.from_rgbs2.append(FromRGB(in_channel, inp_ch))
            self.cond_convs2.append(ConvBlock(in_channel, out_channel))
            if i > self.middle_log_size:
                self.comb_convs2.append(ConvLayer(out_channel * 2, out_channel, 3))
            else:
                self.comb_convs2.append(ConvLayer(out_channel, out_channel, 3))
            in_channel = out_channel

        # Decoder. `convs2`/`to_rgbs2` are not used in forward; they are kept so that the
        # parameter layout matches existing checkpoints and optimizer states.
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.to_rgbs1 = nn.ModuleList()
        self.to_rgbs2 = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[middle_size] * 2
        self.num_layers = (self.log_size - self.middle_log_size) * 2
        for layer_idx in range(self.num_layers):
            res = (layer_idx + 8) // 2
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(1, 1, 2 ** res, 2 ** res))
        for i in range(self.middle_log_size + 1, self.log_size + 1):
            out_channel = self.channels[2 ** i] * 2
            self.convs1.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs1.append(StyledConv(out_channel, out_channel // 2, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs1.append(ToRGB(out_channel // 2, style_dim, out_ch * 4))
            self.convs2.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs2.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs2.append(ToRGB(out_channel, style_dim, out_ch * 4))
            in_channel = out_channel
        # `dwt` is unused; kept for checkpoint compatibility.
        self.dwt = HaarTransform()
        self.iwt = InverseHaarTransform()

        self.n_latent = self.log_size * 2 - (self.middle_log_size * 2 - 1) + 1

    def map_latent(self, styles):
        """Map a style vector (B, style_dim) to per-layer latents (B, n_latent, style_dim)."""
        return self.style(styles).unsqueeze(1).repeat(1, self.n_latent, 1)

    @staticmethod
    def _encode(image, conv_in, from_rgbs, cond_convs):
        out = conv_in(image)
        features = [out]
        for from_rgb, cond_conv in zip(from_rgbs, cond_convs):
            image, out = from_rgb(image, out)
            out = cond_conv(out)
            features.append(out)
        return features

    def encode_conditions(self, condition_img, condition_img2):
        """Run both encoders; returns the two multi-scale feature lists."""
        cond_list = self._encode(condition_img, self.conv_in, self.from_rgbs, self.cond_convs)
        cond_list2 = self._encode(condition_img2, self.conv_in2, self.from_rgbs2, self.cond_convs2)
        return cond_list, cond_list2

    def _noise(self, randomize_noise):
        if randomize_noise:
            return [None] * self.num_layers
        return [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

    def _decode(self, latent, cond_list, cond_list2, noise):
        i = 0
        skip = None
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs1[::2], self.convs1[1::2], noise[::2], noise[1::2], self.to_rgbs1):
            if i == 0:
                out = self.comb_convs[-1](cond_list[-1])
                out2 = self.comb_convs2[-1](cond_list2[-1])
                fused = torch.cat((out, out2), dim=1)
            elif i < 2 * len(self.comb_convs):
                out = torch.cat([out, cond_list[-1 - (i // 2)]], dim=1)
                out = self.comb_convs[-1 - (i // 2)](out)
                out2 = torch.cat([out2, cond_list2[-1 - (i // 2)]], dim=1)
                out2 = self.comb_convs2[-1 - (i // 2)](out2)
                fused = torch.cat((out, out2), dim=1)
            out = conv1(fused, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            out2 = out
            i += 2
        return self.iwt(skip)

    def forward(self, styles, condition_img, condition_img2, randomize_noise=True):
        """Returns the Gaussian parameter map twice (kept as a pair for API compatibility).

        randomize_noise=False uses the fixed noise buffers instead of fresh Gaussian noise.
        """
        noise = self._noise(randomize_noise)
        latent = self.map_latent(styles)
        cond_list, cond_list2 = self.encode_conditions(condition_img, condition_img2)
        image = self._decode(latent, cond_list, cond_list2, noise)
        return image, image


class GaussianMapGeneratorInversion(GaussianMapGenerator):
    """Generator variant used at test time, where latents and encoder features are optimised.

    Same parameters as `GaussianMapGenerator`; only `forward` accepts pre-computed inputs.
    """

    def forward(self, styles, condition_img, condition_img2, randomize_noise=True,
                optimize_latent=False, drive=False):
        """
        optimize_latent: `styles` are per-layer latents (see `map_latent`) and the two
            conditions are pre-computed encoder feature lists (see `encode_conditions`).
        drive: `styles` are per-layer latents, `condition_img` is a new expression map that is
            encoded here, and `condition_img2` is a pre-computed texture feature list.
        """
        noise = self._noise(randomize_noise)
        if drive:
            latent = styles
            cond_list = self._encode(condition_img, self.conv_in, self.from_rgbs, self.cond_convs)
            cond_list2 = condition_img2
        elif optimize_latent:
            latent = styles
            cond_list, cond_list2 = condition_img, condition_img2
        else:
            latent = self.map_latent(styles)
            cond_list, cond_list2 = self.encode_conditions(condition_img, condition_img2)
        image = self._decode(latent, cond_list, cond_list2, noise)
        return image, image
