# -*- coding: utf-8 -*-
import einops
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.registry import ARCH_REGISTRY
from archs.ConStyle_arch import ConStyle


# from basicsr.models.archs.local_arch import Local_Base

def nearest_downsample(x, ratio):
    n, c, h, w = x.shape
    new_h = int(h * ratio)
    new_w = int(w * ratio)
    h_index = np.floor((np.arange(new_h) + 0.5) / ratio)
    w_index = np.floor((np.arange(new_w) + 0.5) / ratio)
    out = x[:, :, h_index, :]
    out = out[:, :, :, w_index]
    return out


class Layer_norm_process(nn.Module):  # n, h, w, c
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(c), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(c), requires_grad=True)
        self.eps = eps

    def forward(self, feature):
        var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
        mean = var_mean[1]
        var = var_mean[0]
        # layer norm process
        feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        gamma = self.gamma.expand_as(feature)
        beta = self.beta.expand_as(feature)
        feature = feature * gamma + beta
        return feature


def block_images_einops(x, patch_size):  # n, h, w, c
    """Image to patches."""
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


class UpSampleRatio(nn.Module):
    def __init__(self, in_features, out_features, ratio=1., use_bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.in_features, self.out_features, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        if self.ratio == 1:
            x = self.Conv_0(x)
        else:
            n, c, h, w = x.shape
            x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
            x = self.Conv_0(x)
        return x


class UpSampleRatio_(nn.Module):
    def __init__(self, ratio=1.):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if not self.ratio == 1:
            n, c, h, w = x.shape
            x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        return x


class BlockGatingUnit(nn.Module):  # input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, c, n, use_bias=True):
        super().__init__()
        self.c = c
        self.n = n
        self.use_bias = use_bias
        self.Dense_0 = nn.Linear(self.n, self.n, self.use_bias)
        self.intermediate_layernorm = Layer_norm_process(self.c // 2)

    def forward(self, x):
        c = x.size(-1)
        c = c // 2  # split size
        u, v = torch.split(x, c, dim=-1)
        v = self.intermediate_layernorm(v)
        v = v.permute(0, 1, 3, 2).contiguous()  # n, (gh gw), c/2, (fh fw)
        v = self.Dense_0(v)  # apply fc on the last dimension (fh fw)
        v = v.permute(0, 1, 3, 2).contiguous()  # n (gh gw) (fh fw) c/2
        return u * (v + 1.)


class GridGatingUnit(nn.Module):  # input shape: n (gh gw) (fh fw) c
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second.
    If applied on other dims, you should swapaxes first.
    """

    def __init__(self, c, n, use_bias=True):
        super().__init__()
        self.c = c
        self.n = n
        self.use_bias = use_bias
        self.intermediate_layernorm = Layer_norm_process(self.c // 2)
        self.Dense_0 = nn.Linear(self.n, self.n, self.use_bias)

    def forward(self, x):
        c = x.size(-1)
        c = c // 2  # split size
        u, v = torch.split(x, c, dim=-1)
        v = self.intermediate_layernorm(v)
        v = v.permute(0, 3, 2, 1).contiguous()  # n, c/2, (fh fw) (gh gw)
        v = self.Dense_0(v)  # apply fc on the last dimension (gh gw)
        v = v.permute(0, 3, 2, 1).contiguous()  # n (gh gw) (fh fw) c/2
        return u * (v + 1.)


class GridGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """Grid gMLP layer that performs global mixing of tokens."""

    def __init__(self, grid_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.grid_size = grid_size
        self.gh = grid_size[0]
        self.gw = grid_size[1]
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.LayerNorm = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.factor, self.use_bias)  # c->c*factor
        self.gelu = nn.GELU()
        self.GridGatingUnit = GridGatingUnit(self.num_channels * self.factor,
                                             n=self.gh * self.gw)  # number of channels????????????????
        self.out_project = nn.Linear(self.num_channels * self.factor // 2, self.num_channels,
                                     self.use_bias)  # c*factor->c
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        n, h, w, num_channels = x.shape
        fh, fw = h // self.gh, w // self.gw
        x = block_images_einops(x, patch_size=(fh, fw))  # n (gh gw) (fh fw) c
        # gMLP1: Global (grid) mixing part, provides global grid communication.
        y = self.LayerNorm(x)
        y = self.in_project(y)  # channel proj
        y = self.gelu(y)
        y = self.GridGatingUnit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        return x


class BlockGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """Block gMLP layer that performs local mixing of tokens."""

    def __init__(self, block_size, num_channels, use_bias=True, factor=2, dropout_rate=0):
        super().__init__()
        self.block_size = block_size
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.LayerNorm = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.factor, self.use_bias)  # c->c*factor
        self.gelu = nn.GELU()
        self.BlockGatingUnit = BlockGatingUnit(self.num_channels * self.factor,
                                               n=self.fh * self.fw)  # number of channels????????????????
        self.out_project = nn.Linear(self.num_channels * self.factor // 2, self.num_channels,
                                     self.use_bias)  # c*factor->c
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        _, h, w, _ = x.shape
        gh, gw = h // self.fh, w // self.fw
        x = block_images_einops(x, patch_size=(self.fh, self.fw))  # n (gh gw) (fh fw) c
        # gMLP2: Local (block) mixing part, provides local block communication.
        y = self.LayerNorm(x)
        y = self.in_project(y)  # channel proj
        y = self.gelu(y)
        y = self.BlockGatingUnit(y)
        y = self.out_project(y)
        y = self.dropout(y)
        x = x + y
        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(self.fh, self.fw))
        return x


class MlpBlock(nn.Module):  # input shape: n, h, w, c
    """A 1-hidden-layer MLP block, applied over the last dimension."""

    def __init__(self, mlp_dim, dropout_rate=0., use_bias=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.Dense_0 = nn.Linear(self.mlp_dim, self.mlp_dim, bias=self.use_bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.Dense_1 = nn.Linear(self.mlp_dim, self.mlp_dim, bias=self.use_bias)

    def forward(self, x):
        x = self.Dense_0(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.Dense_1(x)
        return x


class CALayer(nn.Module):  # input shape: n, h, w, c
    """Squeeze-and-excitation block for channel attention.
    ref: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, features, reduction=4, use_bias=True):
        super().__init__()
        self.features = features
        self.reduction = reduction
        self.use_bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features // self.reduction, kernel_size=(1, 1), stride=1,
                                bias=self.use_bias)  # 1*1 conv
        self.relu = nn.ReLU()
        self.Conv_1 = nn.Conv2d(self.features // self.reduction, self.features, kernel_size=(1, 1), stride=1,
                                bias=self.use_bias)  # 1*1 conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.permute(0, 3, 1, 2).contiguous()  # n, c, h, w
        y = torch.mean(y, dim=(2, 3), keepdim=True)  # keep dimensions for element product in the last step
        y = self.Conv_0(y)
        y = self.relu(y)
        y = self.Conv_1(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 3, 1).contiguous()  # n, h, w, c
        return x * y


class GetSpatialGatingWeights(nn.Module):  # n, h, w, c
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, num_channels, grid_size, block_size, input_proj_factor=2, use_bias=True, dropout_rate=0):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.block_size = block_size
        self.gh = self.grid_size[0]
        self.gw = self.grid_size[1]
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]
        self.input_proj_factor = input_proj_factor
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.input_proj_factor, bias=self.use_bias)
        self.gelu = nn.GELU()
        self.Dense_0 = nn.Linear(self.gh * self.gw, self.gh * self.gw, bias=self.use_bias)
        self.Dense_1 = nn.Linear(self.fh * self.fw, self.fh * self.fw, bias=self.use_bias)
        self.out_project = nn.Linear(self.num_channels * self.input_proj_factor, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        _, h, w, _ = x.shape
        # input projection
        x = self.LayerNorm_in(x)
        x = self.in_project(x)  # channel projection
        x = self.gelu(x)
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)
        # get grid MLP weights
        fh, fw = h // self.gh, w // self.gw
        u = block_images_einops(u, patch_size=(fh, fw))  # n, (gh gw) (fh fw) c
        u = u.permute(0, 3, 2, 1).contiguous()  # n, c, (fh fw) (gh gw)
        u = self.Dense_0(u)
        u = u.permute(0, 3, 2, 1).contiguous()  # n, (gh gw) (fh fw) c
        u = unblock_images_einops(u, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        # get block MLP weights
        gh, gw = h // self.fh, w // self.fw
        v = block_images_einops(v, patch_size=(self.fh, self.fw))  # n, (gh gw) (fh fw) c
        v = v.permute(0, 1, 3, 2).contiguous()  # n (gh gw) c (fh fw)
        v = self.Dense_1(v)
        v = v.permute(0, 1, 3, 2).contiguous()  # n, (gh gw) (fh fw) c
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(self.fh, self.fw))

        x = torch.cat([u, v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        return x


class ResidualSplitHeadMultiAxisGmlpLayer(nn.Module):  # input shape: n, h, w, c
    """The multi-axis gated MLP block."""

    def __init__(self, block_size, grid_size, num_channels, input_proj_factor=2, block_gmlp_factor=2,
                 grid_gmlp_factor=2, use_bias=True, dropout_rate=0.):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.input_proj_factor = input_proj_factor
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.input_proj_factor, bias=self.use_bias)
        self.gelu = nn.GELU()
        self.GridGmlpLayer = GridGmlpLayer(grid_size=self.grid_size,
                                           num_channels=self.num_channels * self.input_proj_factor // 2,
                                           use_bias=self.use_bias, factor=self.grid_gmlp_factor)
        self.BlockGmlpLayer = BlockGmlpLayer(block_size=self.block_size,
                                             num_channels=self.num_channels * self.input_proj_factor // 2,
                                             use_bias=self.use_bias, factor=self.block_gmlp_factor)
        self.out_project = nn.Linear(self.num_channels * self.input_proj_factor, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        shortcut = x
        x = self.LayerNorm_in(x)
        x = self.in_project(x)
        x = self.gelu(x)
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)
        # grid gMLP
        u = self.GridGmlpLayer(u)
        # block gMLP
        v = self.BlockGmlpLayer(v)
        # out projection
        x = torch.cat([u, v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


class RCAB(nn.Module):  # input shape: n, h, w, c
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def __init__(self, features, reduction=4, lrelu_slope=0.2, use_bias=True):
        super().__init__()
        self.features = features
        self.reduction = reduction
        self.lrelu_slope = lrelu_slope
        self.bias = use_bias
        self.LayerNorm = Layer_norm_process(self.features)
        self.conv1 = nn.Conv2d(self.features, self.features, kernel_size=(3, 3), stride=1, bias=self.bias, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.lrelu_slope)
        self.conv2 = nn.Conv2d(self.features, self.features, kernel_size=(3, 3), stride=1, bias=self.bias, padding=1)
        self.channel_attention = CALayer(features=self.features, reduction=self.reduction)

    def forward(self, x):
        shortcut = x
        x = self.LayerNorm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # n, c, h, w
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # n, h, w, c
        x = self.channel_attention(x)
        return x + shortcut


class RDCAB(nn.Module):  # input shape: n, h, w, c
    """Residual dense channel attention block. Used in Bottlenecks."""

    def __init__(self, features, reduction=4, dropout_rate=0, use_bias=True):
        super().__init__()
        self.features = features
        self.reduction = reduction
        self.drop = dropout_rate
        self.bias = use_bias
        self.LayerNorm = Layer_norm_process(self.features)
        self.channel_mixing = MlpBlock(mlp_dim=self.features, dropout_rate=self.drop, use_bias=self.bias)
        self.channel_attention = CALayer(features=self.features, reduction=self.reduction, use_bias=self.bias)

    def forward(self, x):
        y = self.LayerNorm(x)
        y = self.channel_mixing(y)
        y = self.channel_attention(y)
        x = x + y
        return x


class CrossGatingBlock(nn.Module):  # input shape: n, c, h, w
    """Cross-gating MLP block."""

    def __init__(self, x_features, num_channels, block_size, grid_size, cin_y, upsample_y=True, use_bias=True,
                 use_global_mlp=True, dropout_rate=0):
        super().__init__()
        self.cin_y = cin_y
        self.x_features = x_features
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.use_global_mlp = use_global_mlp
        self.drop = dropout_rate
        if self.upsample_y:
            self.ConvTranspose_0 = nn.ConvTranspose2d(self.cin_y, self.num_channels, kernel_size=(2, 2), stride=2,
                                                      bias=self.use_bias)
        self.Conv_0 = nn.Conv2d(self.x_features, self.num_channels, kernel_size=(1, 1), stride=1, bias=self.use_bias)
        self.Conv_1 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=(1, 1), stride=1, bias=self.use_bias)
        self.LayerNorm_x = Layer_norm_process(self.num_channels)
        self.in_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu1 = nn.GELU()
        self.SplitHeadMultiAxisGating_x = GetSpatialGatingWeights(num_channels=self.num_channels,
                                                                  block_size=self.block_size, grid_size=self.grid_size,
                                                                  dropout_rate=self.drop, use_bias=self.use_bias)
        self.LayerNorm_y = Layer_norm_process(self.num_channels)
        self.in_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.gelu2 = nn.GELU()
        self.SplitHeadMultiAxisGating_y = GetSpatialGatingWeights(num_channels=self.num_channels,
                                                                  block_size=self.block_size, grid_size=self.grid_size,
                                                                  dropout_rate=self.drop, use_bias=self.use_bias)
        self.out_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout1 = nn.Dropout(self.drop)
        self.out_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout2 = nn.Dropout(self.drop)

    def forward(self, x, y):
        # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
            y = self.ConvTranspose_0(y)
        x = self.Conv_0(x)
        y = self.Conv_1(y)
        assert y.shape == x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # n,h,w,c
        y = y.permute(0, 2, 3, 1).contiguous()  # n,h,w,c
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.in_project_x(x)
        x = self.gelu1(x)
        gx = self.SplitHeadMultiAxisGating_x(x)
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.in_project_y(y)
        y = self.gelu2(y)
        gy = self.SplitHeadMultiAxisGating_y(y)
        # Apply cross gating
        y = y * gx  ## gating y using x
        y = self.out_project_y(y)
        y = self.dropout1(y)
        y = y + shortcut_y
        x = x * gy  # gating x using y
        x = self.out_project_x(x)
        x = self.dropout2(x)
        x = x + y + shortcut_x  # get all aggregated signals
        return x.permute(0, 3, 1, 2).contiguous(), y.permute(0, 3, 1, 2).contiguous()  # n,c,h,w


class UNetEncoderBlock(nn.Module):  # input shape: n, c, h, w (pytorch default)
    """Encoder block in MAXIM."""

    def __init__(self, cin, num_channels, block_size, grid_size, dec=False, lrelu_slope=0.2,
                 block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2, channels_reduction=4,
                 dropout_rate=0., use_bias=True, use_global_mlp=True):
        super().__init__()
        self.cin = cin
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.lrelu_slope = lrelu_slope
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.reduction = channels_reduction
        self.drop = dropout_rate
        self.dec = dec
        self.use_bias = use_bias
        self.use_global_mlp = use_global_mlp
        if self.cin is not None:
            self.Conv_0 = nn.Conv2d(self.cin, self.num_channels, kernel_size=(1, 1), stride=(1, 1), bias=self.use_bias)
        self.SplitHeadMultiAxisGmlpLayer_0 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size,
                                                                                 grid_size=self.grid_size,
                                                                                 num_channels=self.num_channels,
                                                                                 input_proj_factor=self.input_proj_factor,
                                                                                 block_gmlp_factor=self.block_gmlp_factor,
                                                                                 grid_gmlp_factor=self.grid_gmlp_factor,
                                                                                 dropout_rate=self.drop,
                                                                                 use_bias=self.use_bias)
        self.SplitHeadMultiAxisGmlpLayer_1 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size,
                                                                                 grid_size=self.grid_size,
                                                                                 num_channels=self.num_channels,
                                                                                 input_proj_factor=self.input_proj_factor,
                                                                                 block_gmlp_factor=self.block_gmlp_factor,
                                                                                 grid_gmlp_factor=self.grid_gmlp_factor,
                                                                                 dropout_rate=self.drop,
                                                                                 use_bias=self.use_bias)
        self.channel_attention_block_10 = RCAB(features=self.num_channels, reduction=self.reduction,
                                               lrelu_slope=self.lrelu_slope, use_bias=self.use_bias)
        self.channel_attention_block_11 = RCAB(features=self.num_channels, reduction=self.reduction,
                                               lrelu_slope=self.lrelu_slope, use_bias=self.use_bias)
        self.cross_gating_block = CrossGatingBlock(x_features=self.num_channels, num_channels=self.num_channels,
                                                   block_size=self.block_size,
                                                   grid_size=self.grid_size, cin_y=0, upsample_y=False,
                                                   dropout_rate=self.drop, use_bias=self.use_bias,
                                                   use_global_mlp=self.use_global_mlp)

    def forward(self, x, skip=None, enc=None, dec=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        if self.cin is not None:
            x = self.Conv_0(x)
        shortcut_long = x
        x = x.permute(0, 2, 3, 1).contiguous()  # n,h,w,c
        if self.use_global_mlp:
            x = self.SplitHeadMultiAxisGmlpLayer_0(x)
        x = self.channel_attention_block_10(x)
        if self.use_global_mlp:
            x = self.SplitHeadMultiAxisGmlpLayer_1(x)
        x = self.channel_attention_block_11(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # n,c,h,w
        x = x + shortcut_long
        if enc is not None and dec is not None:  # if stage>0
            x, _ = self.cross_gating_block(x, enc + dec)
        return x


class UNetDecoderBlock(nn.Module):  # input shape: n, c, h, w
    """Decoder block in MAXIM."""

    def __init__(self, cin, num_channels, block_size, grid_size, lrelu_slope=0.2, block_gmlp_factor=2,
                 grid_gmlp_factor=2, input_proj_factor=2, channels_reduction=4, dropout_rate=0.,
                 use_bias=True, use_global_mlp=True):
        super().__init__()
        self.cin = cin
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.lrelu_slope = lrelu_slope
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.reduction = channels_reduction
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.use_global_mlp = use_global_mlp
        self.UNetEncoderBlock_0 = UNetEncoderBlock(self.cin, self.num_channels, self.block_size,
                                                   self.grid_size, lrelu_slope=self.lrelu_slope,
                                                   block_gmlp_factor=self.block_gmlp_factor,
                                                   grid_gmlp_factor=self.grid_gmlp_factor, dec=True,
                                                   input_proj_factor=self.input_proj_factor,
                                                   channels_reduction=self.reduction, dropout_rate=self.dropout_rate,
                                                   use_bias=self.use_bias, use_global_mlp=self.use_global_mlp)

    def forward(self, x, bridge=None):
        x = self.UNetEncoderBlock_0(x, skip=bridge)
        return x


class BottleneckBlock(nn.Module):  # input shape: n,c,h,w
    """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""

    def __init__(self, features, block_size, grid_size, block_gmlp_factor=2, grid_gmlp_factor=2, input_proj_factor=2,
                 channels_reduction=4, use_bias=True, dropout_rate=0.):
        super().__init__()
        self.features = features
        self.block_size = block_size
        self.grid_size = grid_size
        self.block_gmlp_factor = block_gmlp_factor
        self.grid_gmlp_factor = grid_gmlp_factor
        self.input_proj_factor = input_proj_factor
        self.channels_reduction = channels_reduction
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.input_proj = nn.Conv2d(self.features, self.features, kernel_size=(1, 1), stride=1)
        self.SplitHeadMultiAxisGmlpLayer_0 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size,
                                                                                 grid_size=self.grid_size,
                                                                                 num_channels=self.features,
                                                                                 input_proj_factor=self.input_proj_factor,
                                                                                 block_gmlp_factor=self.block_gmlp_factor,
                                                                                 grid_gmlp_factor=self.grid_gmlp_factor,
                                                                                 use_bias=self.use_bias)
        self.SplitHeadMultiAxisGmlpLayer_1 = ResidualSplitHeadMultiAxisGmlpLayer(block_size=self.block_size,
                                                                                 grid_size=self.grid_size,
                                                                                 num_channels=self.features,
                                                                                 input_proj_factor=self.input_proj_factor,
                                                                                 block_gmlp_factor=self.block_gmlp_factor,
                                                                                 grid_gmlp_factor=self.grid_gmlp_factor,
                                                                                 use_bias=self.use_bias)
        self.channel_attention_block_1_0 = RDCAB(features, dropout_rate=self.drop, use_bias=self.use_bias)
        self.channel_attention_block_1_1 = RDCAB(features, dropout_rate=self.drop, use_bias=self.use_bias)

    def forward(self, x):
        assert x.ndim == 4  # Input has shape [batch, c, h, w]
        # input projection
        x = self.input_proj(x)
        shortcut_long = x
        x = x.permute(0, 2, 3, 1).contiguous()  # n, h, w, c
        x = self.SplitHeadMultiAxisGmlpLayer_0(x)
        x = self.channel_attention_block_1_0(x)
        x = self.SplitHeadMultiAxisGmlpLayer_1(x)
        x = self.channel_attention_block_1_1(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # n, c, h, w
        x = x + shortcut_long
        return x


# multi stage
class SAM(nn.Module):  # x shape and x_image shape: n, c, h, w
    """Supervised attention module for multi-stage training.
    Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
    """

    def __init__(self, features, output_channels=3, use_bias=True):
        super().__init__()
        self.features = features  # cin
        self.output_channels = output_channels
        self.use_bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features, kernel_size=(3, 3), bias=self.use_bias, padding=1)
        self.Conv_1 = nn.Conv2d(self.features, self.output_channels, kernel_size=(3, 3), bias=self.use_bias, padding=1)
        self.Conv_2 = nn.Conv2d(self.output_channels, self.features, kernel_size=(3, 3), bias=self.use_bias, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_image):
        """Apply the SAM module to the input and features.
        Args:
          x: the output features from UNet decoder with shape (h, w, c)
          x_image: the input image with shape (h, w, 3)
          train: Whether it is training
        Returns:
          A tuple of tensors (x1, image) where (x1) is the sam features used for the
            next stage, and (image) is the output restored image at current stage.
        """
        # Get features
        x1 = self.Conv_0(x)
        # Output restored image X_s
        if self.output_channels == 3:
            image = self.Conv_1(x) + x_image
        else:
            image = self.Conv_1(x)
        # Get attention maps for features
        x2 = self.Conv_2(image)
        x2 = self.sigmoid(x2)
        # Get attended feature maps
        x1 = x1 * x2
        # Residual connection
        x1 = x1 + x
        return x1, image


class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, padding=None, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvAct, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=bias),
            nn.LeakyReLU(0.1, True)
        )


class UC(nn.Module):
    def __init__(self, n_feat):
        super(UC, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.PixelShuffle(2))

    def forward(self, x):
        x = self.up(x)
        return x


class DC(nn.Module):
    def __init__(self, n_feat):
        super(DC, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.PixelUnshuffle(2))

    def forward(self, x, fea):
        x = self.down(x)
        x = torch.cat([x, fea], 1)
        return x

@ARCH_REGISTRY.register()
class ConStyleMaxim(nn.Module):  # input shape: n, c, h, w
    """The MAXIM model function with multi-stage and multi-scale supervision.
    For more model details, please check the CVPR paper:
    MAXIM: MUlti-Axis MLP for Image Processing (https://arxiv.org/abs/2201.02973)
    Attributes:
      features: initial hidden dimension for the input resolution.
      depth: the number of downsampling depth for the model.
      num_stages: how many stages to use. It will also affects the output list.
      use_bias: whether to use bias in all the conv/mlp layers.
      num_supervision_scales: the number of desired supervision scales.
      lrelu_slope: the negative slope parameter in leaky_relu layers.
      use_global_mlp: whether to use the multi-axis gated MLP block (MAB) in each
        layer.
      use_cross_gating: whether to use the cross-gating MLP block (CGB) in the
        skip connections and multi-stage feature fusion layers.
      high_res_stages: how many stages are specificied as high-res stages. The
        rest (depth - high_res_stages) are called low_res_stages.
      block_size_hr: the block_size parameter for high-res stages.
      block_size_lr: the block_size parameter for low-res stages.
      grid_size_hr: the grid_size parameter for high-res stages.
      grid_size_lr: the grid_size parameter for low-res stages.
      block_gmlp_factor: the input projection factor for block_gMLP layers.
      grid_gmlp_factor: the input projection factor for grid_gMLP layers.
      input_proj_factor: the input projection factor for the MAB block.
      channels_reduction: the channel reduction factor for SE layer.
      num_outputs: the output channels.
      dropout_rate: Dropout rate.
    Returns:
      The output contains a list of arrays consisting of multi-stage multi-scale
      outputs. For example, if num_stages = num_supervision_scales = 3 (the
      model used in the paper), the output specs are: outputs =
      [[output_stage1_scale1, output_stage1_scale2, output_stage1_scale3],
       [output_stage2_scale1, output_stage2_scale2, output_stage2_scale3],
       [output_stage3_scale1, output_stage3_scale2, output_stage3_scale3],]
      The final output can be retrieved by outputs[-1][-1].
    """
    # default: features=32 bias=True
    def __init__(self,
                 features=32,
                 bias=False,
                 bottom_blk_num=2,
                 lrelu_slope=0.2,
                 use_global_mlp=True,
                 high_res_stages=2,
                 block_size_hr=(16, 16),
                 block_size_lr=(8, 8),
                 grid_size_hr=(16, 16),
                 grid_size_lr=(8, 8),
                 block_gmlp_factor=2,
                 grid_gmlp_factor=2,
                 input_proj_factor=2,
                 channels_reduction=4,
                 num_outputs=3,
                 dropout_rate=0.,
                 Train=True
                 ):

        super(ConStyleMaxim, self).__init__()

        # ConStyle
        self.ConStyle = ConStyle(Train)


        self.stage_input_conv_0 = nn.Conv2d(3, features, kernel_size=(3, 3), bias=bias, padding=1)
        self.stage_input_conv_1 = nn.Conv2d(3, features, kernel_size=(3, 3), bias=bias, padding=1)
        self.stage_input_conv_2 = nn.Conv2d(3, features, kernel_size=(3, 3), bias=bias, padding=1)

        # U-net left
        self.Process_Module_l1 = UNetEncoderBlock(cin=2 * features, num_channels=1 * features,
                                                  block_size=block_size_hr if 0 < high_res_stages else block_size_lr,
                                                  grid_size=grid_size_hr if 0 < high_res_stages else block_size_lr,
                                                  lrelu_slope=lrelu_slope,
                                                  block_gmlp_factor=block_gmlp_factor,
                                                  grid_gmlp_factor=grid_gmlp_factor,
                                                  input_proj_factor=input_proj_factor,
                                                  channels_reduction=channels_reduction,
                                                  dropout_rate=dropout_rate,
                                                  use_bias=bias,
                                                  use_global_mlp=use_global_mlp
                                                  )
        self.DC1 = DC(1 * features)
        self.Process_Module_l2 = UNetEncoderBlock(cin=3 * features, num_channels=2 * features,
                                                  block_size=block_size_hr if 1 < high_res_stages else block_size_lr,
                                                  grid_size=grid_size_hr if 1 < high_res_stages else block_size_lr,
                                                  lrelu_slope=lrelu_slope,
                                                  block_gmlp_factor=block_gmlp_factor,
                                                  grid_gmlp_factor=grid_gmlp_factor,
                                                  input_proj_factor=input_proj_factor,
                                                  channels_reduction=channels_reduction,
                                                  dropout_rate=dropout_rate,
                                                  use_bias=bias,
                                                  use_global_mlp=use_global_mlp
                                                  )

        self.DC2 = DC(2 * features)
        self.Process_Module_l3 = UNetEncoderBlock(cin=4 * features, num_channels=3 * features,
                                                  block_size=block_size_hr if 2 < high_res_stages else block_size_lr,
                                                  grid_size=grid_size_hr if 2 < high_res_stages else block_size_lr,
                                                  lrelu_slope=lrelu_slope,
                                                  block_gmlp_factor=block_gmlp_factor,
                                                  grid_gmlp_factor=grid_gmlp_factor,
                                                  input_proj_factor=input_proj_factor,
                                                  channels_reduction=channels_reduction,
                                                  dropout_rate=dropout_rate,
                                                  use_bias=bias,
                                                  use_global_mlp=use_global_mlp
                                                  )
        self.DC3 = DC(3 * features)

        # U-net bottom
        self.Process_Module_bottom = nn.Sequential(*[BottleneckBlock(block_size=block_size_lr,
                                                                     grid_size=grid_size_lr,
                                                                     features=4 * features,
                                                                     block_gmlp_factor=block_gmlp_factor,
                                                                     grid_gmlp_factor=grid_gmlp_factor,
                                                                     input_proj_factor=input_proj_factor,
                                                                     dropout_rate=dropout_rate, use_bias=bias,
                                                                     channels_reduction=channels_reduction
                                                                     ) for i in range(bottom_blk_num)])
        self.local_fusion = nn.Conv2d(8 * features, 3 * features, kernel_size=1, bias=False)

        # cross gating
        self.UpSampleRatio_0 = UpSampleRatio_(2 ** (-2))  # 0->2
        self.UpSampleRatio_1 = UpSampleRatio(2 * features, 1 * features, 2 ** (-1), bias)  # 1->2
        self.UpSampleRatio_2 = UpSampleRatio(3 * features, 1 * features, 1, bias)  # 2->2
        self.cross_gating_block_2 = CrossGatingBlock(x_features=3 * features,
                                                     num_channels=1 * features,
                                                     block_size=block_size_hr if 2 < high_res_stages else block_size_lr,
                                                     grid_size=grid_size_hr if 2 < high_res_stages else block_size_lr,
                                                     cin_y=3 * features,
                                                     upsample_y=True, use_bias=bias,
                                                     dropout_rate=dropout_rate)

        self.UpSampleRatio_3 = UpSampleRatio_(2 ** (-1))  # 0->1
        self.UpSampleRatio_4 = UpSampleRatio(2 * features, 1 * features, 1, bias)  # 1->1
        self.UpSampleRatio_5 = UpSampleRatio(3 * features, 1 * features, 2, bias)  # 2->1
        self.cross_gating_block_1 = CrossGatingBlock(x_features=3 * features,
                                                     num_channels=1 * features,
                                                     block_size=block_size_hr if 1 < high_res_stages else block_size_lr,
                                                     grid_size=grid_size_hr if 1 < high_res_stages else block_size_lr,
                                                     cin_y=1 * features,
                                                     upsample_y=True, use_bias=bias,
                                                     dropout_rate=dropout_rate)

        self.UpSampleRatio_6 = UpSampleRatio_(1)  # 0->0
        self.UpSampleRatio_7 = UpSampleRatio(2 * features, 1 * features, 2, bias)  # 1->0
        self.UpSampleRatio_8 = UpSampleRatio(3 * features, 1 * features, 4, bias)  # 2->0
        self.cross_gating_block_0 = CrossGatingBlock(x_features=3 * features,
                                                     num_channels=1 * features,
                                                     block_size=block_size_hr if 0 < high_res_stages else block_size_lr,
                                                     grid_size=grid_size_hr if 0 < high_res_stages else block_size_lr,
                                                     cin_y=1 * features,
                                                     upsample_y=True, use_bias=bias,
                                                     dropout_rate=dropout_rate)

        # U-net right
        self.UpSampleRatio_9 = UpSampleRatio_(1)  # 2->2
        self.UpSampleRatio_10 = UpSampleRatio_(2 ** (-1))  # 1->2
        self.UpSampleRatio_11 = UpSampleRatio_(2 ** (-2))  # 0->2
        self.UC3 = UC(3 * features)
        self.Process_Module_r3 = UNetDecoderBlock(cin=6 * features,
                                                  num_channels=3 * features,
                                                  lrelu_slope=lrelu_slope,
                                                  block_size=block_size_hr if 2 < high_res_stages else block_size_lr,
                                                  grid_size=grid_size_hr if 2 < high_res_stages else block_size_lr,
                                                  block_gmlp_factor=block_gmlp_factor,
                                                  grid_gmlp_factor=grid_gmlp_factor,
                                                  input_proj_factor=input_proj_factor,
                                                  channels_reduction=channels_reduction,
                                                  use_global_mlp=use_global_mlp,
                                                  dropout_rate=dropout_rate,
                                                  use_bias=bias
                                                  )

        self.UpSampleRatio_12 = UpSampleRatio_(2)  # 2->1
        self.UpSampleRatio_13 = UpSampleRatio_(1)  # 1->1
        self.UpSampleRatio_14 = UpSampleRatio_(2 ** (-1))  # 0->1
        self.UC2 = UC(3 * features)
        self.Process_Module_r2 = UNetDecoderBlock(cin=6 * features,
                                                  num_channels=2 * features,
                                                  lrelu_slope=lrelu_slope,
                                                  block_size=block_size_hr if 1 < high_res_stages else block_size_lr,
                                                  grid_size=grid_size_hr if 1 < high_res_stages else block_size_lr,
                                                  block_gmlp_factor=block_gmlp_factor,
                                                  grid_gmlp_factor=grid_gmlp_factor,
                                                  input_proj_factor=input_proj_factor,
                                                  channels_reduction=channels_reduction,
                                                  use_global_mlp=use_global_mlp,
                                                  dropout_rate=dropout_rate,
                                                  use_bias=bias
                                                  )

        self.UpSampleRatio_15 = UpSampleRatio_(4)  # 2->0
        self.UpSampleRatio_16 = UpSampleRatio_(2)  # 1->0
        self.UpSampleRatio_17 = UpSampleRatio_(1)  # 0->0
        self.UC1 = UC(2 * features)
        self.Process_Module_r1 = UNetDecoderBlock(cin=5 * features,
                                                  num_channels=2 * features,
                                                  lrelu_slope=lrelu_slope,
                                                  block_size=block_size_hr if 0 < high_res_stages else block_size_lr,
                                                  grid_size=grid_size_hr if 0 < high_res_stages else block_size_lr,
                                                  block_gmlp_factor=block_gmlp_factor,
                                                  grid_gmlp_factor=grid_gmlp_factor,
                                                  input_proj_factor=input_proj_factor,
                                                  channels_reduction=channels_reduction,
                                                  use_global_mlp=use_global_mlp,
                                                  dropout_rate=dropout_rate,
                                                  use_bias=bias
                                                  )

        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(128, 4 * features)
        )

        # AffineTransform
        self.AffineTransform2 = ConvAct(128, features, 1)
        self.AffineTransform1 = ConvAct(64, features, 1)
        self.AffineTransform0 = ConvAct(32, features, 1)

        # Finetune
        self.Finetune_Module = nn.Sequential(
            nn.Conv2d(2 * features, num_outputs, kernel_size=3, stride=1, padding=1, bias=bias)
        )

    def forward(self, x, q, feas):
        input_img = x
        q = self.mlp(q).unsqueeze(-1).unsqueeze(-1)
        fea2 = self.AffineTransform2(feas[2])
        fea1 = self.AffineTransform1(feas[1])
        fea0 = self.AffineTransform0(feas[0])

        scale0 = self.stage_input_conv_0(x)
        scale1 = self.stage_input_conv_1(nearest_downsample(x, 1. / (2 ** 1)))
        scale2 = self.stage_input_conv_2(nearest_downsample(x, 1. / (2 ** 2)))

        # encoder
        out_process_l1 = self.Process_Module_l1(scale0, skip=scale0)
        in_process_l2 = self.DC1(out_process_l1, fea0)
        out_process_l2 = self.Process_Module_l2(in_process_l2, skip=scale1)
        in_process_l3 = self.DC2(out_process_l2, fea1)
        out_process_l3 = self.Process_Module_l3(in_process_l3, skip=scale2)
        in_process_bottom = self.DC3(out_process_l3, fea2)

        # bottleneck
        x = self.Process_Module_bottom(in_process_bottom)
        x = torch.cat([x, x * q], dim=1)
        x = self.local_fusion(x)
        global_feature = x

        # cross gating
        skip_features = []
        for i in reversed(range(3)):
            if i == 2:
                signal0 = self.UpSampleRatio_0(out_process_l1)
                signal1 = self.UpSampleRatio_1(out_process_l2)
                signal2 = self.UpSampleRatio_2(out_process_l3)
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                skips, global_feature = self.cross_gating_block_2(signal, global_feature)
                skip_features.append(skips)
            elif i == 1:
                signal0 = self.UpSampleRatio_3(out_process_l1)
                signal1 = self.UpSampleRatio_4(out_process_l2)
                signal2 = self.UpSampleRatio_5(out_process_l3)
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                skips, global_feature = self.cross_gating_block_1(signal, global_feature)
                skip_features.append(skips)
            elif i == 0:
                signal0 = self.UpSampleRatio_6(out_process_l1)
                signal1 = self.UpSampleRatio_7(out_process_l2)
                signal2 = self.UpSampleRatio_8(out_process_l3)
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                skips, global_feature = self.cross_gating_block_0(signal, global_feature)
                skip_features.append(skips)

        # decoder
        for i in reversed(range(3)):
            if i == 2:
                signal2 = self.UpSampleRatio_9(skip_features[0])
                signal1 = self.UpSampleRatio_10(skip_features[1])
                signal0 = self.UpSampleRatio_11(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                x = self.UC3(x)
                x = self.Process_Module_r3(x, bridge=signal)
            elif i == 1:
                signal2 = self.UpSampleRatio_12(skip_features[0])
                signal1 = self.UpSampleRatio_13(skip_features[1])
                signal0 = self.UpSampleRatio_14(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                x = self.UC2(x)
                x = self.Process_Module_r2(x, bridge=signal)
            elif i == 0:
                signal2 = self.UpSampleRatio_15(skip_features[0])
                signal1 = self.UpSampleRatio_16(skip_features[1])
                signal0 = self.UpSampleRatio_17(skip_features[2])
                signal = torch.cat([signal2, signal1, signal0], dim=1)
                x = self.UC1(x)
                x = self.Process_Module_r1(x, bridge=signal)

        x = self.Finetune_Module(x) + input_img
        return x


if __name__ == '__main__':
    from thop import profile
    model = ConStyleMaxim(Train=False)
    input = torch.randn(2, 3, 128, 128)

    q, fea = model.ConStyle(input)
    flops, _ = profile(model, inputs=(input, q[1], fea))
    print('Total params: %.2f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total gflops: %.2f' % (flops / 1e9))

# Total params: 8.10 M
# Total gflops: 25.58