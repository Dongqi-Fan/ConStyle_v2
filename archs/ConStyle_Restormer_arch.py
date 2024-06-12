## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import torch
from torch import nn as nn
from utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
from archs.ConStyle_arch import ConStyle
from einops import rearrange
import numbers


class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, padding=None, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvAct, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=bias),
            nn.PReLU(out_channels)
        )


class LinerBNAct(nn.Sequential):
    def __init__(self, dim_i, dim_o):
        super(LinerBNAct, self).__init__(
            nn.Linear(dim_i, dim_o),
            nn.BatchNorm1d(dim_o),
            nn.LeakyReLU(0.1, True)
        )


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # @为矩阵乘法，*为逐元素相乘
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
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


class UC(nn.Module):
    def __init__(self, n_feat, down_in, down_out):
        super(UC, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.PixelShuffle(2))
        self.down = nn.Conv2d(down_in, down_out, kernel_size=1, stride=1, bias=False)

    def forward(self, x, fea):
        x = self.up(x)
        x = torch.cat([x, fea], 1)
        x = self.down(x)
        return x


def build_last_conv(conv_type, dim):
    if conv_type == "1conv":
        block = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
    elif conv_type == "3conv":
        # to save parameters and memory
        block = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
        )
    elif conv_type == "1conv1x1":
        block = nn.Conv2d(dim, dim, 1, 1, 0)
    return block


@ARCH_REGISTRY.register()
class ConStyleRestormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=48,
                 left_blk_num=[4, 6, 6],
                 bottom_blk_num=8,
                 right_blk_num=[4, 6, 6],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 out_channels=3,
                 Train=True
                 ):
        super(ConStyleRestormer, self).__init__()

        # ConStyle
        self.ConStyle = ConStyle(Train)

        # preprocess
        self.Preprocess_Module = OverlapPatchEmbed(inp_channels, dim)

        # U-net left
        self.Process_Module_l1 = nn.Sequential(*[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias,
                                                                  LayerNorm_type) for i in range(left_blk_num[0])])
        self.DC1 = DC(dim)
        self.Process_Module_l2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias,
                                                                  LayerNorm_type) for i in range(left_blk_num[1])])
        self.DC2 = DC(dim * 2)
        self.Process_Module_l3 = nn.Sequential(*[TransformerBlock(dim * 3, heads[2], ffn_expansion_factor, bias,
                                                                  LayerNorm_type) for i in range(left_blk_num[2])])
        self.DC3 = DC(dim * 3)

        # U-net bottom
        self.Process_Module_bottom = nn.Sequential(*[TransformerBlock(dim * 4, heads[3], ffn_expansion_factor, bias,
                                                                      LayerNorm_type) for i in range(bottom_blk_num)])
        self.local_fusion = nn.Conv2d(dim * 8, dim * 3, kernel_size=1, bias=bias)

        # U-net right
        self.UC3 = UC(dim * 3, dim * 6, dim * 4)
        self.Process_Module_r3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias,
                                                                  LayerNorm_type) for i in range(right_blk_num[2])])

        self.UC2 = UC(dim * 4, dim * 6, dim * 3)
        self.Process_Module_r2 = nn.Sequential(*[TransformerBlock(dim * 3, heads[1], ffn_expansion_factor, bias,
                                                                  LayerNorm_type) for i in range(right_blk_num[1])])

        self.UC1 = UC(dim * 3, dim * 4, dim * 2)
        self.Process_Module_r1 = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias,
                                                                  LayerNorm_type) for i in range(right_blk_num[0])])

        # Finetune
        self.Finetune_Module = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias,
                                                                LayerNorm_type) for i in range(num_refinement_blocks)])
        self.Finetune_Module.append(nn.Conv2d(dim * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias))

        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(128, dim * 4)
        )

        # AffineTransform
        self.AffineTransform2 = ConvAct(128, dim, 1)
        self.AffineTransform1 = ConvAct(64, dim, 1)
        self.AffineTransform0 = ConvAct(32, dim, 1)

    def forward(self, inp_img, q, feas):
        q = self.mlp(q).unsqueeze(-1).unsqueeze(-1)
        fea2 = self.AffineTransform2(feas[2])
        fea1 = self.AffineTransform1(feas[1])
        fea0 = self.AffineTransform0(feas[0])

        # Preprocess
        x = self.Preprocess_Module(inp_img)

        # U-net left
        out_process_l1 = self.Process_Module_l1(x)
        in_process_l2 = self.DC1(out_process_l1, fea0)

        out_process_l2 = self.Process_Module_l2(in_process_l2)
        in_process_l3 = self.DC2(out_process_l2, fea1)

        out_process_l3 = self.Process_Module_l3(in_process_l3)
        bottom = self.DC3(out_process_l3, fea2)

        # U-net bottom
        bottom = self.Process_Module_bottom(bottom)
        bottom = torch.cat([bottom, bottom * q], dim=1)
        bottom = self.local_fusion(bottom)

        # U-net right
        x = self.UC3(bottom, out_process_l3)
        x = self.Process_Module_r3(x)

        x = self.UC2(x, out_process_l2)
        x = self.Process_Module_r2(x)

        x = self.UC1(x, out_process_l1)
        x = self.Process_Module_r1(x)

        # Finetune
        x = self.Finetune_Module(x) + inp_img
        return x


if __name__ == '__main__':
    from thop import profile

    model = ConStyleRestormer(Train=False)
    input = torch.randn(2, 3, 128, 128)
    q, fea = model.ConStyle(input)
    flops, _ = profile(model, inputs=(input, q[1], fea))
    print('Total params: %.2f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total gflops: %.2f' % (flops / 1e9))

# Total params: 15.57 M
# Total gflops: 74.92