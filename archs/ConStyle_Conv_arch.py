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


@ARCH_REGISTRY.register()
class ConStyleConv(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=48,
                 bias=False,
                 out_channels=3,
                 Train=True
                 ):
        super(ConStyleConv, self).__init__()

        # ConStyle
        self.ConStyle = ConStyle(Train)

        # preprocess
        self.Preprocess_Module = ConvAct(inp_channels, dim, 3)

        # U-net left
        self.Process_Module_l1 = nn.Sequential(
            ConvAct(dim, dim, 3),
            ConvAct(dim, dim, 3),
        )
        self.DC1 = DC(dim)

        self.Process_Module_l2 = nn.Sequential(
            ConvAct(dim * 2, dim * 2, 3),
            ConvAct(dim * 2, dim * 2, 3),
        )
        self.DC2 = DC(dim * 2)

        self.Process_Module_l3 = nn.Sequential(
            ConvAct(dim * 3, dim * 3, 3),
            ConvAct(dim * 3, dim * 3, 3),
        )
        self.DC3 = DC(dim * 3)

        # U-net bottom
        self.Process_Module_bottom = nn.Sequential(
            ConvAct(dim * 4, dim * 4, 3),
            ConvAct(dim * 4, dim * 4, 3),
        )
        self.local_fusion = nn.Conv2d(dim * 8, dim * 3, kernel_size=1, bias=bias)

        # U-net right
        self.UC3 = UC(dim * 3, dim * 6, dim * 4)
        self.Process_Module_r3 = nn.Sequential(
            ConvAct(dim * 4, dim * 4, 3),
            ConvAct(dim * 4, dim * 4, 3),
        )

        self.UC2 = UC(dim * 4, dim * 6, dim * 3)
        self.Process_Module_r2 = nn.Sequential(
            ConvAct(dim * 3, dim * 3, 3),
            ConvAct(dim * 3, dim * 3, 3),
        )

        self.UC1 = UC(dim * 3, dim * 4, dim * 2)
        self.Process_Module_r1 = nn.Sequential(
            ConvAct(dim * 2, dim * 2, 3),
            ConvAct(dim * 2, dim * 2, 3),
        )

        # Finetune
        self.Finetune_Module = nn.Sequential(
            ConvAct(dim * 2, dim, 3),
            ConvAct(dim, out_channels, 3),
        )

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
        out_process_l1 = self.Process_Module_l1(x) + x
        in_process_l2 = self.DC1(out_process_l1, fea0)

        out_process_l2 = self.Process_Module_l2(in_process_l2) + in_process_l2
        in_process_l3 = self.DC2(out_process_l2, fea1)

        out_process_l3 = self.Process_Module_l3(in_process_l3) + in_process_l3
        bottom = self.DC3(out_process_l3, fea2)

        # U-net bottom
        bottom = self.Process_Module_bottom(bottom) + bottom
        bottom = torch.cat([bottom, bottom * q], dim=1)
        bottom = self.local_fusion(bottom)

        # U-net right
        x = self.UC3(bottom, out_process_l3)
        x = self.Process_Module_r3(x) + x

        x = self.UC2(x, out_process_l2)
        x = self.Process_Module_r2(x) + x

        x = self.UC1(x, out_process_l1)
        x = self.Process_Module_r1(x) + x

        # Finetune
        x = self.Finetune_Module(x) + inp_img
        return x


if __name__ == '__main__':
    from thop import profile

    model = ConStyleConv(Train=False)
    input = torch.randn(2, 3, 128, 128)

# Total params: 6.78 M
# Total gflops: 25.90
# Forward: 5.1 us

