import torch
from torch import nn as nn
from utils.registry import ARCH_REGISTRY


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


@ARCH_REGISTRY.register()
class OriginConv(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=48,
                 out_channels=3,
                 ):
        super(OriginConv, self).__init__()

        # preprocess
        self.Preprocess_Module = ConvAct(inp_channels, dim, 3)

        # U-net left
        self.Process_Module_l1 = nn.Sequential(
            ConvAct(dim, dim, 3),
            ConvAct(dim, dim, 3),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

        self.Process_Module_l2 = nn.Sequential(
            ConvAct(dim * 2, dim * 2, 3),
            ConvAct(dim * 2, dim * 2, 3),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

        self.Process_Module_l3 = nn.Sequential(
            ConvAct(dim * 4, dim * 4, 3),
            ConvAct(dim * 4, dim * 4, 3),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

        # U-net bottom
        self.Process_Module_bottom = nn.Sequential(
            ConvAct(dim * 4, dim * 4, 3),
            ConvAct(dim * 4, dim * 4, 3),
        )

        # U-net right
        self.up3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.Process_Module_r3 = nn.Sequential(
            ConvAct(dim * 4, dim * 4, 3),
            ConvAct(dim * 4, dim * 4, 3),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.Process_Module_r2 = nn.Sequential(
            ConvAct(dim * 2, dim * 2, 3),
            ConvAct(dim * 2, dim * 2, 3),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.Process_Module_r1 = nn.Sequential(
            ConvAct(dim * 2, dim * 2, 3),
            ConvAct(dim * 2, dim * 2, 3),
        )

        # Finetune
        self.Finetune_Module = nn.Sequential(
            ConvAct(dim * 2, dim, 3),
            ConvAct(dim, out_channels, 3),
        )


    def forward(self, inp_img):
        # Preprocess
        x = self.Preprocess_Module(inp_img)

        # U-net left
        x = self.Process_Module_l1(x) + x
        x = self.down1(x)
        x = self.Process_Module_l2(x) + x
        x = self.down2(x)
        x = self.Process_Module_l3(x) + x
        x = self.down3(x)

        # U-net bottom
        x = self.Process_Module_bottom(x) + x

        # U-net right
        x = self.up3(x)
        x = self.Process_Module_r3(x) + x
        x = self.up2(x)
        x = self.Process_Module_r2(x) + x
        x = self.up1(x)
        x = self.Process_Module_r1(x) + x

        # Finetune
        x = self.Finetune_Module(x) + inp_img
        return x


if __name__ == '__main__':
    from thop import profile

    model = OriginConv()
    input = torch.randn(2, 3, 128, 128)
    flops, _ = profile(model, inputs=(input, ))
    print('Total params: %.2f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total gflops: %.2f' % (flops / 1e9))

# Total params: 5.03 M
# Total gflops: 19.64
# Forward: 3.0 us
