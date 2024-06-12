import torch
from torch import nn as nn
from utils.registry import ARCH_REGISTRY
from archs.ConStyle_arch import ConStyle


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = torch.rand(train_size)
        with torch.no_grad():
            self.forward(imgs)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ConvAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, padding=None, bias=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        super(ConvAct, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=bias),
            nn.LeakyReLU(0.1, True)
        )


class LinerBNAct(nn.Sequential):
    def __init__(self, dim_i, dim_o):
        super(LinerBNAct, self).__init__(
            nn.Linear(dim_i, dim_o),
            nn.BatchNorm1d(dim_o),
            nn.LeakyReLU(0.1, True)
        )


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


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


class DC(nn.Module):
    def __init__(self, n_feat):
        super(DC, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x, fea):
        x = self.down(x)
        x = torch.cat([x, fea], 1)
        return x


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


@ARCH_REGISTRY.register()
class ConStyleNAFNet(nn.Module):
    def __init__(self,
                 img_channel=3,
                 dim=48,
                 left_blk_num=[7, 8, 9],
                 bottom_blk_num=9,
                 right_blk_num=[7, 8, 9],
                 Train=True
                 ):
        super(ConStyleNAFNet, self).__init__()
        # ConStyle
        self.ConStyle = ConStyle(Train=Train)

        # preprocess
        self.Preprocess_Module = nn.Sequential(
            nn.Conv2d(img_channel, dim, kernel_size=3, padding=1, stride=1, groups=1, bias=False)
        )

        # U-net left
        self.Process_Module_l1 = nn.Sequential(*[NAFBlock(dim) for i in range(left_blk_num[0])])
        self.DC1 = DC(dim)
        self.Process_Module_l2 = nn.Sequential(*[NAFBlock(dim * 2) for i in range(left_blk_num[1])])
        self.DC2 = DC(dim * 2)
        self.Process_Module_l3 = nn.Sequential(*[NAFBlock(dim * 3) for i in range(left_blk_num[2])])
        self.DC3 = DC(dim * 3)

        # U-net bottom
        self.Process_Module_bottom = nn.Sequential(*[NAFBlock(dim * 4) for i in range(bottom_blk_num)])
        self.local_fusion = nn.Conv2d(dim * 8, dim * 3, kernel_size=1, bias=False)

        # U-net right
        self.UC3 = UC(dim * 3, dim * 6, dim * 4)
        self.Process_Module_r3 = nn.Sequential(*[NAFBlock(dim * 4) for i in range(right_blk_num[2])])
        self.UC2 = UC(dim * 4, dim * 6, dim * 3)
        self.Process_Module_r2 = nn.Sequential(*[NAFBlock(dim * 3) for i in range(right_blk_num[1])])
        self.UC1 = UC(dim * 3, dim * 4, dim * 2)
        self.Process_Module_r1 = nn.Sequential(*[NAFBlock(dim * 2) for i in range(right_blk_num[0])])

        # Finetune
        self.Finetune_Module = nn.Sequential(
            nn.Conv2d(dim * 2, 3, kernel_size=3, stride=1, padding=1, bias=False)
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
        out_process_l1 = self.Process_Module_l1(x)
        in_process_l2 = self.DC1(out_process_l1, fea0)

        out_process_l2 = self.Process_Module_l2(in_process_l2)
        in_process_l3 = self.DC2(out_process_l2, fea1)

        out_process_l3 = self.Process_Module_l3(in_process_l3)
        in_process_bottom = self.DC3(out_process_l3, fea2)

        # U-net bottom
        bottom = self.Process_Module_bottom(in_process_bottom)
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

    model = ConStyleNAFNet(Train=False)
    input = torch.randn(2, 3, 128, 128)
    q, fea = model.ConStyle(input)
    flops, _ = profile(model, inputs=(input, q[1], fea))
    print('Total params: %.2f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total gflops: %.2f' % (flops / 1e9))


# Total params: 12.74 M
# Total gflops: 46.97

