from archs.ConStyle_arch import ConStyle
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from utils.registry import ARCH_REGISTRY


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # print('mu, var', mu.mean(), var.mean())
        # d.append([mu.mean(), var.mean()])
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class KBAFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, att, selfk, selfg, selfb, selfw):
        B, nset, H, W = att.shape
        KK = selfk ** 2
        selfc = x.shape[1]

        att = att.reshape(B, nset, H * W).transpose(-2, -1)

        ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset = selfk, selfg, selfc, KK, nset
        ctx.x, ctx.att, ctx.selfb, ctx.selfw = x, att, selfb, selfw

        bias = att @ selfb
        attk = att @ selfw

        uf = torch.nn.functional.unfold(x, kernel_size=selfk, padding=selfk // 2)

        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        x = attk @ uf.unsqueeze(-1)  #
        del attk, uf
        x = x.squeeze(-1).reshape(B, H * W, selfc) + bias
        x = x.transpose(-1, -2).reshape(B, selfc, H, W)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, att, selfb, selfw = ctx.x, ctx.att, ctx.selfb, ctx.selfw
        selfk, selfg, selfc, KK, nset = ctx.selfk, ctx.selfg, ctx.selfc, ctx.KK, ctx.nset

        B, selfc, H, W = grad_output.size()

        dbias = grad_output.reshape(B, selfc, H * W).transpose(-1, -2)

        dselfb = att.transpose(-2, -1) @ dbias
        datt = dbias @ selfb.transpose(-2, -1)

        attk = att @ selfw
        uf = F.unfold(x, kernel_size=selfk, padding=selfk // 2)
        # for unfold att / less memory cost
        uf = uf.reshape(B, selfg, selfc // selfg * KK, H * W).permute(0, 3, 1, 2)
        attk = attk.reshape(B, H * W, selfg, selfc // selfg, selfc // selfg * KK)

        dx = dbias.view(B, H * W, selfg, selfc // selfg, 1)

        dattk = dx @ uf.view(B, H * W, selfg, 1, selfc // selfg * KK)
        duf = attk.transpose(-2, -1) @ dx
        del attk, uf

        dattk = dattk.view(B, H * W, -1)
        datt += dattk @ selfw.transpose(-2, -1)
        dselfw = att.transpose(-2, -1) @ dattk

        duf = duf.permute(0, 2, 3, 4, 1).view(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=selfk, padding=selfk // 2)

        datt = datt.transpose(-1, -2).view(B, nset, H, W)

        return dx, datt, None, None, dselfb, dselfw


class KBBlock_s(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, nset=32, k=3, gc=4, lightweight=False):
        super(KBBlock_s, self).__init__()
        self.k, self.c = k, c
        self.nset = nset
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        if not lightweight:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,
                          bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                          bias=True),
            )

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                                bias=True)

        interc = min(c, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.sg = SimpleGate()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def init_p(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        sca = self.sca(x)
        x1 = self.conv11(x)

        # KBA module
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
        x = x * x1 * sca

        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        # FFN
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma


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


@ARCH_REGISTRY.register()
class ConStyleKBNet(nn.Module):
    def __init__(self,
                 img_channel=3,
                 dim=32,
                 left_blk_num=[2, 2, 2],
                 bottom_blk_num=2,
                 right_blk_num=[2, 2, 2],
                 basicblock='KBBlock_s',
                 lightweight=False,
                 ffn_scale=2,
                 Train=True
                 ):
        super(ConStyleKBNet, self).__init__()
        basicblock = eval(basicblock)

        # ConStyle
        self.ConStyle = ConStyle(Train)

        # preprocess
        self.Preprocess_Module = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        )

        # U-net left
        self.Process_Module_l1 = nn.Sequential(*[basicblock(dim, FFN_Expand=ffn_scale, lightweight=lightweight)
                                                 for i in range(left_blk_num[0])])
        self.DC1 = DC(dim)
        self.Process_Module_l2 = nn.Sequential(*[basicblock(dim * 2, FFN_Expand=ffn_scale, lightweight=lightweight)
                                                 for i in range(left_blk_num[1])])
        self.DC2 = DC(dim * 2)
        self.Process_Module_l3 = nn.Sequential(*[basicblock(dim * 3, FFN_Expand=ffn_scale, lightweight=lightweight)
                                                 for i in range(left_blk_num[2])])
        self.DC3 = DC(dim * 3)

        # U-net bottom
        self.Process_Module_bottom = nn.Sequential(*[basicblock(dim * 4, FFN_Expand=ffn_scale, lightweight=lightweight)
                                                     for i in range(bottom_blk_num)])
        self.local_fusion = nn.Conv2d(dim * 8, dim * 3, kernel_size=1, bias=False)

        # U-net right
        self.UC3 = UC(dim * 3, dim * 6, dim * 4)
        self.Process_Module_r3 = nn.Sequential(*[basicblock(dim * 4, FFN_Expand=ffn_scale, lightweight=lightweight)
                                                 for i in range(right_blk_num[2])])
        self.UC2 = UC(dim * 4, dim * 6, dim * 3)
        self.Process_Module_r2 = nn.Sequential(*[basicblock(dim * 3, FFN_Expand=ffn_scale, lightweight=lightweight)
                                                 for i in range(right_blk_num[1])])
        self.UC1 = UC(dim * 3, dim * 4, dim * 2)
        self.Process_Module_r1 = nn.Sequential(*[basicblock(dim * 2, FFN_Expand=ffn_scale, lightweight=lightweight)
                                                 for i in range(right_blk_num[0])])

        # Finetune
        self.Finetune_Module = nn.Sequential(
            nn.Conv2d(dim * 2, img_channel, kernel_size=3, stride=1, padding=1, bias=False)
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
    model = ConStyleKBNet(Train=False)
    input = torch.randn(2, 3, 128, 128)
    q, fea = model.ConStyle(input)
    flops, _ = profile(model, inputs=(input, q[1], fea))
    print('Total params: %.2f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total gflops: %.2f' % (flops / 1e9))

# Total params: 5.07 M
# Total gflops: 10.52