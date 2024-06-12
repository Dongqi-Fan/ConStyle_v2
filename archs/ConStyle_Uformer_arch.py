import torch
from torch import nn as nn
from utils.registry import ARCH_REGISTRY
from archs.ConStyle_arch import ConStyle
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math


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


class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout=0.,
                 last_stage=False, bias=True):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q, k, v


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'


class Attention(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x


class FastLeFF(nn.Module):

    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.):
        super().__init__()

        from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                                    act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        return flops


def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 modulator=False, cross_modulator=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size * win_size, dim)  # modulator
        else:
            self.modulator = None

        if cross_modulator:
            self.cross_modulator = nn.Embedding(win_size * win_size, dim)  # cross_modulator
            self.cross_attn = Attention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                        proj_drop=drop,
                                        token_projection=token_projection, )
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_modulator = None

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn', 'mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'leff':
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        elif token_mlp == 'fastleff':
            self.mlp = FastLeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        if self.cross_modulator is not None:
            shortcut = x
            x_cross = self.norm_cross(x)
            x_cross = self.cross_attn(x, self.cross_modulator.weight)
            x = shortcut + x_cross

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows

        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x


class BasicUformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='ffn', shift_flag=True,
                 modulator=False, cross_modulator=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, win_size=win_size,
                                      shift_size=0 if (i % 2 == 0) else win_size // 2,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                      modulator=modulator, cross_modulator=cross_modulator)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, win_size=win_size,
                                      shift_size=0,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                                      modulator=modulator, cross_modulator=cross_modulator)
                for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x


class DC(nn.Module):
    def __init__(self, n_feat):
        super(DC, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.PixelUnshuffle(2))

    def forward(self, x, fea):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.down(x)
        x = torch.cat([x, fea], 1)
        x = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return x


class UC(nn.Module):
    def __init__(self, n_feat, down_in, down_out):
        super(UC, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.PixelShuffle(2))
        self.down = nn.Conv2d(down_in, down_out, kernel_size=1, stride=1, bias=False)

    def forward(self, x, fea):
        B, L, C = x.shape
        B_fea, L_fea, C_fea = fea.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        H_fea = int(math.sqrt(L_fea))
        W_fea = int(math.sqrt(L_fea))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        fea = fea.transpose(1, 2).contiguous().view(B_fea, C_fea, H_fea, W_fea)
        x = self.up(x)
        x = self.down(torch.cat([x, fea], 1))
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class LocalFusion(nn.Module):
    def __init__(self, dim):
        super(LocalFusion, self).__init__()
        self.fusion = nn.Conv2d(dim * 8, dim * 3, kernel_size=1, bias=False)

    def forward(self, x, q):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = torch.cat([x, x * q], dim=1)
        x = self.fusion(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


#@ARCH_REGISTRY.register()
class ConStyleUformer(nn.Module):
    def __init__(self,
                 img_size=128,
                 in_chans=3,
                 dd_in=3,
                 dim=32,
                 depths=[2, 2, 2, 2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8, 8, 4, 2],
                 win_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 token_projection='linear',
                 token_mlp='leff',
                 shift_flag=True,
                 modulator=True,
                 Train=True,
                 cross_modulator=False,
                 **kwargs):
        super(ConStyleUformer, self).__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.dim = dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # ConStyle
        self.ConStyle = ConStyle(Train)

        # preprocess
        self.Preprocess_Module = InputProj(in_channel=dd_in, out_channel=dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)

        # U-net left
        self.Process_Module_l1 = BasicUformerLayer(dim=dim,
                                                   output_dim=dim,
                                                   input_resolution=(img_size, img_size),
                                                   depth=depths[0],
                                                   num_heads=num_heads[0],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   shift_flag=shift_flag
                                                   )
        self.DC1 = DC(dim)
        self.Process_Module_l2 = BasicUformerLayer(dim=dim * 2,
                                                   output_dim=dim * 2,
                                                   input_resolution=(img_size // 2, img_size // 2),
                                                   depth=depths[1],
                                                   num_heads=num_heads[1],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   shift_flag=shift_flag
                                                   )
        self.DC2 = DC(dim * 2)
        self.Process_Module_l3 = BasicUformerLayer(dim=dim * 3,
                                                   output_dim=dim * 3,
                                                   input_resolution=(img_size // 4, img_size // 4),
                                                   depth=depths[2],
                                                   num_heads=num_heads[2],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   shift_flag=shift_flag)
        self.DC3 = DC(dim * 3)

        # U-net bottom
        self.Process_Module_bottom = BasicUformerLayer(dim=dim * 4,
                                                       output_dim=dim * 4,
                                                       input_resolution=(img_size // 8, img_size // 8),
                                                       depth=depths[3],
                                                       num_heads=num_heads[3],
                                                       win_size=win_size,
                                                       mlp_ratio=self.mlp_ratio,
                                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop=drop_rate, attn_drop=attn_drop_rate,
                                                       drop_path=conv_dpr,
                                                       norm_layer=norm_layer,
                                                       use_checkpoint=use_checkpoint,
                                                       token_projection=token_projection, token_mlp=token_mlp, shift_flag=shift_flag)
        self.local_fusion = LocalFusion(dim)

        # U-net right
        self.UC3 = UC(dim * 3, dim * 6, dim * 4)
        self.Process_Module_r3 = BasicUformerLayer(dim=dim * 4,
                                                   output_dim=dim * 4,
                                                   input_resolution=(img_size // 4, img_size // 4),
                                                   depth=depths[4],
                                                   num_heads=num_heads[4],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[:depths[4]],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   shift_flag=shift_flag,
                                                   modulator=modulator, cross_modulator=cross_modulator)
        self.UC2 = UC(dim * 4, dim * 6, dim * 3)
        self.Process_Module_r2 = BasicUformerLayer(dim=dim * 3,
                                                   output_dim=dim * 3,
                                                   input_resolution=(img_size // 2, img_size // 2),
                                                   depth=depths[5],
                                                   num_heads=num_heads[5],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[sum(depths[4:5]):sum(depths[4:6])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   shift_flag=shift_flag,
                                                   modulator=modulator, cross_modulator=cross_modulator)
        self.UC1 = UC(dim * 3, dim * 4, dim * 2)
        self.Process_Module_r1 = BasicUformerLayer(dim=dim * 2,
                                                   output_dim=dim * 2,
                                                   input_resolution=(img_size, img_size),
                                                   depth=depths[6],
                                                   num_heads=num_heads[6],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   shift_flag=shift_flag,
                                                   modulator=modulator, cross_modulator=cross_modulator)

        # Finetune
        self.Finetune_Module = OutputProj(in_channel=2 * dim, out_channel=in_chans, kernel_size=3, stride=1)

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
        bottom = self.local_fusion(bottom, q)

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

    model = ConStyleUformer(Train=False)
    input = torch.randn(2, 3, 128, 128)
    q, fea = model.ConStyle(input)
    flops, _ = profile(model, inputs=(input, q[1], fea))
    print('Total params: %.2f M' % (sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total gflops: %.2f' % (flops / 1e9))

# Total params: 4.14 M
# Total gflops: 13.48
