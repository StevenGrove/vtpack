import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from vtpack.layers.bilinear_attention import bilinear_attention

__all__ = [
    "dpvt_tiny_256", "dpvt_small_256", "dpvt_medium_256", "dpvt_large_256"
]


class GradScale(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, scale=1.0):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


grad_scale = GradScale.apply


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., num_points=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.num_points = num_points

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_weight = nn.Linear(dim, self.num_heads * num_points)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.offsets = nn.Linear(dim, num_heads * num_points * 2)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32)
        thetas = thetas * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, i, :] *= i + 1
        self.offsets.bias = nn.Parameter(grid_init.view(-1))

    @torch.no_grad()
    def get_reference_points(self, x, H, W):
        dh = torch.linspace(0, H - 1, H, device=x.device, dtype=x.dtype)
        dw = torch.linspace(0, W - 1, W, device=x.device, dtype=x.dtype)
        mh, mw = torch.meshgrid(dh, dw, indexing="ij")
        ref_points = torch.stack([mw, mh], dim=-1)
        return ref_points.reshape(1, H * W, 1, 2)

    def forward(self, x, H, W):
        B, N, C = x.shape

        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3).reshape(-1, H, W, C // self.num_heads)

        ref_points = self.get_reference_points(v, H, W)
        offsets = self.offsets(x).reshape(B, N, self.num_heads, self.num_points, 2)
        offsets = offsets.permute(0, 2, 1, 3, 4).reshape(-1, N, self.num_points, 2)

        grid = grad_scale(ref_points + offsets, 0.1)

        weight = self.attn_weight(x).reshape(B, N, self.num_heads, self.num_points)
        weight = weight.permute(0, 2, 1, 3).reshape(-1, N, self.num_points)
        weight = weight.softmax(dim=-1, dtype=v.dtype)
        weight = self.attn_drop(weight)

        x = bilinear_attention(v, grid, weight, "point")
        x = x.reshape(B, self.num_heads, N, C // self.num_heads)
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_points=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, num_points=num_points)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_points=[8, 4, 2, 1], num_stages=4):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, num_points=num_points[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        self.norm = norm_layer(embed_dims[3]) if num_classes > 0 else nn.Identity()

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

        # feature attributes
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        self._out_feature_channels = {f"res{2+i}": dim for i, dim in enumerate(embed_dims)}
        self._size_divisibility = 32

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear", align_corners=False).reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]

        outputs = {}
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)

            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outputs[f"res{i+2}"] = x
            elif self.num_classes == 0:
                outputs[f"res{i+2}"] = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        if self.num_classes == 0:
            return outputs

        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)

        if self.num_classes > 0:
            x = self.head(x)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def dpvt_tiny_256(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        img_size=256, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], num_points=[4, 4, 4, 4],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def dpvt_small_256(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        img_size=256, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], num_points=[4, 4, 4, 4],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def dpvt_medium_256(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        img_size=256, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], num_points=[4, 4, 4, 4],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def dpvt_large_256(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        img_size=256, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], num_points=[4, 4, 4, 4],
        **kwargs)
    model.default_cfg = _cfg()

    return model
