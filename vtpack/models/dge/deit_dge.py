# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial

from timm.models.vision_transformer import _cfg, _init_vit_weights
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed, DropPath, trunc_normal_

from vtpack.layers import DynamicGrainedEncoder
from vtpack.layers.sparse_ops import batched_sparse_attention, batched_sparse_gemm

__all__ = [
    "deit_dge_s124_tiny_patch16_256", "deit_dge_s124_small_patch16_256",
    "deit_dge_s124_base_patch16_256",
]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def complexity(self, num_inputs, num_queries):
        comp = num_queries * self.fc1.in_features * self.fc1.out_features  # fc1
        comp += num_queries * self.fc1.out_features  # act
        comp += num_queries * self.fc2.in_features * self.fc2.out_features  # fc2
        return comp

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def complexity(self, num_inputs, num_queries):
        num_channels = self.dim
        comp = num_queries * num_channels ** 2  # q embed
        comp += (num_inputs * num_channels ** 2) * 2  # kv embed
        comp += (num_queries * num_inputs * num_channels) * 2  # attention
        comp += num_queries * num_inputs * self.num_heads * 3  # softmax
        comp += num_queries * num_channels ** 2  # proj
        return comp

    def forward(self, x, q, q_lengths):
        B, N, C = x.shape
        q = self.q(q).reshape(-1, self.num_heads, C // self.num_heads).permute(1, 0, 2)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 3, 0, 1, 4)
        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)

        if not self.training:
            x = batched_sparse_attention(q, k, v, q_lengths, self.scale)
            x = x.transpose(0, 1).reshape(-1, C)
        else:
            if (q_lengths.max() - q_lengths.min()) == 0:
                q = q.reshape(self.num_heads, B, -1, C // self.num_heads)
                attn = (q @ k.transpose(-1, -2)) * self.scale
                attn = attn.softmax(dim=-1, dtype=v.dtype)
                attn = self.attn_drop(attn)
                x = (attn @ v).permute(1, 2, 0, 3).reshape(-1, C)
            else:
                kv_lengths = q_lengths.new_full([B], kv.shape[3])
                k = k.reshape(self.num_heads, -1, C // self.num_heads)
                v = v.reshape(self.num_heads, -1, C // self.num_heads)
                attn = batched_sparse_gemm(q, k, q_lengths, kv_lengths, False, True) * self.scale
                attn = attn.softmax(dim=-1, dtype=v.dtype)
                attn = self.attn_drop(attn)
                x = batched_sparse_gemm(attn, v, q_lengths, kv_lengths, False, False)
                x = x.transpose(0, 1).reshape(-1, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, split_sizes=[2, 1]):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dge = DynamicGrainedEncoder(in_channels=dim, split_sizes=split_sizes, complexity_handler=self.complexity)

    def complexity(self, num_inputs, num_queries):
        comp = num_inputs * self.dim * 2  # norm1 and norm2
        comp += self.attn.complexity(num_inputs, num_queries)
        comp += self.mlp.complexity(num_inputs, num_queries)
        return comp

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        q = self.dge.compress(x, H, W)
        q = q + self.drop_path(self.attn(self.norm1(x), self.norm1(q), self.dge.states["batches"]))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        x = self.dge.decompress(q)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init="", cls_token=True, split_sizes=[2, 1]):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if cls_token else 0
        self.num_tokens += 1 if distilled else 0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls_token else None
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                split_sizes=split_sizes)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        assert weight_init in ("jax", "jax_nlhb", "nlhb", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith("jax"):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            if self.cls_token is not None:
                trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)
        self.apply(self._init_dynamic_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def _init_dynamic_weights(self, m):
        if isinstance(m, DynamicGrainedEncoder):
            m.router.init_parameters()

    def complexity(self, x):
        N = x.shape[1] * x.shape[2]
        comp = N * self.num_features * 3  # patch embed
        comp += self.num_features * self.num_classes  # classifier
        comp_static, comp_dynamic = [], []

        def append_complexity(m):
            if isinstance(m, DynamicGrainedEncoder):
                comp = m.get_complexity()
                comp_static.append(comp["static"])
                comp_dynamic.append(comp["dynamic"])

        self.apply(append_complexity)
        comp_static = (sum(comp_static) + comp).mean()
        comp_dynamic = (sum(comp_dynamic) + comp).mean()
        return {"static": comp_static, "dynamic": comp_dynamic}

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.cls_token is None:
            return x
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            if self.cls_token is not None:
                x = self.head(x)
            else:
                x = self.head(x.mean(dim=1))
        return x


@register_model
def deit_dge_s124_tiny_patch16_256(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=256, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), split_sizes=[4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def deit_dge_s124_small_patch16_256(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=256, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), split_sizes=[4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def deit_dge_s124_base_patch16_256(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), split_sizes=[4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model
