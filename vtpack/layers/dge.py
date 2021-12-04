import math
import torch
import torch.nn.functional as F

from torch import nn
from typing import List
from timm.models.layers import trunc_normal_

from .grained_sample import (
    grained_sample_compress,
    grained_sample_decompress,
    grained_sample_index_to_patch,
    grained_sample_compress_dense,
    grained_sample_decompress_dense,
)
from vtpack import _C


class DynamicGrainedEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        split_sizes: List[int],
        complexity_handler=None,
        sparse_train: bool = False,
    ):
        super().__init__()
        self.region_size = max(split_sizes)
        self.in_channels = in_channels
        self.splits = nn.Parameter(torch.tensor(split_sizes), requires_grad=False)
        self.router = DynamicRouter(in_channels, split_sizes)
        self.complexity_handler = complexity_handler
        self.sparse_train = sparse_train
        self.states = {
            "comp": None, "input": None, "patches": None, "batches": None,
            "gate": None, "size": None, "num_extra_tokens": 0,
        }

    def get_complexity(self):
        if self.complexity_handler is None:
            return {"static": None, "dynamic": None}

        B, N, C = self.states["input"].shape

        gate = self.states["gate"]
        if not self.training:
            indices = gate.argmax(dim=-1, keepdim=True)
            gate = torch.zeros_like(gate).scatter_(-1, indices, 1.0)

        num_queries = gate * (self.splits[None, None, None, :] ** 2)
        num_queries = num_queries.view(B, -1).double().sum(-1)
        num_queries = num_queries + self.states["num_extra_tokens"]
        num_inputs = num_queries.new_full([B], N, dtype=torch.double)

        dge_cp = self.router.complexity(num_inputs, num_queries)
        static_cp = self.complexity_handler(num_inputs, num_inputs) + dge_cp
        dynamic_cp = self.complexity_handler(num_inputs, num_queries) + dge_cp
        return {"static": static_cp, "dynamic": dynamic_cp}

    def compress(self, x, H, W):
        B, N, C = x.shape
        num_extra_tokens = N - H * W

        gate = self.router(x[:, num_extra_tokens:, :].reshape(B, H, W, C))
        if self.training:
            gate = F.gumbel_softmax(gate, dim=-1, hard=True)

        if self.training and not self.sparse_train:
            patches = None
            comp, batches = grained_sample_compress_dense(
                x, self.splits, self.region_size, H, W, num_extra_tokens
            )
        else:
            indices = gate.argmax(dim=-1)
            patches, batches = grained_sample_index_to_patch(
                indices, self.splits, self.region_size, num_extra_tokens
            )
            comp = grained_sample_compress(x, patches, H, W, num_extra_tokens)

        self.states.update({
            "comp": comp, "input": x, "patches": patches, "batches": batches,
            "gate": gate, "size": (H, W), "num_extra_tokens": num_extra_tokens
        })
        return comp

    def decompress(self, x):
        x = x - self.states["comp"]

        if self.training and not self.sparse_train:
            x = grained_sample_decompress_dense(
                x, self.states["gate"], self.states["input"], self.splits, self.region_size,
                *self.states["size"], self.states["num_extra_tokens"], True
            )
        else:
            if self.training:
                B, N, C = self.states["input"].shape
                num_extra_tokens = self.states["num_extra_tokens"]
                gate = self.states["gate"].max(dim=-1, keepdim=True)[0]
                gate = gate.repeat_interleave(self.region_size, 1).repeat_interleave(self.region_size, 2)
                gate = gate.reshape(B, -1, gate.shape[-1])
                if num_extra_tokens > 0:
                    gate = torch.cat([gate.new_ones([B, num_extra_tokens, 1]), gate], dim=1)
                gate = grained_sample_compress(
                    gate, self.states["patches"], *self.states["size"], num_extra_tokens)
                x = x * gate

            x = grained_sample_decompress(
                x, self.states["patches"], self.states["input"], *self.states["size"],
                self.states["num_extra_tokens"], True
            )
        return x


class DynamicRouter(nn.Module):

    def __init__(self, num_channels, split_sizes, init_gate=0.95):
        super(DynamicRouter, self).__init__()
        self.num_channels = num_channels
        self.region_size = max(split_sizes)
        self.split_sizes = split_sizes
        self.init_gate = init_gate

        self.gate_pool = nn.AvgPool2d(self.region_size, self.region_size)
        self.gate = nn.Linear(num_channels, len(split_sizes))

        self.init_parameters()

    def init_parameters(self):
        trunc_normal_(self.gate.weight, std=.01)
        num_splits = len(self.split_sizes)
        if len(self.split_sizes) == 1:
            nn.init.constant_(self.gate.bias.data, 0)
        else:
            bias_value = math.log(math.sqrt(
                self.init_gate * (1 - num_splits) / (self.init_gate - 1)))
            self.gate.bias.data[0] = bias_value
            self.gate.bias.data[1:] = -bias_value

    def complexity(self, num_inputs, num_queries):
        comp = num_inputs / (self.region_size ** 2) * self.num_channels  # gate_pool
        comp += num_inputs / (self.region_size ** 2) * self.num_channels * len(self.split_sizes)  # gate
        return comp

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.gate_pool(x)
        x = x.permute(0, 2, 3, 1)
        x = self.gate(x)
        return x
