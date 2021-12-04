import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from vtpack import _C


class _BilinearAttention(Function):

    @staticmethod
    def forward(ctx, input, grid, weight, mode="channel"):
        input = input.contiguous()
        grid = grid.contiguous()
        weight = weight.contiguous()

        ctx.mode = mode
        ctx.save_for_backward(input, grid, weight)
        if mode == "channel":
            output = _C.bilinear_attention_channelwise_forward(input, grid, weight)
        elif mode == "point":
            output = _C.bilinear_attention_pointwise_forward(input, grid, weight)
        else:
            raise NotImplementedError(f"Mode {mode} is not supported")
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()

        mode = ctx.mode
        input, grid, weight = ctx.saved_tensors
        if mode == "channel":
            grad_input, grad_grid, grad_weight = _C.bilinear_attention_channelwise_backward(
                grad_output, input, grid, weight)
        elif mode == "point":
            grad_input, grad_grid, grad_weight = _C.bilinear_attention_pointwise_backward(
                grad_output, input, grid, weight)
        else:
            raise NotImplementedError(f"Mode {mode} is not supported")
        return grad_input, grad_grid, grad_weight, None


bilinear_attention = _BilinearAttention.apply
