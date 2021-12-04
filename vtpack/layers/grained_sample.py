import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from vtpack import _C


class _GrainedSampleCompress(Function):

    @staticmethod
    def forward(ctx, input, patches, height, width, num_extra_tokens):
        input = input.contiguous()
        patches = patches.contiguous()

        ctx.height = height
        ctx.width = width
        ctx.num_extra_tokens = num_extra_tokens
        ctx.save_for_backward(input, patches)

        output = _C.grained_sample_compress(input, patches, height, width, num_extra_tokens)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()

        input, patches, = ctx.saved_tensors
        height = ctx.height
        width = ctx.width
        num_extra_tokens = ctx.num_extra_tokens

        grad_output *= 1 / (patches[:, -1, None] ** 2)
        grad_input = _C.grained_sample_decompress(grad_output, patches, input, height, width, num_extra_tokens, False)
        return grad_input, None, None, None, None


grained_sample_compress = _GrainedSampleCompress.apply


class _GrainedSampleDecompress(Function):

    @staticmethod
    def forward(ctx, input, patches, ref_input, height, width, num_extra_tokens, is_add_ref):
        input = input.contiguous()
        patches = patches.contiguous()
        ref_input = ref_input.contiguous()

        ctx.height = height
        ctx.width = width
        ctx.num_extra_tokens = num_extra_tokens
        ctx.is_add_ref = is_add_ref
        ctx.save_for_backward(patches)

        output = _C.grained_sample_decompress(input, patches, ref_input, height, width, num_extra_tokens, is_add_ref)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()

        patches, = ctx.saved_tensors
        height = ctx.height
        width = ctx.width
        num_extra_tokens = ctx.num_extra_tokens

        grad_input = _C.grained_sample_compress(grad_output, patches, height, width, num_extra_tokens)
        grad_input *= patches[:, -1, None] ** 2

        if ctx.is_add_ref:
            grad_ref_input = grad_output
        else:
            grad_ref_input = torch.zeros_like(grad_output)
        return grad_input, None, grad_ref_input, None, None, None, None


grained_sample_decompress = _GrainedSampleDecompress.apply


class _GrainedSampleIndexToPatch(Function):

    @staticmethod
    def forward(ctx, indices, split_sizes, region_size, num_extra_tokens):
        indices = indices.contiguous()
        split_sizes = split_sizes.contiguous()

        patches, batch_lengths = _C.grained_sample_index_to_patch(
            indices, split_sizes, region_size, num_extra_tokens
        )
        return patches, batch_lengths

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        return None, None, None, None


grained_sample_index_to_patch = _GrainedSampleIndexToPatch.apply


class _GrainedSampleIndexToPatchDense(Function):

    @staticmethod
    def forward(ctx, indices, split_sizes, region_size, num_extra_tokens):
        indices = indices.contiguous()
        split_sizes = split_sizes.contiguous()

        patches, batch_lengths = _C.grained_sample_index_to_patch_dense(
            indices, split_sizes, region_size, num_extra_tokens
        )
        return patches, batch_lengths

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        return None, None, None, None


grained_sample_index_to_patch_dense = _GrainedSampleIndexToPatchDense.apply


def grained_sample_compress_dense(
    x, split_sizes, region_size, H, W, num_extra_tokens
):
    B, N, C = x.shape
    x_s = x[:, num_extra_tokens:, :].reshape(B, H, W, C)

    comp = []
    for i, win in enumerate(split_sizes):
        comp_i = F.avg_pool2d(
            x_s.permute(0, 3, 1, 2),
            kernel_size=region_size // win.item(),
            stride=region_size // win.item()
        ).permute(0, 2, 3, 1)
        comp.append(comp_i.reshape(B, -1, C))

    if num_extra_tokens > 0:
        comp.insert(0, x[:, :num_extra_tokens, :])

    comp = torch.cat(comp, dim=1)
    batches = torch.full([B], comp.shape[1], dtype=torch.long)
    comp = comp.reshape(-1, C)
    return comp, batches


def grained_sample_decompress_dense(
    x, gate, reference, split_sizes, region_size, H, W, num_extra_tokens, is_add_ref
):
    B, N, C = reference.shape
    H_r, W_r = gate.shape[1:3]
    x = x.reshape(B, -1, C)

    out_s = []
    start_idx = num_extra_tokens
    for i, win in enumerate(split_sizes):
        step = region_size // win.item()
        H_i, W_i = H // step, W // step
        x_i = x[:, start_idx:start_idx + H_i * W_i, :]
        x_i = x_i.reshape(B, H_i, 1, W_i, 1, C)
        x_i = x_i.repeat(1, 1, step, 1, step, 1)
        x_i = x_i.reshape(B, H_r, H // H_r, W_r, W // W_r, C)
        gate_i = gate[..., i].reshape(B, H_r, 1, W_r, 1, 1)
        out_s.append((x_i * gate_i).reshape(B, H, W, C))
        start_idx = start_idx + H_i * W_i
    out_s = sum(out_s).reshape(B, -1, C)

    reference = reference if is_add_ref else 0
    if num_extra_tokens > 0:
        output = torch.cat([x[:, :num_extra_tokens, :], out_s], dim=1) + reference
    else:
        output = out_s + reference

    return output
