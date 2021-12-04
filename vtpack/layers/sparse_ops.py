import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from vtpack import _C


class _BatchedSparseGEMM(Function):

    @staticmethod
    def forward(ctx, a, b, a_lengths, b_lengths, is_a_transposed, is_b_transposed):
        a = a.contiguous()
        b = b.contiguous()
        a_lengths = a_lengths.contiguous()
        b_lengths = b_lengths.contiguous()

        ctx.is_a_transposed = is_a_transposed
        ctx.is_b_transposed = is_b_transposed
        ctx.save_for_backward(a, b, a_lengths, b_lengths)

        output = _C.batched_sparse_gemm_forward(
            a, b, a_lengths, b_lengths,
            is_a_transposed, is_b_transposed)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        a, b, a_lengths, b_lengths = ctx.saved_tensors

        if not ctx.is_a_transposed:
            grad_a = _C.batched_sparse_gemm_forward(
                grad_output, b, a_lengths, b_lengths,
                ctx.is_a_transposed, not ctx.is_b_transposed)
        else:
            grad_a = _C.batched_sparse_gemm_forward(
                b, grad_output, b_lengths, a_lengths,
                ctx.is_b_transposed, ctx.is_a_transposed)

        if not ctx.is_b_transposed:
            grad_b = _C.batched_sparse_gemm_forward(
                a, grad_output, a_lengths, a_lengths,
                not ctx.is_a_transposed, ctx.is_b_transposed)
        else:
            grad_b = _C.batched_sparse_gemm_forward(
                grad_output, a, a_lengths, a_lengths,
                ctx.is_b_transposed, ctx.is_a_transposed)

        return grad_a, grad_b, None, None, None, None


batched_sparse_gemm = _BatchedSparseGEMM.apply


class _BatchedSparseAttention(Function):

    @staticmethod
    def forward(ctx, q, k, v, q_lengths, scale):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_lengths = q_lengths.contiguous()
        return _C.batched_sparse_attention_forward(q, k, v, q_lengths, scale)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward is not implemented.")

        return None, None, None, None, None


batched_sparse_attention = _BatchedSparseAttention.apply
