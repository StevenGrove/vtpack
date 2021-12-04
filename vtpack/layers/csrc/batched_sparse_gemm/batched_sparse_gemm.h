#pragma once
#include <torch/extension.h>

at::Tensor batched_sparse_gemm_forward(
    const at::Tensor a,
    const at::Tensor b,
    const at::Tensor a_lengths,
    const at::Tensor b_lengths,
    bool is_a_transposed,
    bool is_b_transposed
);

at::Tensor batched_sparse_gemm_v2_forward(
    const at::Tensor a,
    const at::Tensor b,
    const at::Tensor a_lengths,
    const at::Tensor b_lengths,
    bool is_a_transposed,
    bool is_b_transposed
);

at::Tensor batched_sparse_attention_forward(
    const at::Tensor a,
    const at::Tensor b,
    const at::Tensor c,
    const at::Tensor batch_lengths,
    const float scale
);
