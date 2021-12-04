#pragma once
#include <torch/extension.h>

extern at::Tensor sparse_bilinear_attention_pointwise_forward(
    const at::Tensor & input,
    const at::Tensor & grid,
    const at::Tensor & weight,
    const at::Tensor & batches);
