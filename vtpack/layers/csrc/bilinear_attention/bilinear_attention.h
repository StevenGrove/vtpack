#pragma once
#include <torch/extension.h>

extern at::Tensor bilinear_attention_pointwise_forward(
    const at::Tensor & input,
    const at::Tensor & grid,
    const at::Tensor & weight);

extern std::tuple<at::Tensor, at::Tensor, at::Tensor>
bilinear_attention_pointwise_backward(
    const at::Tensor & grad_output,
    const at::Tensor & input,
    const at::Tensor & grid,
    const at::Tensor & weight);

extern at::Tensor bilinear_attention_channelwise_forward(
    const at::Tensor & input,
    const at::Tensor & grid,
    const at::Tensor & weight);

extern std::tuple<at::Tensor, at::Tensor, at::Tensor>
bilinear_attention_channelwise_backward(
    const at::Tensor & grad_output,
    const at::Tensor & input,
    const at::Tensor & grid,
    const at::Tensor & weight);
