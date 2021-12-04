# pragma once
#include <ATen/ATen.h>

at::Tensor grained_sample_compress(
    const at::Tensor & input,
    const at::Tensor & patches,
    const int height,
    const int width,
    const int num_extra_tokens
);

at::Tensor grained_sample_decompress(
    const at::Tensor & input,
    const at::Tensor & patches,
    const at::Tensor & reference,
    const int height,
    const int width,
    const int num_extra_tokens,
    const bool is_add_ref
);

std::tuple<at::Tensor, at::Tensor>
grained_sample_index_to_patch(
    const at::Tensor & indices,
    const at::Tensor & split_sizes,
    const int region_size,
    const int num_extra_tokens
);

std::tuple<at::Tensor, at::Tensor>
grained_sample_index_to_patch_dense(
    const at::Tensor & indices,
    const at::Tensor & split_sizes,
    const int region_size,
    const int num_extra_tokens
);
