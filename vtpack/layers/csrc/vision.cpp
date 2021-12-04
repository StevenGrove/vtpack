#pragma once
#include <torch/extension.h>

#include "batched_sparse_gemm/batched_sparse_gemm.h"
#include "grained_sample/grained_sample.h"
#include "bilinear_attention/bilinear_attention.h"
#include "sparse_bilinear_attention/sparse_bilinear_attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    /* batched sparse gemm */
    m.def("batched_sparse_gemm_forward", &batched_sparse_gemm_forward, "batched_sparse_gemm_forward");
    m.def("batched_sparse_gemm_v2_forward", &batched_sparse_gemm_v2_forward, "batched_sparse_gemm_v2_forward");
    m.def("batched_sparse_attention_forward", &batched_sparse_attention_forward, "batched_sparse_attention_forward");

    /* grained sample */
    m.def("grained_sample_compress", &grained_sample_compress, "grained sample compress");
    m.def("grained_sample_decompress", &grained_sample_decompress, "grained sample decompress");
    m.def("grained_sample_index_to_patch", &grained_sample_index_to_patch, "grained sample index to patch");
    m.def("grained_sample_index_to_patch_dense", &grained_sample_index_to_patch_dense, "grained sample index to patch dense");

    /* bilinear attention */
    m.def("bilinear_attention_pointwise_forward", &bilinear_attention_pointwise_forward, "bilinear attention pointwise forward");
    m.def("bilinear_attention_pointwise_backward", &bilinear_attention_pointwise_backward, "bilinear attention pointwise backward");
    m.def("bilinear_attention_channelwise_forward", &bilinear_attention_channelwise_forward, "bilinear attention channelwise forward");
    m.def("bilinear_attention_channelwise_backward", &bilinear_attention_channelwise_backward, "bilinear attention channelwise backward");

    /* sparse bilinear attention */
    m.def("sparse_bilinear_attention_pointwise_forward", &sparse_bilinear_attention_pointwise_forward, "sparse bilinear attention pointwise forward");
}
