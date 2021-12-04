#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10d/Utils.hpp>
#include <cuda.h>

#include <THC/THCAtomics.cuh>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define MAX_STREAM_NUM 32

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_CUDNN(func)                                                      \
{                                                                              \
    cudnnStatus_t status = (func);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudnnGetErrorString(status), status);                 \
    }                                                                          \
}

static inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                                cublasOperation_t transa, cublasOperation_t transb,
                                                int num_batch, int m, int n, int k,
                                                at::Half *alpha, const at::Half *A, int lda,
                                                at::Half *B, int ldb, at::Half *beta,
                                                at::Half *C, int ldc,
                                                int strideA, int strideB, int strideC)
{
    half * A_t = (half *)(A);
    half * B_t = (half *)(B);
    half * C_t = (half *)(C);
    half * alpha_t = (half *)(alpha);
    half * beta_t = (half *)(beta);
    return cublasHgemmStridedBatched(handle, transa, transb, m, n, k, alpha_t, A_t, lda, strideA, B_t, ldb, strideB, beta_t, C_t, ldc, strideC, num_batch);
}

static inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                                cublasOperation_t transa, cublasOperation_t transb,
                                                int num_batch, int m, int n, int k,
                                                float *alpha, const float *A, int lda,
                                                float *B, int ldb, float *beta,
                                                float *C, int ldc,
                                                int strideA, int strideB, int strideC)
{
    return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, num_batch);
}

static inline cublasStatus_t cublasXgemmBatched(cublasHandle_t handle,
                                                cublasOperation_t transa, cublasOperation_t transb,
                                                int num_batch, int m, int n, int k,
                                                double *alpha, const double *A, int lda,
                                                double *B, int ldb, double *beta,
                                                double *C, int ldc,
                                                int strideA, int strideB, int strideC)
{
    return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, num_batch);
}

static cublasHandle_t global_handle = NULL;
static cudaStream_t * streamArray = NULL;
static cublasHandle_t * global_handle_array = NULL;

at::Tensor batched_sparse_gemm_v2_forward(
    const at::Tensor a,
    const at::Tensor b,
    const at::Tensor a_lengths,
    const at::Tensor b_lengths,
    bool is_a_transposed,
    bool is_b_transposed
) {
    // a: [H, La, k], b: [H, Lb, k]
    const int num_batches = a_lengths.size(0);
    const int num_heads   = a.size(0);

    if (streamArray == NULL) {
        cublasCreate(&global_handle);
        streamArray = (cudaStream_t *)malloc(MAX_STREAM_NUM * sizeof(cudaStream_t *));
        global_handle_array = (cublasHandle_t *)malloc(MAX_STREAM_NUM * sizeof(cublasHandle_t *));
 
        for (int i = 0; i < MAX_STREAM_NUM; i++)
        {
            CHECK_CUDA( cudaStreamCreate(&streamArray[i]) )
            cublasCreate(&(global_handle_array[i]));
            cublasSetStream(global_handle_array[i], streamArray[i]);
        }
    }

    int64_t * a_lengths_ptr = a_lengths.data_ptr<int64_t>();
    int64_t * b_lengths_ptr = b_lengths.data_ptr<int64_t>();

    if (is_b_transposed)
        assert(b.size(1) % num_batches == 0);

    const int out_dim1 = is_a_transposed ? num_batches * a.size(2) : a.size(1);
    const int out_dim2 = is_b_transposed ? b.size(1) / num_batches : b.size(2);
    auto output = at::empty({num_heads, out_dim1, out_dim2}, a.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.type(), "sparse_mm_forward", ([&] {
        scalar_t * dA = a.contiguous().data_ptr<scalar_t>();
        scalar_t * dB = b.contiguous().data_ptr<scalar_t>();
        scalar_t * dC = output.data_ptr<scalar_t>();

        scalar_t alpha_t = 1.0;
        scalar_t beta_t = 0.0;

        auto transa = is_a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        auto transb = is_b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;

        for (int i = 0; i < num_batches; i++)
        {
            const int m = is_a_transposed ? a.size(2) : a_lengths_ptr[i];
            const int k = is_b_transposed ? b.size(2) : b_lengths_ptr[i];
            const int n = is_b_transposed ? b_lengths_ptr[i] : b.size(2);

            cublasXgemmBatched(global_handle_array[i % MAX_STREAM_NUM],
                               transb, transa, num_heads, n, m, k,
                               &alpha_t, dB, is_b_transposed ? k : n,
                               dA, is_a_transposed ? m : k,
                               &beta_t, dC, is_a_transposed ? m : n,
                               b.size(1) * b.size(2), a.size(1) * a.size(2), out_dim1 * out_dim2);
            dA += m * k;
            dB += k * n;
            dC += m * n;
        }
    }));

    return output;
}

at::Tensor batched_sparse_gemm_forward(
    const at::Tensor a,
    const at::Tensor b,
    const at::Tensor a_lengths,
    const at::Tensor b_lengths,
    bool is_a_transposed,
    bool is_b_transposed
) {
    // a: [H, La, k], b: [H, Lb, k]
    const int num_batches = a_lengths.size(0);
    const int num_heads   = a.size(0);

    int64_t a_start_idx = 0;
    int64_t b_start_idx = 0;

    int64_t * a_lengths_ptr = a_lengths.data_ptr<int64_t>();
    int64_t * b_lengths_ptr = b_lengths.data_ptr<int64_t>();

    std::vector<at::Tensor> outputs;
    for (int i = 0; i < num_batches; i++) {
        at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();
        {
            at::cuda::CUDAStreamGuard guard(stream);

            const int64_t la = a_lengths_ptr[i];
            const int64_t lb = b_lengths_ptr[i];
            auto a_i = a.slice(1, a_start_idx, a_start_idx + la);
            auto b_i = b.slice(1, b_start_idx, b_start_idx + lb);

            if (is_a_transposed)
                a_i = a_i.transpose(-1, -2);
            if (is_b_transposed)
                b_i = b_i.transpose(-1, -2);

            auto out_i = at::bmm(a_i, b_i);
            outputs.push_back(out_i);

            a_start_idx += la;
            b_start_idx += lb;
        }
    }

    auto output = at::cat(outputs, 1);

    return output;
}

static inline cudnnDataType_t convert_attype(at::DeprecatedTypeProperties type) {
    const auto& the_type = type;
    at::ScalarType _st = the_type.scalarType();
    assert(_st != at::ScalarType::Half);
    switch (_st) {
        case at::ScalarType::Float:
            return CUDNN_DATA_FLOAT;
        case at::ScalarType::Half:
            return CUDNN_DATA_HALF;
        case at::ScalarType::Double:
            return CUDNN_DATA_DOUBLE;
        default:
            printf("Convert error.");
    }
    return CUDNN_DATA_HALF;
}

static cudaStream_t * attn_cuda_streams = NULL;
static cublasHandle_t * attn_cublas_handles = NULL;
static cudnnHandle_t * attn_cudnn_handles = NULL;
static cudnnTensorDescriptor_t srcTensorDesc;

at::Tensor batched_sparse_attention_forward(
    const at::Tensor a,
    const at::Tensor b,
    const at::Tensor c,
    const at::Tensor batch_lengths,
    const float scale
) {
    // a: [H, L, C], b: [H, B, N, C], c: [H, B, N, C]
    int num_heads = a.size(0);
    int num_instances = a.size(1);
    int batch = b.size(1);
    int k = b.size(3);
    int n = b.size(2);

    if (attn_cuda_streams == NULL) {
        attn_cuda_streams = (cudaStream_t *)malloc(MAX_STREAM_NUM * sizeof(cudaStream_t *));
        attn_cublas_handles = (cublasHandle_t *)malloc(MAX_STREAM_NUM * sizeof(cublasHandle_t *));
        attn_cudnn_handles = (cudnnHandle_t *)malloc(MAX_STREAM_NUM * sizeof(cudnnHandle_t *));

        for (int i = 0; i < MAX_STREAM_NUM; i++)
        {
            cudaStreamCreate(&attn_cuda_streams[i]);
            cublasCreate(&attn_cublas_handles[i]);
            cublasSetStream(attn_cublas_handles[i], attn_cuda_streams[i]);
            cudnnCreate(&attn_cudnn_handles[i]);
            cudnnSetStream(attn_cudnn_handles[i], attn_cuda_streams[i]);
        }
        cudnnCreateTensorDescriptor(&srcTensorDesc);
    }

    auto attn = at::empty({num_heads, num_instances, n}, a.options());
    auto output = at::empty({num_heads, num_instances, k}, a.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "batched_sparse_attention_forward", ([&] {
        scalar_t * a_ptr = a.contiguous().data_ptr<scalar_t>();
        scalar_t * b_ptr = b.contiguous().data_ptr<scalar_t>();
        scalar_t * c_ptr = c.contiguous().data_ptr<scalar_t>();
        scalar_t * attn_ptr = attn.data_ptr<scalar_t>();
        scalar_t * output_ptr = output.data_ptr<scalar_t>();

        int64_t * batch_lengths_ptr = batch_lengths.data_ptr<int64_t>();

        scalar_t alpha_t = 1.0;
        scalar_t beta_t = 0.0;
        scalar_t scale_t = static_cast<scalar_t>(scale);

        cudnnDataType_t cudnn_type = convert_attype(a.type());

        for (int i = 0; i < batch; i++)
        {
            int m = batch_lengths_ptr[i];

            // A * B
            cublasXgemmBatched(attn_cublas_handles[i % MAX_STREAM_NUM],
                               CUBLAS_OP_T, CUBLAS_OP_N,
                               num_heads, n, m, k,
                               &scale_t, b_ptr, k,
                               a_ptr, k, &beta_t, attn_ptr, n,
                               batch * k * n, num_instances * k, num_instances * n);

            // Softmax(attn) * scale
            cudnnSetTensor4dDescriptorEx(srcTensorDesc, cudnn_type, num_heads,
                                         n, m, 1, num_instances * n, 1, n, 1);
            cudnnSoftmaxForward(attn_cudnn_handles[i % MAX_STREAM_NUM],
                                CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                &alpha_t, srcTensorDesc, attn_ptr,
                                &beta_t, srcTensorDesc, attn_ptr);

            // attn * C
            cublasXgemmBatched(attn_cublas_handles[i % MAX_STREAM_NUM],
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               num_heads, k, m, n,
                               &alpha_t, c_ptr, k,
                               attn_ptr, n, &beta_t, output_ptr, k,
                               batch * k * n, num_instances * n, num_instances * k);

            a_ptr += m * k;
            b_ptr += k * n;
            attn_ptr += m * n;

            c_ptr += k * n;
            output_ptr += m * k;
        }
    }));

    return output;
}
