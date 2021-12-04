#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

using namespace at;

template <typename scalar_t>
__device__ __forceinline__ static scalar_t upsample_get_value_bounded(
    const PackedTensorAccessor64<scalar_t, 4>& data,
    int batch,
    int channel,
    int height,
    int width,
    int y,
    int x) {

    if (y >= 0 && y < height && x >= 0 && x < width) {
        return data[batch][y][x][channel];
    }
    return static_cast<scalar_t>(0);
}

template <typename scalar_t>
__device__ __forceinline__ static void add_value_bounded(
    PackedTensorAccessor64<scalar_t, 4>& data,
    int batch,
    int channel,
    int height,
    int width,
    int y,
    int x,
    const scalar_t delta,
    const int memory_span) {

    const int num_channels = data.size(3);
    const int offset = ((batch * height + y) * width + x) * num_channels + channel;
    if (y >= 0 && y < height && x >= 0 && x < width) {
        at::native::fastAtomicAdd(data.data(), offset, memory_span, delta, true);
    }
}

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ static scalar_t bilinear_interp2d(
        const PackedTensorAccessor64<scalar_t, 4>& input,
        const int batch_idx,
        const int channel_idx,
        const scalar_t y,
        const scalar_t x
    ){

    const int height = input.size(1);
    const int width  = input.size(2);

    const int in_x = floor(x);
    const int in_y = floor(y);
    const accscalar_t t_x = x - in_x;
    const accscalar_t t_y = y - in_y;

    const accscalar_t w_lt = (1 - t_x) * (1 - t_y);
    const accscalar_t w_rt = t_x * (1 - t_y);
    const accscalar_t w_lb = (1 - t_x) * t_y;
    const accscalar_t w_rb = t_x * t_y;

    const accscalar_t res =
        w_lt * upsample_get_value_bounded<scalar_t>(
            input, batch_idx, channel_idx, height, width, in_y, in_x) +
        w_rt * upsample_get_value_bounded<scalar_t>(
            input, batch_idx, channel_idx, height, width, in_y, in_x + 1) +
        w_lb * upsample_get_value_bounded<scalar_t>(
            input, batch_idx, channel_idx, height, width, in_y + 1, in_x) +
        w_rb * upsample_get_value_bounded<scalar_t>(
            input, batch_idx, channel_idx, height, width, in_y + 1, in_x + 1);

    return static_cast<scalar_t>(res);
}

template <typename scalar_t, typename accscalar_t>
__global__ void sparse_bilinear_attention_pointwise_forward_kernel(
        const PackedTensorAccessor64<scalar_t, 4> input,
        const PackedTensorAccessor64<scalar_t, 3> grid,
        const PackedTensorAccessor64<scalar_t, 2> weight,
        const PackedTensorAccessor64<int64_t, 1> batches,
        PackedTensorAccessor64<scalar_t, 2> output
    ){

    const int height        = input.size(1);
    const int width         = input.size(2);
    const int num_channels  = input.size(3); 
    const int num_grids     = grid.size(0);
    const int num_points    = grid.size(1);

    extern __shared__ int tile[];
    scalar_t * points  = (scalar_t *)tile;
    scalar_t * weights = (scalar_t *)tile + blockDim.x * num_points * 2;

    CUDA_KERNEL_LOOP(index, num_grids){
        const int grid_idx  = index;
        const int batch_idx = batches[grid_idx];

        for (int point_idx = 0; point_idx < num_points; ++point_idx) {
            points[(point_idx * 2) * blockDim.x + threadIdx.x]     = grid[grid_idx][point_idx][0];
            points[(point_idx * 2 + 1) * blockDim.x + threadIdx.x] = grid[grid_idx][point_idx][1];
            weights[point_idx * blockDim.x + threadIdx.x]          = weight[grid_idx][point_idx];
        }

        for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx) {

            scalar_t val = static_cast<scalar_t>(0);
            for (int point_idx = 0; point_idx < num_points; ++point_idx) {
                const scalar_t x = points[(point_idx * 2) * blockDim.x + threadIdx.x];
                const scalar_t y = points[(point_idx * 2 + 1) * blockDim.x + threadIdx.x];
                const scalar_t w = weights[point_idx * blockDim.x + threadIdx.x];
                val += bilinear_interp2d<scalar_t, accscalar_t>(
                    input, batch_idx, channel_idx, y, x) * w;
            }

            output[grid_idx][channel_idx] = val;
        }
    }
}

Tensor sparse_bilinear_attention_pointwise_forward(
        const Tensor & input,
        const Tensor & grid,
        const Tensor & weight,
        const Tensor & batches
    ){

    const int num_channels = input.size(3);
    const int num_grids    = grid.size(0);
    const int num_points   = grid.size(1);

    auto output = at::empty({num_grids, num_channels}, input.options());

    // launch kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int max_threads = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
            "sparse_bilinear_attention_pointwise_forward", [&] {

        using accscalar_t = at::acc_type<scalar_t, true>;
        
        sparse_bilinear_attention_pointwise_forward_kernel<scalar_t, accscalar_t>
            <<<at::cuda::ATenCeilDiv(num_grids, max_threads),
               max_threads,
               max_threads * num_points * 3 * sizeof(scalar_t),
               stream>>>(
            input.packed_accessor64<scalar_t, 4>(),
            grid.packed_accessor64<scalar_t, 3>(),
            weight.packed_accessor64<scalar_t, 2>(),
            batches.packed_accessor64<int64_t, 1>(),
            output.packed_accessor64<scalar_t, 2>());

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return output;
}
