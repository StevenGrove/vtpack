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
__global__ void bilinear_attention_pointwise_forward_kernel(
        const PackedTensorAccessor64<scalar_t, 4> input,
        const PackedTensorAccessor64<scalar_t, 4> grid,
        const PackedTensorAccessor64<scalar_t, 3> weight,
        PackedTensorAccessor64<scalar_t, 3> output
    ){

    const int num_batches   = input.size(0);
    const int height        = input.size(1);
    const int width         = input.size(2);
    const int num_channels  = input.size(3); 
    const int num_grids     = grid.size(1);
    const int num_points    = grid.size(2);
    const int num_instances = num_batches * num_grids;

    extern __shared__ int tile[];
    scalar_t * points  = (scalar_t *)tile;
    scalar_t * weights = (scalar_t *)tile + blockDim.x * num_points * 2;

    CUDA_KERNEL_LOOP(index, num_instances){
        const int batch_idx = index / num_grids;
        const int grid_idx  = index % num_grids;

        for (int point_idx = 0; point_idx < num_points; ++point_idx) {
            points[(point_idx * 2) * blockDim.x + threadIdx.x]     = grid[batch_idx][grid_idx][point_idx][0];
            points[(point_idx * 2 + 1) * blockDim.x + threadIdx.x] = grid[batch_idx][grid_idx][point_idx][1];
            weights[point_idx * blockDim.x + threadIdx.x]          = weight[batch_idx][grid_idx][point_idx];
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

            output[batch_idx][grid_idx][channel_idx] = val;
        }
    }
}

template <typename scalar_t, typename accscalar_t>
__global__ void bilinear_attention_pointwise_backward_kernel(
        const PackedTensorAccessor64<scalar_t, 3> grad_output,
        const PackedTensorAccessor64<scalar_t, 4> input,
        const PackedTensorAccessor64<scalar_t, 4> grid,
        const PackedTensorAccessor64<scalar_t, 3> weight,
        PackedTensorAccessor64<scalar_t, 4> grad_input,
        PackedTensorAccessor64<scalar_t, 4> grad_grid,
        PackedTensorAccessor64<scalar_t, 3> grad_weight,
        const int grad_input_memory_span
    ){

    const int num_batches   = input.size(0);
    const int height        = input.size(1);
    const int width         = input.size(2);
    const int num_channels  = input.size(3); 
    const int num_grids     = grid.size(1);
    const int num_points    = grid.size(2);
    const int num_instances = num_batches * num_grids;

    CUDA_KERNEL_LOOP(index, num_instances){
        const int batch_idx   = index / num_grids;
        const int grid_idx    = index % num_grids;

        for (int point_idx = 0; point_idx < num_points; ++point_idx) {

            const scalar_t w  = weight[batch_idx][grid_idx][point_idx];
            const scalar_t x = grid[batch_idx][grid_idx][point_idx][0];
            const scalar_t y = grid[batch_idx][grid_idx][point_idx][1];

            const int in_x = floor(x);
            const int in_y = floor(y);
            const accscalar_t t_x = x - in_x;
            const accscalar_t t_y = y - in_y;

            const accscalar_t w_lt = (1 - t_x) * (1 - t_y);
            const accscalar_t w_rt = t_x * (1 - t_y);
            const accscalar_t w_lb = (1 - t_x) * t_y;
            const accscalar_t w_rb = t_x * t_y;

            accscalar_t gix = static_cast<accscalar_t>(0);
            accscalar_t giy = static_cast<accscalar_t>(0);
            accscalar_t giw = static_cast<accscalar_t>(0);

            for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx) {
                const scalar_t grad_output_v = grad_output[batch_idx][grid_idx][channel_idx];

                add_value_bounded<scalar_t>(grad_input, batch_idx, channel_idx, height, width,
                    in_y, in_x, static_cast<scalar_t>(grad_output_v * w * w_lt),
                    grad_input_memory_span);
                add_value_bounded<scalar_t>(grad_input, batch_idx, channel_idx, height, width,
                    in_y, in_x + 1, static_cast<scalar_t>(grad_output_v * w * w_rt),
                    grad_input_memory_span);
                add_value_bounded<scalar_t>(grad_input, batch_idx, channel_idx, height, width,
                    in_y + 1, in_x, static_cast<scalar_t>(grad_output_v * w * w_lb),
                    grad_input_memory_span);
                add_value_bounded<scalar_t>(grad_input, batch_idx, channel_idx, height, width,
                    in_y + 1, in_x + 1, static_cast<scalar_t>(grad_output_v * w * w_rb),
                    grad_input_memory_span);

                const scalar_t val_lt = upsample_get_value_bounded<scalar_t>(
                    input, batch_idx, channel_idx, height, width, in_y, in_x);
                const scalar_t val_rt = upsample_get_value_bounded<scalar_t>(
                    input, batch_idx, channel_idx, height, width, in_y, in_x + 1);
                const scalar_t val_lb = upsample_get_value_bounded<scalar_t>(
                    input, batch_idx, channel_idx, height, width, in_y + 1, in_x);
                const scalar_t val_rb = upsample_get_value_bounded<scalar_t>(
                    input, batch_idx, channel_idx, height, width, in_y + 1, in_x + 1);

                const scalar_t g_lt = val_lt * grad_output_v;
                gix -= g_lt * (1 - t_y);
                giy -= g_lt * (1 - t_x);

                const scalar_t g_rt = val_rt * grad_output_v;
                gix += g_rt * (1 - t_y);
                giy -= g_rt * t_x;

                const scalar_t g_lb = val_lb * grad_output_v;
                gix -= g_lb * t_y;
                giy += g_lb * (1 - t_x);

                const scalar_t g_rb = val_rb * grad_output_v;
                gix += g_rb * t_y;
                giy += g_rb * t_x;

                const scalar_t output_val = static_cast<scalar_t>(
                    val_lt * w_lt + val_rt * w_rt + val_lb * w_lb + val_rb * w_rb);
                giw += output_val * grad_output_v;
            }

            grad_grid[batch_idx][grid_idx][point_idx][0] = static_cast<scalar_t>(gix * w);
            grad_grid[batch_idx][grid_idx][point_idx][1] = static_cast<scalar_t>(giy * w);
            grad_weight[batch_idx][grid_idx][point_idx]  = static_cast<scalar_t>(giw);
        }

    }
}

template <typename scalar_t, typename accscalar_t>
__global__ void bilinear_attention_channelwise_forward_kernel(
        const PackedTensorAccessor64<scalar_t, 4> input,
        const PackedTensorAccessor64<scalar_t, 4> grid,
        const PackedTensorAccessor64<scalar_t, 3> weight,
        PackedTensorAccessor64<scalar_t, 3> output
    ){

    const int num_batches   = input.size(0);
    const int height        = input.size(1);
    const int width         = input.size(2);
    const int num_channels  = input.size(3); 
    const int num_grids     = grid.size(1);
    const int num_points    = grid.size(2);
    const int num_instances = num_batches * num_grids;

    CUDA_KERNEL_LOOP(index, num_instances){
        const int batch_idx = index / num_grids;
        const int grid_idx  = index % num_grids;

        for (int point_idx = 0; point_idx < num_points; ++point_idx) {
            const scalar_t x = grid[batch_idx][grid_idx][point_idx][0];
            const scalar_t y = grid[batch_idx][grid_idx][point_idx][1];

            scalar_t output_val = static_cast<scalar_t>(0);
            for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx) {
                const scalar_t input_val = bilinear_interp2d<scalar_t, accscalar_t>(
                    input, batch_idx, channel_idx, y, x);
                output_val += input_val * weight[batch_idx][grid_idx][channel_idx];
            }
            output[batch_idx][grid_idx][point_idx] = output_val;
        }
    }
}

template <typename scalar_t, typename accscalar_t>
__global__ void bilinear_attention_channelwise_backward_kernel(
        const PackedTensorAccessor64<scalar_t, 3> grad_output,
        const PackedTensorAccessor64<scalar_t, 4> input,
        const PackedTensorAccessor64<scalar_t, 4> grid,
        const PackedTensorAccessor64<scalar_t, 3> weight,
        PackedTensorAccessor64<scalar_t, 4> grad_input,
        PackedTensorAccessor64<scalar_t, 4> grad_grid,
        PackedTensorAccessor64<scalar_t, 3> grad_weight,
        const int grad_input_memory_span
    ){

    const int num_batches   = input.size(0);
    const int height        = input.size(1);
    const int width         = input.size(2);
    const int num_channels  = input.size(3); 
    const int num_grids     = grid.size(1);
    const int num_points    = grid.size(2);
    const int num_instances = num_batches * num_grids;

    CUDA_KERNEL_LOOP(index, num_instances){
        const int batch_idx   = index / num_grids;
        const int grid_idx    = index % num_grids;

        for (int point_idx = 0; point_idx < num_points; ++point_idx) {
            const scalar_t grad_output_v = grad_output[batch_idx][grid_idx][point_idx];

            const scalar_t x = grid[batch_idx][grid_idx][point_idx][0];
            const scalar_t y = grid[batch_idx][grid_idx][point_idx][1];

            const int in_x = floor(x);
            const int in_y = floor(y);
            const accscalar_t t_x = x - in_x;
            const accscalar_t t_y = y - in_y;

            const accscalar_t w_lt = (1 - t_x) * (1 - t_y);
            const accscalar_t w_rt = t_x * (1 - t_y);
            const accscalar_t w_lb = (1 - t_x) * t_y;
            const accscalar_t w_rb = t_x * t_y;

            accscalar_t gix = static_cast<accscalar_t>(0);
            accscalar_t giy = static_cast<accscalar_t>(0);

            for (int channel_idx = 0; channel_idx < num_channels; ++channel_idx) {
                const scalar_t w  = weight[batch_idx][grid_idx][channel_idx];

                add_value_bounded<scalar_t>(grad_input, batch_idx, channel_idx, height, width,
                    in_y, in_x, static_cast<scalar_t>(grad_output_v * w * w_lt),
                    grad_input_memory_span);
                add_value_bounded<scalar_t>(grad_input, batch_idx, channel_idx, height, width,
                    in_y, in_x + 1, static_cast<scalar_t>(grad_output_v * w * w_rt),
                    grad_input_memory_span);
                add_value_bounded<scalar_t>(grad_input, batch_idx, channel_idx, height, width,
                    in_y + 1, in_x, static_cast<scalar_t>(grad_output_v * w * w_lb),
                    grad_input_memory_span);
                add_value_bounded<scalar_t>(grad_input, batch_idx, channel_idx, height, width,
                    in_y + 1, in_x + 1, static_cast<scalar_t>(grad_output_v * w * w_rb),
                    grad_input_memory_span);

                const scalar_t val_lt = upsample_get_value_bounded<scalar_t>(
                    input, batch_idx, channel_idx, height, width, in_y, in_x);
                const scalar_t val_rt = upsample_get_value_bounded<scalar_t>(
                    input, batch_idx, channel_idx, height, width, in_y, in_x + 1);
                const scalar_t val_lb = upsample_get_value_bounded<scalar_t>(
                    input, batch_idx, channel_idx, height, width, in_y + 1, in_x);
                const scalar_t val_rb = upsample_get_value_bounded<scalar_t>(
                    input, batch_idx, channel_idx, height, width, in_y + 1, in_x + 1);

                const scalar_t g_lt = val_lt * grad_output_v * w;
                gix -= g_lt * (1 - t_y);
                giy -= g_lt * (1 - t_x);

                const scalar_t g_rt = val_rt * grad_output_v * w;
                gix += g_rt * (1 - t_y);
                giy -= g_rt * t_x;

                const scalar_t g_lb = val_lb * grad_output_v * w;
                gix -= g_lb * t_y;
                giy += g_lb * (1 - t_x);

                const scalar_t g_rb = val_rb * grad_output_v * w;
                gix += g_rb * t_y;
                giy += g_rb * t_x;

                const scalar_t input_val = static_cast<scalar_t>(
                    val_lt * w_lt + val_rt * w_rt + val_lb * w_lb + val_rb * w_rb);

                const int offset = (batch_idx * num_grids + grid_idx) * num_channels + channel_idx;
                at::native::fastAtomicAdd(grad_weight.data(), offset, num_batches * num_grids * num_channels,
                    input_val * grad_output_v, true);
            }

            grad_grid[batch_idx][grid_idx][point_idx][0] = static_cast<scalar_t>(gix);
            grad_grid[batch_idx][grid_idx][point_idx][1] = static_cast<scalar_t>(giy);
        }

    }
}

Tensor bilinear_attention_pointwise_forward(
        const Tensor & input,
        const Tensor & grid,
        const Tensor & weight
    ){

    const int num_batches  = input.size(0);
    const int num_channels = input.size(3);
    const int num_grids    = grid.size(1);
    const int num_points   = grid.size(2);

    auto output = at::empty({num_batches, num_grids, num_channels}, input.options());

    // launch kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int num_instances = num_batches * num_grids;
    const int max_threads = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "bilinear_attention_pointwise_forward", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        
        bilinear_attention_pointwise_forward_kernel<scalar_t, accscalar_t>
            <<<at::cuda::ATenCeilDiv(num_instances, max_threads),
               max_threads,
               max_threads * num_points * 3 * sizeof(scalar_t),
               stream>>>(
            input.packed_accessor64<scalar_t, 4>(),
            grid.packed_accessor64<scalar_t, 4>(),
            weight.packed_accessor64<scalar_t, 3>(),
            output.packed_accessor64<scalar_t, 3>());

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return output;
}

std::tuple<Tensor, Tensor, Tensor>
bilinear_attention_pointwise_backward(
        const Tensor & grad_output,
        const Tensor & input,
        const Tensor & grid,
        const Tensor & weight
    ){

    const int num_batches  = input.size(0);
    const int height       = input.size(1);
    const int width        = input.size(2);
    const int num_channels = input.size(3);
    const int num_grids    = grid.size(1);
    const int num_points   = grid.size(2);

    auto grad_input  = at::zeros({num_batches, height, width, num_channels}, grad_output.options());
    auto grad_grid   = at::empty({num_batches, num_grids, num_points, 2}, grad_output.options());
    auto grad_weight = at::empty({num_batches, num_grids, num_points}, grad_output.options());

    // launch kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int num_instances = num_batches * num_grids;
    const int max_threads = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "bilinear_attention_pointwise_backward", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        
        bilinear_attention_pointwise_backward_kernel<scalar_t, accscalar_t>
            <<<at::cuda::ATenCeilDiv(num_instances, max_threads),
               max_threads,
               0,
               stream>>>(
            grad_output.packed_accessor64<scalar_t, 3>(),
            input.packed_accessor64<scalar_t, 4>(),
            grid.packed_accessor64<scalar_t, 4>(),
            weight.packed_accessor64<scalar_t, 3>(),
            grad_input.packed_accessor64<scalar_t, 4>(),
            grad_grid.packed_accessor64<scalar_t, 4>(),
            grad_weight.packed_accessor64<scalar_t, 3>(),
            static_cast<int>(grad_input.numel()));

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return std::make_tuple(grad_input, grad_grid, grad_weight);
}

Tensor bilinear_attention_channelwise_forward(
        const Tensor & input,
        const Tensor & grid,
        const Tensor & weight
    ){

    const int num_batches  = input.size(0);
    const int num_channels = input.size(3);
    const int num_grids    = grid.size(1);
    const int num_points   = grid.size(2);

    auto output = at::empty({num_batches, num_grids, num_points}, input.options());

    // launch kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int num_instances = num_batches * num_grids;
    const int max_threads = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "bilinear_attention_channelwise_forward", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        
        bilinear_attention_channelwise_forward_kernel<scalar_t, accscalar_t>
            <<<at::cuda::ATenCeilDiv(num_instances, max_threads),
               max_threads,
               0,
               stream>>>(
            input.packed_accessor64<scalar_t, 4>(),
            grid.packed_accessor64<scalar_t, 4>(),
            weight.packed_accessor64<scalar_t, 3>(),
            output.packed_accessor64<scalar_t, 3>());

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return output;
}

std::tuple<Tensor, Tensor, Tensor>
bilinear_attention_channelwise_backward(
        const Tensor & grad_output,
        const Tensor & input,
        const Tensor & grid,
        const Tensor & weight
    ){

    const int num_batches  = input.size(0);
    const int height       = input.size(1);
    const int width        = input.size(2);
    const int num_channels = input.size(3);
    const int num_grids    = grid.size(1);
    const int num_points   = grid.size(2);

    auto grad_input  = at::zeros({num_batches, height, width, num_channels}, grad_output.options());
    auto grad_grid   = at::empty({num_batches, num_grids, num_points, 2}, grad_output.options());
    auto grad_weight = at::zeros({num_batches, num_grids, num_channels}, grad_output.options());

    // launch kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int num_instances = num_batches * num_grids;
    const int max_threads = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "bilinear_attention_channelwise_backward", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        
        bilinear_attention_channelwise_backward_kernel<scalar_t, accscalar_t>
            <<<at::cuda::ATenCeilDiv(num_instances, max_threads),
               max_threads,
               0,
               stream>>>(
            grad_output.packed_accessor64<scalar_t, 3>(),
            input.packed_accessor64<scalar_t, 4>(),
            grid.packed_accessor64<scalar_t, 4>(),
            weight.packed_accessor64<scalar_t, 3>(),
            grad_input.packed_accessor64<scalar_t, 4>(),
            grad_grid.packed_accessor64<scalar_t, 4>(),
            grad_weight.packed_accessor64<scalar_t, 3>(),
            static_cast<int>(grad_input.numel()));

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return std::make_tuple(grad_input, grad_grid, grad_weight);
}
