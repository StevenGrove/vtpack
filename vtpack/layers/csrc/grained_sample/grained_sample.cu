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
__device__ __forceinline__ static void add_value(
    PackedTensorAccessor64<scalar_t, 3>& data,
    int batch,
    int node,
    int channel,
    const scalar_t delta) {
    
    const int num_channels = data.size(3);
    const int offset = batch * (data.size(1) + node) * data.size(0) + channel;
    atomicAdd(data.data() + offset, delta);
}

template <typename scalar_t, typename accscalar_t>
__global__ void grained_sample_compress_kernel(
        const PackedTensorAccessor64<scalar_t, 3> input,
        const PackedTensorAccessor64<int64_t, 2> patches,
        PackedTensorAccessor64<scalar_t, 2> output,
        const int height,
        const int width,
        const int num_extra_tokens
    ){

    const int num_batches  = input.size(0);
    const int num_channels = input.size(2);
    const int num_patches  = patches.size(0);

    CUDA_KERNEL_LOOP(index, num_patches * num_channels){
        const int inst_idx    = index / num_channels;
        const int channel_idx = index % num_channels;
        const int batch_idx   = patches[inst_idx][0];
        const int y_s         = patches[inst_idx][1];
        const int x_s         = patches[inst_idx][2];
        const int patch_size  = patches[inst_idx][3];

        if (y_s < 0){
            const int extra_token_idx = -y_s - 1;
            output[inst_idx][channel_idx] = input[batch_idx][extra_token_idx][channel_idx];
        }
        else{
            accscalar_t out_data = static_cast<accscalar_t>(0);
            for (int dy = 0; dy < patch_size; dy++){
                for (int dx = 0; dx < patch_size; dx++){
                    out_data += input[batch_idx][(y_s + dy) * width + x_s + dx + num_extra_tokens][channel_idx];
                }
            }

            out_data = out_data / (patch_size * patch_size);
            output[inst_idx][channel_idx] = static_cast<scalar_t>(out_data);
        } 
    }
}

template <typename scalar_t>
__global__ void grained_sample_decompress_kernel(
        const PackedTensorAccessor64<scalar_t, 2> input,
        const PackedTensorAccessor64<int64_t, 2> patches,
        PackedTensorAccessor64<scalar_t, 3> output,
        const int height,
        const int width,
        const int num_extra_tokens
    ){

    const int num_patches  = input.size(0);
    const int num_channels = input.size(1);

    CUDA_KERNEL_LOOP(index, num_patches * num_channels){
        const int patch_idx   = index / num_channels;
        const int channel_idx = index % num_channels;
        const int batch_idx   = patches[patch_idx][0];
        const int y_s         = patches[patch_idx][1];
        const int x_s         = patches[patch_idx][2];
        const int patch_size  = patches[patch_idx][3];

        if (y_s < 0){
            const int extra_token_idx = -y_s - 1;
            output[batch_idx][extra_token_idx][channel_idx] = input[patch_idx][channel_idx];
        }
        else{
            const scalar_t in_data = input[patch_idx][channel_idx];
            for (int dy = 0; dy < patch_size; dy++){
                for (int dx = 0; dx < patch_size; dx++){
                    output[batch_idx][(y_s + dy) * width + x_s + dx + num_extra_tokens][channel_idx] = in_data;
                }
            }
        }
    }
}

template <typename scalar_t>
__global__ void grained_sample_decompress_kernel(
        const PackedTensorAccessor64<scalar_t, 2> input,
        const PackedTensorAccessor64<int64_t, 2> patches,
        const PackedTensorAccessor64<scalar_t, 3> reference,
        PackedTensorAccessor64<scalar_t, 3> output,
        const int height,
        const int width,
        const int num_extra_tokens
    ){

    const int num_patches  = input.size(0);
    const int num_channels = input.size(1);

    CUDA_KERNEL_LOOP(index, num_patches * num_channels){
        const int patch_idx   = index / num_channels;
        const int channel_idx = index % num_channels;
        const int batch_idx   = patches[patch_idx][0];
        const int y_s         = patches[patch_idx][1];
        const int x_s         = patches[patch_idx][2];
        const int patch_size  = patches[patch_idx][3];

        const scalar_t in_data = input[patch_idx][channel_idx];
        if (y_s < 0){
            const int extra_token_idx = -y_s - 1;
            const scalar_t ref = reference[batch_idx][extra_token_idx][channel_idx];
            output[batch_idx][extra_token_idx][channel_idx] = in_data + ref;
        }
        else{
            for (int dy = 0; dy < patch_size; dy++){
                for (int dx = 0; dx < patch_size; dx++){
                    const scalar_t ref = reference[batch_idx][(y_s + dy) * width + x_s + dx + num_extra_tokens][channel_idx];
                    output[batch_idx][(y_s + dy) * width + x_s + dx + num_extra_tokens][channel_idx] = in_data + ref;
                }
            }
        }
    }
}

__global__ void grained_sample_index_to_patch_kernel(
        const PackedTensorAccessor64<int64_t, 3> indices,
        const PackedTensorAccessor64<int64_t, 3> patch_sizes,
        const PackedTensorAccessor64<int64_t, 1> output_lengths,
        PackedTensorAccessor64<int64_t, 2> patches,
        const int region_size,
        const int num_extra_tokens
    ){

    const int num_batches  = indices.size(0);
    const int height       = indices.size(1);
    const int width        = indices.size(2);
    const int img_size     = height * width;
    const int batch_size   = 1 + img_size;
    const int num_parallel = num_batches * batch_size;

    CUDA_KERNEL_LOOP(index, num_parallel){
        const int batch_idx      = index / batch_size;
        const int index_in_batch = index % batch_size;

        // extra token
        if (index_in_batch == 0){
            int patch_idx = output_lengths[index] - num_extra_tokens;
            for (int idx = 0; idx < num_extra_tokens; idx++){
                patches[patch_idx][0] = batch_idx;
                patches[patch_idx][1] = -idx - 1;
                patches[patch_idx][3] = 1;
                patch_idx++;
            }
            continue;
        }

        const int y_idx      = (index_in_batch - 1) / width;
        const int x_idx      = (index_in_batch - 1) % width;
        const int patch_y    = y_idx * region_size;
        const int patch_x    = x_idx * region_size;
        const int patch_size = patch_sizes[batch_idx][y_idx][x_idx];
        const int step       = region_size / patch_size;

        int patch_idx  = output_lengths[index] - patch_size * patch_size;
        for (int dy = 0; dy < patch_size; dy++){
            for (int dx = 0; dx < patch_size; dx++){
                patches[patch_idx][0] = batch_idx;
                patches[patch_idx][1] = patch_y + dy * step;
                patches[patch_idx][2] = patch_x + dx * step;
                patches[patch_idx][3] = step;
                patch_idx++;
            }
        }
    }
}

Tensor grained_sample_compress(
        const Tensor & input,
        const Tensor & patches,
        const int height,
        const int width,
        const int num_extra_tokens
    ){

    // dims of input: [B, N, C]
    const int num_batches  = input.size(0);
    const int num_channels = input.size(2);

    // dims of each patch: [batch_idx, x0, y0, size]
    const int num_patches  = patches.size(0);

    auto output = at::empty({num_patches, num_channels}, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int max_threads = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grained_sample_compress", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        grained_sample_compress_kernel<scalar_t, accscalar_t>
            <<<at::cuda::ATenCeilDiv(num_patches * num_channels, max_threads),
               max_threads,
               0,
               stream>>>(
            input.packed_accessor64<scalar_t, 3>(),
            patches.packed_accessor64<int64_t, 2>(),
            output.packed_accessor64<scalar_t, 2>(),
            height,
            width,
            num_extra_tokens
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return output;
}

Tensor grained_sample_decompress(
        const Tensor & input,
        const Tensor & patches,
        const Tensor & reference,
        const int height,
        const int width,
        const int num_extra_tokens,
        const bool is_add_ref
    ){

    const int num_batches  = reference.size(0);
    const int num_insts    = reference.size(1);
    const int num_channels = reference.size(2);

    // dims of each patch: [batch_idx, x0, y0, size]
    const int num_patches  = patches.size(0);

    auto output = at::empty({num_batches, num_insts, num_channels}, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int max_threads = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "grained_sample_decompress", [&] {

        if (is_add_ref) {
            grained_sample_decompress_kernel<scalar_t>
                <<<at::cuda::ATenCeilDiv(num_patches * num_channels, max_threads),
                   max_threads,
                   0,
                   stream>>>(
                input.packed_accessor64<scalar_t, 2>(),
                patches.packed_accessor64<int64_t, 2>(),
                reference.packed_accessor64<scalar_t, 3>(),
                output.packed_accessor64<scalar_t, 3>(),
                height,
                width,
                num_extra_tokens
            );
        }
        else {
            grained_sample_decompress_kernel<scalar_t>
                <<<at::cuda::ATenCeilDiv(num_patches * num_channels, max_threads),
                   max_threads,
                   0,
                   stream>>>(
                input.packed_accessor64<scalar_t, 2>(),
                patches.packed_accessor64<int64_t, 2>(),
                output.packed_accessor64<scalar_t, 3>(),
                height,
                width,
                num_extra_tokens
            );
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return output;
}

std::tuple<Tensor, Tensor>
grained_sample_index_to_patch(
        const Tensor & indices,
        const Tensor & split_sizes,
        const int region_size,
        const int num_extra_tokens
    ){
    const int num_batches  = indices.size(0);
    const int height       = indices.size(1);
    const int width        = indices.size(2);

    auto patch_sizes = split_sizes.index({indices});
    auto extra_token_sizes = at::full({num_batches, 1}, num_extra_tokens, patch_sizes.options());
    auto patch_lengths = at::cat({extra_token_sizes, (patch_sizes * patch_sizes).view({num_batches, -1})}, -1);
    auto output_lengths = patch_lengths.view(-1).cumsum(0);

    auto batch_lengths  = patch_lengths.sum(1).to(at::kCPU);
    const int64_t num_patches = batch_lengths.sum().data_ptr<int64_t>()[0];

    auto patches = at::empty({num_patches, 4}, indices.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int max_threads = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);

    const int num_parallel = num_batches * (1 + height * width);

    grained_sample_index_to_patch_kernel
        <<<at::cuda::ATenCeilDiv(num_parallel, max_threads),
           max_threads,
           0,
           stream>>>(
        indices.packed_accessor64<int64_t, 3>(),
        patch_sizes.packed_accessor64<int64_t, 3>(),
        output_lengths.packed_accessor64<int64_t, 1>(),
        patches.packed_accessor64<int64_t, 2>(),
        region_size,
        num_extra_tokens
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(patches, batch_lengths);
}

__global__ void grained_sample_index_to_patch_dense_kernel(
        const PackedTensorAccessor64<int64_t, 3> indices,
        const PackedTensorAccessor64<int64_t, 1> split_sizes,
        PackedTensorAccessor64<int64_t, 2> patches,
        const int region_size,
        const int num_extra_tokens,
        const int num_patches_per_batch
    ){

    const int num_batches  = indices.size(0);
    const int height       = indices.size(1);
    const int width        = indices.size(2);
    const int num_splits   = split_sizes.size(0);
    const int img_size     = height * width;
    const int num_parallel = num_batches * num_patches_per_batch;

    CUDA_KERNEL_LOOP(index, num_parallel){
        const int batch_idx      = index / num_patches_per_batch;
        const int index_in_batch = index % num_patches_per_batch;

        if (index_in_batch < num_extra_tokens){
            patches[index][0] = batch_idx;
            patches[index][1] = -index_in_batch - 1;
            patches[index][3] = 1;
            continue;
        }

        int spatial_idx = index_in_batch - num_extra_tokens;
        for (int i = 0; i < num_splits; i++){
            const int split_size = split_sizes[i];
            const int spatial_len = img_size * split_size * split_size;
            if (spatial_idx < spatial_len){
                const int step = region_size / split_size;
                const int y_s  = spatial_idx / (width * split_size);
                const int x_s  = spatial_idx % (width * split_size);
                patches[index][0] = batch_idx;
                patches[index][1] = y_s * step;
                patches[index][2] = x_s * step;
                patches[index][3] = step;
                break;
            }
            spatial_idx -= spatial_len;
        }
    }
}

std::tuple<Tensor, Tensor>
grained_sample_index_to_patch_dense(
        const Tensor & indices,
        const Tensor & split_sizes,
        const int region_size,
        const int num_extra_tokens
    ){
    const int num_batches  = indices.size(0);
    const int height       = indices.size(1);
    const int width        = indices.size(2);
    const int num_splits   = split_sizes.size(0);

    auto split_sizes_cpu = split_sizes.to(at::kCPU);
    auto split_lengths = (split_sizes_cpu * split_sizes_cpu) * height * width;

    const int num_patches_per_batch = num_extra_tokens + split_lengths.sum().data_ptr<int64_t>()[0];

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int max_threads = std::min(
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);

    auto patches = at::empty({num_batches * num_patches_per_batch, 4}, indices.options());
    auto batch_lengths = at::full({num_batches}, num_patches_per_batch, split_sizes_cpu.options());

    grained_sample_index_to_patch_dense_kernel
        <<<at::cuda::ATenCeilDiv(num_batches * num_patches_per_batch, max_threads),
           max_threads,
           0,
           stream>>>(
        indices.packed_accessor64<int64_t, 3>(),
        split_sizes.packed_accessor64<int64_t, 1>(),
        patches.packed_accessor64<int64_t, 2>(),
        region_size,
        num_extra_tokens,
        num_patches_per_batch
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(patches, batch_lengths);
}
