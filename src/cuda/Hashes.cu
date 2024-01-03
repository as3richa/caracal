#include <cassert>
#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>

#include "DevicePointer.h"

namespace caracal {

void ComputeDots(PitchedView<float> dots,
                 cublasHandle_t cublas_handle,
                 ConstPitchedView<float> vectors,
                 size_t vectors_count,
                 ConstPitchedView<float> planes,
                 size_t planes_count,
                 size_t dimensions) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const cublasStatus_t status = cublasSgemm(cublas_handle,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            planes_count,
                                            vectors_count,
                                            dimensions,
                                            &alpha,
                                            planes.Ptr(),
                                            planes.Pitch() / sizeof(float),
                                            vectors.Ptr(),
                                            vectors.Pitch() / sizeof(float),
                                            &beta,
                                            dots.Ptr(),
                                            dots.Pitch() / sizeof(float));
  if (status != CUBLAS_STATUS_SUCCESS) {
    // FIXME: ???
    CARACAL_CUDA_EXCEPTION_THROW_ON_ERROR(cudaErrorUnknown);
  }
}

__global__ static void ConvertDotsToBitsKernel(PitchedView<uint32_t> bits,
                                               ConstPitchedView<float> dots,
                                               size_t width,
                                               size_t height) {
  const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= width || y >= height) {
    return;
  }

  const uint32_t word = __ballot_sync(__activemask(), dots[y][x] >= 0.0);

  if (threadIdx.x % 32 == 0) {
    bits[y][x / 32] = word;
  }
}

void ConvertDotsToBits(PitchedView<uint64_t> bits,
                       ConstPitchedView<float> dots,
                       size_t width,
                       size_t height) {
  const dim3 block(64, 16, 1);
  const dim3 grid(
      (2 * width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);

  ConvertDotsToBitsKernel<<<grid, block>>>(
      static_cast<PitchedView<uint32_t>>(bits), dots, width, height);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();
}

void ComputeHashes(PitchedView<uint64_t> hashes,
                   cublasHandle_t cublas_handle,
                   ConstPitchedView<float> vectors,
                   size_t vectors_count,
                   ConstPitchedView<float> planes,
                   size_t planes_count,
                   size_t dimensions) {

  PitchedDevicePointer<float> dots =
      PitchedDevicePointer<float>::MallocPitch(planes_count, vectors_count);

  ComputeDots(dots.View(),
              cublas_handle,
              vectors,
              vectors_count,
              planes,
              planes_count,
              dimensions);

  ConvertDotsToBits(hashes, dots.ConstView(), planes_count, vectors_count);
}

__global__ static void
ComputeHashDistancesKernel(PitchedView<uint16_t> distances,
                           ConstPitchedView<uint64_t> left_hashes,
                           size_t left_hashes_count,
                           ConstPitchedView<uint64_t> right_hashes,
                           size_t right_hashes_count,
                           size_t hash_length) {
  extern __shared__ uint64_t shared_memory[];
  uint64_t *cached_left_hashes = shared_memory;
  uint64_t *cached_right_hashes = shared_memory + hash_length * blockDim.y;

  const size_t left_hash_i = threadIdx.y + blockIdx.y * blockDim.y;
  const size_t right_hash_i = threadIdx.z + blockIdx.z * blockDim.z;

  if (left_hash_i >= left_hashes_count || right_hash_i >= right_hashes_count) {
    return;
  }

  uint64_t *cached_left_hash = cached_left_hashes + threadIdx.y * hash_length;

  if (threadIdx.z == 0) {
#pragma unroll
    for (size_t i = threadIdx.x; i < hash_length; i += blockDim.x) {
      cached_left_hash[i] = left_hashes[left_hash_i][i];
    }
  }

  uint64_t *cached_right_hash = cached_right_hashes + threadIdx.z * hash_length;

  if (threadIdx.y == 0) {
#pragma unroll
    for (size_t i = threadIdx.x; i < hash_length; i += blockDim.x) {
      cached_right_hash[i] = right_hashes[right_hash_i][i];
    }
  }

  __syncthreads();

  unsigned int partial_distance = 0;

#pragma unroll
  for (size_t i = threadIdx.x; i < hash_length; i += blockDim.x) {
    partial_distance += __popcll(cached_left_hash[i] ^ cached_right_hash[i]);
  }

  assert(blockDim.x == 4 || blockDim.x == 2 || blockDim.x == 1);

  if (blockDim.x == 4) {
    partial_distance +=
        __shfl_down_sync(0xffffffff, partial_distance, 2, blockDim.x);
  }

  if (blockDim.x >= 2) {
    partial_distance +=
        __shfl_down_sync(0xffffffff, partial_distance, 1, blockDim.x);
  }

  if (threadIdx.x == 0) {
    distances[left_hash_i][right_hash_i] = partial_distance;
  }
}

void ComputeHashDistances(PitchedView<uint16_t> distances,
                          ConstPitchedView<uint64_t> left_hashes,
                          size_t left_hashes_count,
                          ConstPitchedView<uint64_t> right_hashes,
                          size_t right_hashes_count,
                          size_t hash_length) {
  size_t thread_count;
  if (hash_length >= 4) {
    thread_count = 4;
  } else if (hash_length >= 2) {
    thread_count = 2;
  } else {
    thread_count = 1;
  }

  size_t right_block_size;
  size_t left_block_size;

  if (left_hashes_count >= 32 && thread_count == 1) {
    right_block_size = left_block_size = 32;
  } else {
    if (left_hashes_count >= 8) {
      left_block_size = 8;
    } else if (left_hashes_count >= 4) {
      left_block_size = 4;
    } else if (left_hashes_count >= 2) {
      left_block_size = 2;
    } else {
      left_block_size = 1;
    }

    right_block_size = 1024 / left_block_size / thread_count;
  }

  const size_t shared_memory_size =
      sizeof(uint64_t) * (left_block_size + right_block_size) * hash_length;

  dim3 block(thread_count, left_block_size, right_block_size);
  dim3 grid(1,
            (left_hashes_count + block.y - 1) / block.y,
            (right_hashes_count + block.z - 1) / block.z);

  printf("%d %d %d\n", block.x, block.y, block.z);
  printf("%d %d\n", grid.y, grid.z);
  printf("%d\n", shared_memory_size);
  ComputeHashDistancesKernel<<<grid, block, shared_memory_size>>>(
      distances,
      left_hashes,
      left_hashes_count,
      right_hashes,
      right_hashes_count,
      hash_length);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();
}

} // namespace caracal