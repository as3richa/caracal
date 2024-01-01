#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>

#include <cstdio>
namespace caracal {

cudaError_t ComputeDots(float **dots, size_t *dots_pitch,
                        cublasHandle_t cublas_handle, float *vectors,
                        size_t vectors_count, size_t vectors_pitch,
                        float *planes, size_t planes_count, size_t planes_pitch,
                        size_t dimensions) {
  cudaError_t error;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  error = cudaMallocPitch((void **)dots, dots_pitch,
                          planes_count * sizeof(float), vectors_count);
  if (error != cudaSuccess) {
    return error;
  }

  const cublasStatus_t status = cublasSgemm(
      cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, planes_count, vectors_count,
      dimensions, &alpha, planes, planes_pitch / sizeof(float), vectors,
      vectors_pitch / sizeof(float), &beta, *dots, *dots_pitch / sizeof(float));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaFree(dots);
    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

__global__ static void ConvertDotsToBitsKernel(uint32_t *bits,
                                               size_t bits_pitch, float *dots,
                                               size_t height, size_t width,
                                               size_t dots_pitch) {
  const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= width || y >= height) {
    return;
  }

  const size_t dots_i = x + y * (dots_pitch / sizeof(float));
  const uint32_t word = __ballot_sync(__activemask(), dots[dots_i] >= 0.0);

  if (threadIdx.x % 32 == 0) {
    const size_t bits_i = (x / 32) + y * (bits_pitch / sizeof(uint32_t));
    bits[bits_i] = word;
  }
}

cudaError_t ConvertDotsToBits(uint64_t **bits, size_t *bits_pitch, float *dots,
                              size_t height, size_t width, size_t dots_pitch) {
  cudaError_t error;

  error = cudaMallocPitch((void **)bits, bits_pitch,
                          ((width + 64 - 1) / 64) * sizeof(uint64_t), height);
  if (error != cudaSuccess) {
    return error;
  }

  const dim3 block(64, 16, 1);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y, 1);

  ConvertDotsToBitsKernel<<<grid, block>>>((uint32_t *)*bits, *bits_pitch, dots,
                                           height, width, dots_pitch);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaFree(*bits);
    return error;
  }

  return cudaSuccess;
}

cudaError_t ComputeHashes(uint64_t **hashes, size_t *hashes_pitch,
                          cublasHandle_t cublas_handle, float *vectors,
                          size_t vectors_count, size_t vectors_pitch,
                          float *planes, size_t planes_count,
                          size_t planes_pitch, size_t dimensions) {
  cudaError_t error;

  float *dots;
  size_t dots_pitch;

  error = ComputeDots(&dots, &dots_pitch, cublas_handle, vectors, vectors_count,
                      vectors_pitch, planes, planes_count, planes_pitch,
                      dimensions);
  if (error != cudaSuccess) {
    return error;
  }

  error = ConvertDotsToBits(hashes, hashes_pitch, dots, vectors_count,
                            planes_count, dots_pitch);
  if (error != cudaSuccess) {
    cudaFree(dots);
    return error;
  }

  cudaFree(dots);
  return cudaSuccess;
}

__global__ static void ComputeHashDistancesKernel(
    uint16_t *distances, size_t distances_pitch, uint64_t *left_hashes,
    size_t left_hashes_count, size_t left_hashes_pitch, uint64_t *right_hashes,
    size_t right_hashes_count, size_t right_hashes_pitch, size_t hash_length) {
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
    const uint64_t *left_hash =
        left_hashes + left_hash_i * (left_hashes_pitch / sizeof(uint64_t));

#pragma unroll
    for (size_t i = threadIdx.x; i < hash_length; i += blockDim.x) {
      cached_left_hash[i] = left_hash[i];
    }
  }

  uint64_t *cached_right_hash = cached_right_hashes + threadIdx.z * hash_length;

  if (threadIdx.y == 0) {
    const uint64_t *right_hash =
        right_hashes + right_hash_i * (right_hashes_pitch / sizeof(uint64_t));

#pragma unroll
    for (size_t i = threadIdx.x; i < hash_length; i += blockDim.x) {
      cached_right_hash[i] = right_hash[i];
    }
  }

  __syncthreads();

  unsigned int partial_distance = 0;

#pragma unroll
  for (size_t i = threadIdx.x; i < hash_length; i += blockDim.x) {
    partial_distance += __popcll(cached_left_hash[i] ^ cached_right_hash[i]);
  }

#define PARTIAL_DISTANCE_SHUFFLE(delta)                                        \
  do {                                                                         \
    if (blockDim.x >= (2 * delta)) {                                           \
      partial_distance +=                                                      \
          __shfl_down_sync(0xffffffff, partial_distance, delta, blockDim.x);   \
    }                                                                          \
  } while (0)

  PARTIAL_DISTANCE_SHUFFLE(2);
  PARTIAL_DISTANCE_SHUFFLE(1);

#undef PARTIAL_DISTANCE_SHUFFLE

  if (threadIdx.x == 0) {
    uint16_t *distance = distances +
                         left_hash_i * (distances_pitch / sizeof(uint16_t)) +
                         right_hash_i;
    *distance = partial_distance;
  }
}

cudaError_t ComputeHashDistances(
    uint16_t **distances, size_t *distances_pitch, uint64_t *left_hashes,
    size_t left_hashes_count, size_t left_hashes_pitch, uint64_t *right_hashes,
    size_t right_hashes_count, size_t right_hashes_pitch, size_t hash_length) {
  cudaError_t error;

  error =
      cudaMallocPitch((void **)distances, distances_pitch,
                      right_hashes_count * sizeof(uint16_t), left_hashes_count);
  if (error != cudaSuccess) {
    return error;
  }

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

  if (right_hashes_count >= 32 && thread_count == 1) {
    left_block_size = right_block_size = 32;
  } else {
    if (right_hashes_count >= 8) {
      right_block_size = 8;
    } else if (right_hashes_count >= 4) {
      right_block_size = 4;
    } else if (right_hashes_count >= 2) {
      right_block_size = 2;
    } else {
      right_block_size = 1;
    }

    left_block_size = 1024 / right_block_size / thread_count;
  }

  const size_t shared_memory_size =
      sizeof(uint64_t) * (left_block_size + right_block_size) * hash_length;
  
  printf("%d %d %d\n", thread_count, left_block_size, right_block_size);

  dim3 block(thread_count, left_block_size, right_block_size);
  printf("%d %d %d\n", 1, (left_hashes_count + block.y - 1) / block.y,
            (right_hashes_count + block.z - 1) / block.z);
  dim3 grid(1, (left_hashes_count + block.y - 1) / block.y,
            (right_hashes_count + block.z - 1) / block.z);

  ComputeHashDistancesKernel<<<grid, block, shared_memory_size>>>(
      *distances, *distances_pitch, left_hashes, left_hashes_count,
      left_hashes_pitch, right_hashes, right_hashes_count, right_hashes_pitch,
      hash_length);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaFree(*distances);
    return error;
  }

  return cudaSuccess;
}

} // namespace caracal