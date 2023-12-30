#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>

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
  if (y >= height || x >= width) {
    return;
  }

  const size_t dots_i = x + y * (dots_pitch / sizeof(float));
  const uint32_t word = __ballot_sync(0xffffffff, dots[dots_i] >= 0.0);

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

  const dim3 block(64, 1, 1);
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

template <size_t max_hash_length, size_t left_block_size,
          size_t right_block_size>
__global__ static void ComputeHashDistancesKernel(
    uint16_t *distances, size_t distances_pitch, uint64_t *left_hashes,
    size_t left_hashes_count, size_t left_hashes_pitch, uint64_t *right_hashes,
    size_t right_hashes_count, size_t right_hashes_pitch, size_t hash_length) {
  __shared__ uint64_t cached_left_hashes[left_block_size][max_hash_length];
  __shared__ uint64_t cached_right_hashes[right_block_size][max_hash_length];
  __shared__ uint16_t
      partial_distances[left_block_size][right_block_size][max_hash_length];

  const size_t offset = threadIdx.x;
  const size_t left_hash = threadIdx.y + blockIdx.y * left_block_size;
  const size_t right_hash = threadIdx.z + blockIdx.z * right_block_size;
  if (offset >= hash_length || left_hash >= left_hashes_count ||
      right_hash >= right_hashes_count) {
    return;
  }

  const size_t left_i =
      offset + left_hash * (left_hashes_pitch / sizeof(uint64_t));
  cached_left_hashes[left_hash][offset] = left_hashes[left_i];

  const size_t right_i =
      offset + right_hash * (right_hashes_pitch / sizeof(uint64_t));
  cached_right_hashes[right_hash][offset] = right_hashes[right_i];

  __syncthreads();

  uint16_t *q = &partial_distances[threadIdx.y][threadIdx.z][offset];

  *q = __popcll(cached_left_hashes[threadIdx.y][offset] ^
                cached_right_hashes[threadIdx.z][offset]);

  __syncthreads();

#define UNROLLED_INTERLEAVED_ADDRESSING_SUM(delta)                             \
  do {                                                                         \
    if (max_hash_length > delta) {                                             \
      if (offset % (2 * delta) == 0 && offset + delta < hash_length) {         \
        *q += *(q + delta);                                                    \
      }                                                                        \
      __syncthreads();                                                         \
    }                                                                          \
  } while (0)

  UNROLLED_INTERLEAVED_ADDRESSING_SUM(1);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(2);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(4);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(8);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(16);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(32);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(64);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(128);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(256);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(512);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(1024);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(2048);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(4096);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(8192);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(16384);
  UNROLLED_INTERLEAVED_ADDRESSING_SUM(32768);

#undef UNROLLED_INTERLEAVED_ADDRESSING_SUM

  if (offset == 0) {
    const size_t distances_i =
        right_hash + left_hash * (distances_pitch / sizeof(uint16_t));
    distances[distances_i] = *q;
  }
}

cudaError_t ComputeHashDistances(
    uint16_t **distances, size_t *distances_pitch, uint64_t *left_hashes,
    size_t left_hashes_count, size_t left_hashes_pitch, uint64_t *right_hashes,
    size_t right_hashes_count, size_t right_hashes_pitch, size_t hash_length) {
  cudaError_t error;

  error =
      cudaMallocPitch((void **)distances, distances_pitch, right_hashes_count,
                      left_hashes_count * sizeof(uint16_t));
  if (error != cudaSuccess) {
    return error;
  }

#define SELECT_KERNEL_FOR_HASH_LENGTH(max_hash_length, left_block_size,        \
                                      right_block_size)                        \
  else if (hash_length < max_hash_length) {                                    \
    dim3 block(hash_length, left_block_size, right_block_size);                \
    dim3 grid(hash_length, (left_hashes_count + block.x - 1) / block.x,        \
              (right_hashes_count + block.y - 1) / block.y);                   \
    ComputeHashDistancesKernel<max_hash_length, left_block_size,               \
                               right_block_size><<<grid, block>>>(             \
        *distances, *distances_pitch, left_hashes, left_hashes_count,          \
        left_hashes_pitch, right_hashes, right_hashes_count,                   \
        right_hashes_pitch, hash_length);                                      \
  }

  {
    if (false) {
    }
    SELECT_KERNEL_FOR_HASH_LENGTH(8, 8, 16)
    SELECT_KERNEL_FOR_HASH_LENGTH(16, 8, 8)
    SELECT_KERNEL_FOR_HASH_LENGTH(32, 4, 8)
    SELECT_KERNEL_FOR_HASH_LENGTH(64, 4, 4)
    SELECT_KERNEL_FOR_HASH_LENGTH(128, 2, 4)
    SELECT_KERNEL_FOR_HASH_LENGTH(256, 2, 2)
    SELECT_KERNEL_FOR_HASH_LENGTH(512, 1, 2)
    SELECT_KERNEL_FOR_HASH_LENGTH(1024, 1, 1)
  }

#undef SELECT_KERNEL_FOR_HASH_LENGTH
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaFree(*distances);
    return error;
  }

  return cudaSuccess;
}

} // namespace caracal