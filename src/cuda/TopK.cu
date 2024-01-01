#include "TopK.h"

#include <algorithm>
#include <vector>

#include <cassert>
#include <cstdio>

namespace caracal {

static const size_t histogram_bits = 12;
static const size_t histogram_size = 1 << histogram_bits;
static const size_t block_size = 1024;
static const size_t block_histogram_steps = 4;

__device__ static void ComputePrefixSums(unsigned int *data,
                                         unsigned int *temp) {
  const int k = threadIdx.x;

  const uint2 u = ((uint2 *)data)[2 * k];
  const uint2 v = ((uint2 *)data)[2 * k + 1];

  temp[2 * k] = u.x + u.y;
  temp[2 * k + 1] = v.x + v.y;

  const unsigned int ux = u.x;
  const unsigned int vx = v.x;

  if (k == 0) {
    data[histogram_size] = data[histogram_size - 1];
  }

  int offset = 1;
#pragma unroll

  for (int d = histogram_size >> 1; d > 0; d >>= 1) {
    __syncthreads();

    if (k < d) {
      // const int offset = 1 << (histogram_bits - 1 - d);
      const int i = offset * (2 * k + 1) - 1;
      const int j = offset * (2 * k + 2) - 1;
      temp[j] += temp[i];
    }
    offset *= 2;
  }

  __syncthreads();

  if (k == 0) {
    temp[histogram_size - 1] = 0;
  }

#pragma unroll
  for (int d = 1; d < histogram_size; d *= 2) {

    offset >>= 1;
    __syncthreads();

    if (k < d) {
      // const int offset = 1 << (histogram_bits - 1 - d);
      const int i = offset * (2 * k + 1) - 1;
      const int j = offset * (2 * k + 2) - 1;
      unsigned int t = temp[i];
      temp[i] = temp[j];
      temp[j] += t;
    }
  }

  __syncthreads();

  unsigned int p = temp[2 * k];
  unsigned int q = temp[2 * k + 1];

  ((uint4 *)data)[k] = make_uint4(p, p + ux, q, q + vx);

  __syncthreads();

  if (k == 0) {
    data[histogram_size] += data[histogram_size - 1];
  }

  __syncthreads();
}

template <bool truncate>
__global__ static void
ComputeHistogramsKernel(uint16_t *values, size_t count, size_t batches,
                        size_t values_pitch, size_t truncate_width,
                        unsigned int *histograms, size_t histograms_pitch,
                        size_t k) {
  __shared__ unsigned int block_histogram[histogram_size];

#pragma unroll
  for (size_t i = 0; i < block_histogram_steps; i++) {
    block_histogram[threadIdx.x + i * block_size] = 0;
  }
  __syncthreads();

  const size_t i = threadIdx.x + blockIdx.x * block_size;
  const size_t batch = blockIdx.y;

  if (i < count) {
    uint16_t *batch_values = values + batch * (values_pitch / sizeof(uint16_t));
    const uint16_t value = batch_values[i];
    const int truncated_value = (truncate) ? (value >> truncate_width) : value;
    atomicAdd(&block_histogram[truncated_value], 1);
  }

  __syncthreads();

  unsigned int *histogram =
      histograms + batch * (histograms_pitch / sizeof(unsigned int));

#pragma unroll
  for (size_t i = 0; i < block_histogram_steps; i++) {
    __syncthreads();
    atomicAdd(&histogram[threadIdx.x + i * block_size],
              block_histogram[threadIdx.x + i * block_size]);
  }
}

__global__ static void
ComputePrefixSumsAndThresholdsKernel(unsigned int *histograms,
                                     size_t histograms_pitch,
                                     uint16_t *thresholds, size_t k) {
  // FIXME: wrong alignment
  __shared__ unsigned int block_histogram[histogram_size + 1];

  const size_t batch = blockIdx.y;
  unsigned int *histogram =
      histograms + batch * (histograms_pitch / sizeof(unsigned int));

  const size_t x = threadIdx.x;

  ComputePrefixSums(histogram, block_histogram);

#pragma unroll
  for (size_t i = 0; i < block_histogram_steps; i++) {
    size_t j = x + i * block_size;

    if (histogram[j] >= k) {
      break;
    }

    if (k <= histogram[j + 1]) {
      thresholds[batch] = j;
    }
  }
}

template <bool truncate>
__global__ static void
SelectKernel(size_t *results, size_t results_pitch, uint16_t *values,
             size_t count, size_t batches, size_t values_pitch,
             size_t truncate_width, uint16_t *thresholds,
             unsigned int *histograms, size_t histograms_pitch, size_t k) {
  const size_t i = threadIdx.x + blockIdx.x * block_size;
  const size_t batch = blockIdx.y;

  if (i >= count) {
    return;
  }

  uint16_t *batch_values = values + batch * (values_pitch / sizeof(uint16_t));
  const uint16_t value = batch_values[i];
  const int truncated_value = (truncate) ? (value >> truncate_width) : value;

  const size_t threshold = thresholds[batch];

  unsigned int *histogram =
      histograms + batch * (histograms_pitch / sizeof(unsigned int));

  size_t *result = results + batch * (results_pitch / sizeof(size_t));

  if (truncated_value > histogram_size) {
    for (size_t x = 0; x < k; x++) {
      result[x] = 0x10101010101010LLU;
    }
  }

  if (truncated_value < threshold) {
    unsigned int j = atomicAdd(&histogram[truncated_value], 1);
    result[j] = i;

    if (j >= k) {
      for (size_t x = 0; x < k; x++) {
        result[x] = ~(size_t)0;
      }
    }
  } else if (truncated_value == threshold) {
    unsigned int j = atomicAdd(&histogram[truncated_value], 1);
    if (j < k) {
      result[j] = i;
    }
  }
}

cudaError_t TopK(size_t **results, size_t *results_pitch, uint16_t *values,
                 size_t count, size_t batches, size_t values_pitch, size_t bits,
                 size_t k) {
  cudaError_t error;

  error = cudaMallocPitch(results, results_pitch, k * sizeof(size_t), batches);
  if (error != cudaSuccess) {
    return error;
  }

  if (k > count) {
    k = count;
  }

  uint16_t *thresholds;
  error = cudaMalloc(&thresholds, batches * sizeof(uint16_t));
  if (error != cudaSuccess) {
    cudaFree(*results);
    return error;
  }

  unsigned int *histograms;
  size_t histograms_pitch;
  error = cudaMallocPitch(&histograms, &histograms_pitch,
                          (histogram_size + 1) * sizeof(unsigned int), batches);

  if (error != cudaSuccess) {
    cudaFree(*results);
    cudaFree(thresholds);
    return error;
  }

  dim3 block(block_size);
  dim3 grid((count + block.x - 1) / block.x, batches);

  if (bits > histogram_bits) {
    ComputeHistogramsKernel<true>
        <<<grid, block>>>(values, count, batches, values_pitch, 16 - bits,
                          histograms, histograms_pitch, k);
  } else {
    ComputeHistogramsKernel<false><<<grid, block>>>(values, count, batches,
                                                    values_pitch, 0, histograms,
                                                    histograms_pitch, k);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaFree(*results);
    cudaFree(thresholds);
    cudaFree(histograms);
    return error;
  }

  ComputePrefixSumsAndThresholdsKernel<<<dim3(1, batches), block>>>(
      histograms, histograms_pitch, thresholds, k);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaFree(*results);
    cudaFree(thresholds);
    cudaFree(histograms);
    return error;
  }

  unsigned int *host_histograms = (unsigned int *)malloc(
      (histogram_size + 1) * batches * sizeof(unsigned int));
  cudaMemcpy2D(host_histograms, (histogram_size + 1) * sizeof(unsigned int),
               histograms, histograms_pitch,
               (histogram_size + 1) * sizeof(unsigned int), batches,
               cudaMemcpyDeviceToHost);

  uint16_t *host_thresholds = (uint16_t *)malloc(sizeof(uint16_t) * batches);
  cudaMemcpy(host_thresholds, thresholds, sizeof(uint16_t) * batches,
             cudaMemcpyDeviceToHost);

  for (size_t y = 0; y < batches; y++) {
    unsigned int *host_histogram = host_histograms + y * (histogram_size + 1);
    assert(host_histogram[0] == 0);
    // printf("%d %d\n", host_histogram[histogram_size], count);
    assert(host_histogram[histogram_size] == count);
    for (size_t i = 1; i <= histogram_size; i++) {
      assert(host_histogram[i - 1] <= host_histogram[i]);
      assert(host_histogram[i] <= count);
      // printf("%zu: %u; ", i, host_histogram[i]);
    }

    // printf("%d %d %d %d\n", host_thresholds[y],
    // host_histogram[host_thresholds[y]], k, host_histogram[host_thresholds[y]+
    // 1]);
    assert(host_thresholds[y] < histogram_size);
    assert(host_histogram[host_thresholds[y]] < k &&
           k <= host_histogram[host_thresholds[y] + 1]);
  }

  if (bits > histogram_bits) {
    SelectKernel<true><<<grid, block>>>(
        *results, *results_pitch, values, count, batches, values_pitch,
        16 - bits, thresholds, histograms, histograms_pitch, k);
  } else {
    SelectKernel<false><<<grid, block>>>(
        *results, *results_pitch, values, count, batches, values_pitch, 0,
        thresholds, histograms, histograms_pitch, k);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaFree(*results);
    cudaFree(thresholds);
    cudaFree(histograms);
    return error;
  }

  cudaFree(thresholds);
  cudaFree(histograms);

  return cudaSuccess;
}

} // namespace caracal