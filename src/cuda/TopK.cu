#include "TopK.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include "DevicePointer.h"

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

__global__ static void
ComputeHistogramsKernel(PitchedView<uint32_t> histograms,
                        ConstPitchedView<uint16_t> values,
                        size_t count,
                        size_t batches,
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
    const uint16_t value = values[batch][i];
    atomicAdd(&block_histogram[value], 1);
  }

  __syncthreads();

#pragma unroll
  for (size_t i = 0; i < block_histogram_steps; i++) {
    __syncthreads();
    atomicAdd(&histograms[batch][threadIdx.x + i * block_size],
              block_histogram[threadIdx.x + i * block_size]);
  }
}

__global__ static void ComputePrefixSumsAndThresholdsKernel(
    PitchedView<unsigned int> histograms, uint16_t *thresholds, size_t k) {
  __shared__ __align__(
      2 *
      sizeof(unsigned int)) unsigned int block_histogram[histogram_size + 1];

  const size_t batch = blockIdx.y;
  const size_t x = threadIdx.x;

  ComputePrefixSums(histograms[batch], block_histogram);

#pragma unroll
  for (size_t i = 0; i < block_histogram_steps; i++) {
    size_t j = x + i * block_size;

    if (histograms[batch][j] >= k) {
      break;
    }

    if (k <= histograms[batch][j + 1]) {
      thresholds[batch] = j;
    }
  }

#ifndef NDEBUG
  __syncthreads();
  const uint16_t threshold = thresholds[batch];
  assert(threshold < histogram_size);
  assert(histograms[batch][threshold] < thresholds[batch] &&
         thresholds[batch] <= histograms[batch][threshold + 1]);
#endif
}

__global__ static void SelectKernel(PitchedView<size_t> results,
                                    ConstPitchedView<uint16_t> values,
                                    size_t count,
                                    size_t batches,
                                    const uint16_t *thresholds,
                                    PitchedView<unsigned int> histograms,
                                    size_t k) {
  const size_t i = threadIdx.x + blockIdx.x * block_size;
  const size_t batch = blockIdx.y;

  if (i >= count) {
    return;
  }

  const uint16_t value = values[batch][i];
  const size_t threshold = thresholds[batch];

  if (value <= threshold) {
    const unsigned int j = atomicAdd(&histograms[batch][value], 1);
    assert(value == threshold || j < k);

    if (value < threshold || j < k) {
      results[batch][j] = i;
    }
  }
}

void TopK(PitchedView<size_t> results,
          ConstPitchedView<uint16_t> values,
          size_t count,
          size_t batches,
          size_t k) {
  if (k > count) {
    k = count;
  }

  DevicePointer<uint16_t> thresholds = DevicePointer<uint16_t>::Malloc(batches);

  PitchedDevicePointer<unsigned int> histograms =
      PitchedDevicePointer<unsigned int>::MallocPitch(histogram_size + 1,
                                                      batches);
  cudaMemset2D(histograms.View().Ptr(),
               histograms.View().Pitch(),
               0,
               (histogram_size + 1) * sizeof(unsigned int),
               batches);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();

  dim3 block(block_size);
  dim3 grid((count + block.x - 1) / block.x, batches);

  ComputeHistogramsKernel<<<grid, block>>>(
      histograms.View(), values, count, batches, k);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();

  ComputePrefixSumsAndThresholdsKernel<<<dim3(1, batches), block>>>(
      histograms.View(), thresholds.Ptr(), k);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();

  /*
    unsigned int *host_histograms = (unsigned int *)malloc(
        batches * (histogram_size + 1) * sizeof(unsigned int));
    error = cudaMemcpy2D(
        host_histograms, (histogram_size + 1) * sizeof(unsigned int),
    histograms, histograms_pitch, (histogram_size + 1) * sizeof(unsigned int),
    batches, cudaMemcpyDeviceToHost); assert(error == cudaSuccess);

    uint16_t *host_thresholds = (uint16_t *)malloc(sizeof(uint16_t) * batches);
    error = cudaMemcpy(host_thresholds, thresholds, sizeof(uint16_t) * batches,
                       cudaMemcpyDeviceToHost);
    assert(error == cudaSuccess);

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
      // host_histogram[host_thresholds[y]], k,
    host_histogram[host_thresholds[y]+
      // 1]);
      assert(host_thresholds[y] < histogram_size);
      assert(host_histogram[host_thresholds[y]] < k &&
             k <= host_histogram[host_thresholds[y] + 1]);
    }
    */

  SelectKernel<<<grid, block>>>(
      results, values, count, batches, thresholds.Ptr(), histograms.View(), k);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR(error);
}

} // namespace caracal