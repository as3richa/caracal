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
static const size_t reduced_histogram_size = 2048;

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
    atomicAdd(&histograms[batch][threadIdx.x + i * block_size],
              block_histogram[threadIdx.x + i * block_size]);
  }
}

__device__ static void ComputePrefixSums(unsigned int *data,
                                         unsigned int *temp) {
  const int thread = threadIdx.x;

  const uint4 v = ((uint4 *)data)[thread];

  temp[2 * thread] = v.x + v.y;
  temp[2 * thread + 1] = v.z + v.w;

  const unsigned int v1 = v.x;
  const unsigned int v2 = v.z;

  int offset = 1;

#pragma unroll
  for (int d = reduced_histogram_size >> 1; d > 0; d >>= 1) {
    __syncthreads();

    if (thread < d) {
      const int i = offset * (2 * thread + 1) - 1;
      const int j = offset * (2 * thread + 2) - 1;
      temp[j] += temp[i];
    }
    offset *= 2;
  }

  __syncthreads();

  if (thread == 0) {
    temp[reduced_histogram_size] = temp[reduced_histogram_size - 1];
    temp[reduced_histogram_size - 1] = 0;
  }

#pragma unroll
  for (int d = 1; d < reduced_histogram_size; d *= 2) {

    offset >>= 1;
    __syncthreads();

    if (thread < d) {
      const int i = offset * (2 * thread + 1) - 1;
      const int j = offset * (2 * thread + 2) - 1;
      unsigned int t = temp[i];
      temp[i] = temp[j];
      temp[j] += t;
    }
  }

  __syncthreads();

  const unsigned int p = temp[2 * thread];
  const unsigned int q = temp[2 * thread + 1];
  ((uint4 *)data)[thread] = make_uint4(p, p + v1, q, q + v2);

  if (thread == 0) {
    data[histogram_size] = temp[reduced_histogram_size];
  }

  __syncthreads();
}

__global__ static void
ComputePrefixSumsAndThresholdsKernel(PitchedView<unsigned int> histograms,
                                     uint16_t *thresholds,
                                     size_t count,
                                     size_t k) {
  __shared__ __align__(
      4 * sizeof(unsigned int)) unsigned int temp[reduced_histogram_size + 1];

  assert(k <= count);
  assert(blockDim.x == block_size && blockDim.y == 1 && blockDim.z == 1);

  const size_t batch = blockIdx.y;
  const size_t x = threadIdx.x;

  ComputePrefixSums(histograms[batch], temp);

#ifndef NDEBUG
  __syncthreads();

  assert(histograms[batch][0] == 0);
  assert(histograms[batch][histogram_size] == count);

  for (size_t i = 0; i < block_histogram_steps; i++) {
    size_t j = x + i * block_size;
    assert(histograms[batch][j] <= histograms[batch][j + 1]);
  }
#else
  (void)count;
#endif

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
  assert(histograms[batch][threshold] < k &&
         k <= histograms[batch][threshold + 1]);
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
      histograms.View(), thresholds.Ptr(), count, k);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();

  SelectKernel<<<grid, block>>>(
      results, values, count, batches, thresholds.Ptr(), histograms.View(), k);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR(error);
}

} // namespace caracal