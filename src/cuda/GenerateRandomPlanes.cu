#include "GenerateRandomPlanes.h"

#include <cstdint>

#include <curand_kernel.h>

namespace caracal {

__global__ static void GenerateRandomPlanesKernel(float *planes, size_t count,
                                                  size_t dimensions,
                                                  size_t pitch, uint64_t seed) {
  const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t y = threadIdx.y + blockIdx.y * blockDim.y;
  if (y >= count || x >= dimensions) {
    return;
  }

  const size_t i = x + y * (pitch / sizeof(float));
  curandState_t state;
  curand_init(seed, i, 0, &state);
  planes[i] = curand_uniform(&state) - 0.5;
}

cudaError_t GenerateRandomPlanes(float **planes, size_t *pitch, size_t count,
                                 size_t dimensions, uint64_t seed) {
  cudaError_t error;

  error = cudaMallocPitch((void **)planes, pitch, dimensions * sizeof(float),
                          count);
  if (error != cudaSuccess) {
    return error;
  }

  const dim3 block(16, 16, 1);
  const dim3 grid((dimensions + block.x - 1) / block.x,
                  (count + block.y - 1) / block.y, 1);

  GenerateRandomPlanesKernel<<<grid, block>>>(*planes, count, dimensions,
                                              *pitch, seed);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaFree(*planes);
    return error;
  }

  return cudaSuccess;
}

} // namespace caracal