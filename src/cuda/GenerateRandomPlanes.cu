#include "GenerateRandomPlanes.h"

#include <cstdint>

#include <curand_kernel.h>

#include "DevicePointer.h"

namespace caracal {

__global__ static void GenerateRandomPlanesKernel(PitchedView<float> planes,
                                                  size_t count,
                                                  size_t dimensions,
                                                  uint64_t seed) {
  const size_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t y = threadIdx.y + blockIdx.y * blockDim.y;

  if (y >= count || x >= dimensions) {
    return;
  }

  const size_t sequence_number = y * dimensions + x;

  curandState_t state;
  curand_init(seed, sequence_number, 0, &state);

  planes[y][x] = 2 * (curand_uniform(&state) - 0.5);
}

void GenerateRandomPlanes(PitchedView<float> planes,
                          size_t count,
                          size_t dimensions,
                          uint64_t seed) {
  const dim3 block(16, 16, 1);
  const dim3 grid(
      (dimensions + block.x - 1) / block.x, (count + block.y - 1) / block.y, 1);

  GenerateRandomPlanesKernel<<<grid, block>>>(planes, count, dimensions, seed);
  const cudaError_t error = cudaGetLastError();
  CARACAL_CUDA_EXCEPTION_THROW_ON_ERORR(error);
}

} // namespace caracal