#include <cassert>
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "../GenerateRandomPlanes.h"

int main(void) {
  const size_t count = 1337;
  const size_t dimensions = 4200;

  cudaError_t error;

  float *planes;
  size_t pitch;
  error =
      caracal::GenerateRandomPlanes(&planes, &pitch, count, dimensions, 1337);
  assert(error == cudaSuccess);

  std::vector<float> host_planes(count * dimensions);
  error = cudaMemcpy2D(host_planes.data(), dimensions * sizeof(float), planes,
                       pitch, dimensions, count, cudaMemcpyDeviceToHost);
  assert(error == cudaSuccess);

  error = cudaFree(planes);
  assert(error == cudaSuccess);

  for (size_t y = 0; y < count; y++) {
    for (size_t x = 0; x < dimensions; x++) {
      const float value = host_planes[x + y * dimensions];
      assert(-1.0 <= value && value <= 1.0);
    }
  }
}