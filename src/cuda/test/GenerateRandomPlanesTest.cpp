#include <cassert>
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "../DevicePointer.h"
#include "../GenerateRandomPlanes.h"

int main(void) {
  const size_t count = 1337;
  const size_t dimensions = 4200;

  std::vector<float> host_planes(count * dimensions);

  caracal::PitchedDevicePointer<float> planes =
      caracal::PitchedDevicePointer<float>::MallocPitch(count, dimensions);

  caracal::GenerateRandomPlanes(planes.View(), count, dimensions, 1337);

  const cudaError_t error = cudaMemcpy2D(host_planes.data(),
                                         dimensions * sizeof(float),
                                         planes.View().Ptr(),
                                         planes.View().Pitch(),
                                         dimensions,
                                         count,
                                         cudaMemcpyDeviceToHost);
  CARACAL_CUDA_EXCEPTION_THROW_ON_ERORR(error);

  for (size_t y = 0; y < count; y++) {
    for (size_t x = 0; x < dimensions; x++) {
      const float value = host_planes[x + y * dimensions];
      assert(-1.0 <= value && value <= 1.0);
    }
  }
}