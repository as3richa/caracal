#ifndef CARACAL_CUDA_GENERATE_RANDOM_PLANES_H_
#define CARACAL_CUDA_GENERATE_RANDOM_PLANES_H_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "DevicePointer.h"

namespace caracal {

void GenerateRandomPlanes(PitchedView<float> planes,
                          size_t count,
                          size_t dimensions,
                          uint64_t seed);

}

#endif