#ifndef CARACAL_CUDA_GENERATE_RANDOM_PLANES_H_
#define CARACAL_CUDA_GENERATE_RANDOM_PLANES_H_

#include <cstdint>

namespace caracal {

cudaError_t GenerateRandomPlanes(float **planes, size_t *pitch, size_t count,
                                 size_t dimensions, uint64_t seed);

}

#endif