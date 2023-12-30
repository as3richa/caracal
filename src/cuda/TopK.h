#ifndef CARACAL_CUDA_TOP_K_H_
#define CARACAL_CUDA_TOP_K_H_

#include <cstddef>
#include <cstdint>

namespace caracal {

cudaError_t TopK(size_t *result, const uint16_t *distances, size_t count,
                 size_t k);

}

#endif