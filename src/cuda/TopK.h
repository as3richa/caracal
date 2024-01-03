#ifndef CARACAL_CUDA_TOP_K_H_
#define CARACAL_CUDA_TOP_K_H_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "DevicePointer.h"

namespace caracal {

void TopK(PitchedView<size_t> results,
          ConstPitchedView<uint16_t> values,
          size_t count,
          size_t batches,
          size_t k);

}

#endif