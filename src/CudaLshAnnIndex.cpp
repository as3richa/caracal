#include "CudaLshAnnIndex.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "cuda/DevicePointer.h"
#include "cuda/GenerateRandomPlanes.h"
#include "cuda/Hashes.h"
#include "cuda/TopK.h"

namespace caracal {

CudaLshAnnIndex::CudaLshAnnIndex(size_t dimensions,
                                 size_t count,
                                 const float *vectors,
                                 size_t hash_bits,
                                 uint64_t seed)
    : dimensions(dimensions), count(count), hash_bits(hash_bits),
      planes(PitchedDevicePointer<float>::MallocPitch(dimensions, hash_bits)),
      hashes(PitchedDevicePointer<uint64_t>::MallocPitch((hash_bits + 63) / 64,
                                                         count /* FIXME */)) {
  // FIXME: this is nonsense
  if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    assert(false);
  }

  GenerateRandomPlanes(planes.View(), hash_bits, dimensions, seed);

  PitchedDevicePointer<float> device_vectors =
      PitchedDevicePointer<float>::MemcpyPitch(vectors, dimensions, count);

  ComputeHashes(hashes.View(),
                cublas_handle,
                device_vectors.ConstView(),
                count,
                planes.ConstView(),
                hash_bits,
                dimensions);
}

CudaLshAnnIndex::~CudaLshAnnIndex() {
  // FIXME: this is nonsense
  if (cublasDestroy(cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    assert(false);
  }
}

void CudaLshAnnIndex::Query(size_t *results,
                            size_t count,
                            const float *vectors,
                            size_t neighbors) const {
  PitchedDevicePointer<size_t> device_results =
      PitchedDevicePointer<size_t>::MallocPitch(neighbors, count);

  {
    PitchedDevicePointer<uint16_t> distances =
        PitchedDevicePointer<uint16_t>::MallocPitch(this->count, count);

    {
      // FIXME
      PitchedDevicePointer<uint64_t> hashes =
          PitchedDevicePointer<uint64_t>::MallocPitch((hash_bits + 63) / 64,
                                                      count);

      {
        PitchedDevicePointer<float> device_vectors =
            PitchedDevicePointer<float>::MemcpyPitch(
                vectors, dimensions, count);

        ComputeHashes(hashes.View(),
                      cublas_handle,
                      device_vectors.ConstView(),
                      count,
                      planes.ConstView(),
                      hash_bits,
                      dimensions);
      }

      ComputeHashDistances(distances.View(),
                           hashes.ConstView(),
                           count,
                           this->hashes.ConstView(),
                           this->count,
                           (hash_bits + 63) / 64 /* FIXME */);
    }

    // FIXME: only supports distances with up to 12 bits
    TopK(device_results.View(),
         distances.ConstView(),
         this->count,
         count,
         neighbors);
  }

  cudaMemcpy2D(results,
               neighbors * sizeof(size_t),
               device_results.View().Ptr(),
               device_results.View().Pitch(),
               neighbors * sizeof(size_t),
               count,
               cudaMemcpyDeviceToHost);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();

#ifndef NDEBUG
  for (size_t y = 0; y < count; y++) {
    for (size_t x = 0; x < std::min(neighbors, this->count); x++) {
      assert(results[x + y * neighbors] < this->count);
    }
  }
#endif
}

} // namespace caracal