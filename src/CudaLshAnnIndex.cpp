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

#define CARACAL_CUDA_LSH_ANN_INDEX_SANITY_CHECKS

// FIXME: this is nonsense
template <typename T>
void check(T result,
           char const *const func,
           const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr,
            "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file,
            line,
            static_cast<unsigned int>(result),
            cudaGetErrorName(result),
            func);
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

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
        PitchedDevicePointer<uint16_t>::MallocPitch(count, this->count);

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
                           this->hashes.ConstView(),
                           this->count,
                           hashes.ConstView(),
                           count,
                           (hash_bits + 63) / 64 /* FIXME */);
    }

    // FIXME: only supports distances with up to 12 bits
    TopK(device_results.View(),
         distances.ConstView(),
         this->count,
         count,
         neighbors);
  }

  checkCudaErrors(cudaMemcpy2D(results,
                               neighbors * sizeof(size_t),
                               device_results.View().Ptr(),
                               device_results.View().Pitch(),
                               neighbors * sizeof(size_t),
                               count,
                               cudaMemcpyDeviceToHost));

#ifndef NDEBUG
  for (size_t y = 0; y < count; y++) {
    const size_t *result = results + y * neighbors;
    for (size_t x = 0; x < std::min(neighbors, this->count); x++) {
      assert(result[x] < this->count);
    }
  }
#endif
}

} // namespace caracal