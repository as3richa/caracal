#ifndef CARACAL_CUDA_LSH_ANN_INDEX_H_
#define CARACAL_CUDA_LSH_ANN_INDEX_H_

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>

namespace caracal {

class CudaLshAnnIndex {
public:
  CudaLshAnnIndex(size_t dimensions, size_t count, const float *vectors,
                  size_t hash_bits, size_t seed);

  ~CudaLshAnnIndex();

  void Query(size_t *results, size_t count, const float *vectors,
             size_t neighbors) const;

private:
  size_t dimensions;
  size_t count;
  size_t hash_bits;
  cublasHandle_t cublas_handle;
  float *planes;
  size_t planes_pitch;
  uint64_t *hashes;
  size_t hashes_pitch;
};

} // namespace caracal

#endif