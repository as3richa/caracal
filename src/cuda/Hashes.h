#ifndef CARACAL_CUDA_HASHES_H_
#define CARACAL_CUDA_HASHES_H_

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace caracal {

void ComputeHashes(PitchedView<uint64_t> hashes,
                   cublasHandle_t cublas_handle,
                   ConstPitchedView<float> vectors,
                   size_t vectors_count,
                   ConstPitchedView<float> planes,
                   size_t planes_count,
                   size_t dimensions);

void ComputeHashDistances(PitchedView<uint16_t> distances,
                          ConstPitchedView<uint64_t> left_hashes,
                          size_t left_hashes_count,
                          ConstPitchedView<uint64_t> right_hashes,
                          size_t right_hashes_count,
                          size_t hash_length);

} // namespace caracal

#endif