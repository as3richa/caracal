#ifndef CARACAL_CUDA_HASHES_H_
#define CARACAL_CUDA_HASHES_H_

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace caracal {

cudaError_t ComputeHashes(uint64_t **hashes, size_t *hashes_pitch,
                          cublasHandle_t cublas_handle, float *vectors,
                          size_t vectors_count, size_t vectors_pitch,
                          float *planes, size_t planes_count,
                          size_t planes_pitch, size_t dimensions);

cudaError_t ComputeHashDistances(
    uint16_t **distances, size_t *distances_pitch, uint64_t *left_hashes,
    size_t left_hashes_count, size_t left_hashes_pitch, uint64_t *right_hashes,
    size_t right_hashes_count, size_t right_hashes_pitch, size_t hash_length);

} // namespace caracal

#endif