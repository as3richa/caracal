#include "CudaLshAnnIndex.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

// FIXME: this is nonsense
#include <cassert>
#include <cstdio>
#include <cstdlib>
#define CURAND_CALL(x)                                                         \
  do {                                                                         \
    if ((x) != CURAND_STATUS_SUCCESS) {                                        \
      printf("Error at %s:%d\n", __FILE__, __LINE__);                          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

cudaError_t ComputeHashes(uint64_t **hashes, size_t *hashes_pitch,
                          cublasHandle_t cublas_handle, float *vectors,
                          size_t vectors_count, size_t vectors_pitch,
                          float *planes, size_t planes_count,
                          size_t planes_pitch, size_t dimensions);

cudaError_t GenerateRandomPlanes(float **planes, size_t *pitch,
                                 cublasHandle_t cublas_handle, size_t count,
                                 size_t dimensions, uint64_t seed);

cudaError_t ComputeHashDistances(
    uint16_t **distances, size_t *distances_pitch, uint64_t *left_hashes,
    size_t left_hashes_count, size_t left_hashes_pitch, uint64_t *right_hashes,
    size_t right_hashes_count, size_t right_hashes_pitch, size_t hash_length);

namespace caracal {

CudaLshAnnIndex::CudaLshAnnIndex(size_t dimensions, size_t count,
                                 const float *vectors, size_t hash_bits,
                                 uint64_t seed)
    : dimensions(dimensions), count(count), hash_bits(hash_bits) {
  // FIXME: this is nonsense
  if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    assert(false);
  }

  checkCudaErrors(GenerateRandomPlanes(&planes, &planes_pitch, cublas_handle,
                                       hash_bits, dimensions, seed));

  {
    float *device_vectors;
    size_t vectors_pitch;
    checkCudaErrors(cudaMallocPitch((void **)&device_vectors, &vectors_pitch,
                                    dimensions * sizeof(float), count));
    cudaMemcpy2D(device_vectors, vectors_pitch, vectors,
                 dimensions * sizeof(float), dimensions * sizeof(float), count,
                 cudaMemcpyHostToDevice);

    checkCudaErrors(ComputeHashes(&hashes, &hashes_pitch, cublas_handle,
                                  device_vectors, count, vectors_pitch, planes,
                                  hash_bits, planes_pitch, dimensions));

    cudaFree(device_vectors);
  }
}

CudaLshAnnIndex::~CudaLshAnnIndex() {
  // FIXME: this is nonsense
  if (cublasDestroy(cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    assert(false);
  }
  checkCudaErrors(cudaFree(planes));
  checkCudaErrors(cudaFree(hashes));
}

void CudaLshAnnIndex::Query(size_t *results, size_t count, const float *vectors,
                            size_t neighbors) const {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  float *device_vectors;
  size_t vectors_pitch;
  checkCudaErrors(cudaMallocPitch((void **)&device_vectors, &vectors_pitch,
                                  dimensions * sizeof(float), count));
  cudaMemcpy2D(device_vectors, vectors_pitch, vectors,
               dimensions * sizeof(float), dimensions * sizeof(float), count,
               cudaMemcpyHostToDevice);

  uint64_t *hashes;
  size_t hashes_pitch;
  checkCudaErrors(ComputeHashes(&hashes, &hashes_pitch, cublas_handle,
                                device_vectors, count, vectors_pitch, planes,
                                hash_bits, planes_pitch, dimensions));

  uint16_t *distances;
  size_t distances_pitch;
  checkCudaErrors(ComputeHashDistances(
      &distances, &distances_pitch, this->hashes, this->count,
      this->hashes_pitch, hashes, count, hashes_pitch, (hash_bits + 63) / 64));

  // checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
  // matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, &alpha, d_B,
  // matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWB));
}

} // namespace caracal