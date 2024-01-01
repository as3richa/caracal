#include "CudaLshAnnIndex.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "cuda/GenerateRandomPlanes.h"
#include "cuda/Hashes.h"
#include "cuda/TopK.h"

// FIXME: this is nonsense
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

namespace caracal {

CudaLshAnnIndex::CudaLshAnnIndex(size_t dimensions, size_t count,
                                 const float *vectors, size_t hash_bits,
                                 uint64_t seed)
    : dimensions(dimensions), count(count), hash_bits(hash_bits) {
  // FIXME: this is nonsense
  if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    assert(false);
  }

  checkCudaErrors(GenerateRandomPlanes(&planes, &planes_pitch, hash_bits,
                                       dimensions, seed));

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
  float *device_vectors;
  size_t vectors_pitch;
  checkCudaErrors(cudaMallocPitch((void **)&device_vectors, &vectors_pitch,
                                  dimensions * sizeof(float), count));
  checkCudaErrors(cudaMemcpy2D(
      device_vectors, vectors_pitch, vectors, dimensions * sizeof(float),
      dimensions * sizeof(float), count, cudaMemcpyHostToDevice));

  uint64_t *hashes;
  size_t hashes_pitch;
  checkCudaErrors(ComputeHashes(&hashes, &hashes_pitch, cublas_handle,
                                device_vectors, count, vectors_pitch, planes,
                                hash_bits, planes_pitch, dimensions));

  cudaFree(device_vectors);

  uint16_t *distances;
  size_t distances_pitch;
  checkCudaErrors(ComputeHashDistances(
      &distances, &distances_pitch, hashes, count, hashes_pitch, this->hashes,
      this->count, this->hashes_pitch, (hash_bits + 63) / 64 /* FIXME */));

  size_t *device_results;
  size_t results_pitch;
  checkCudaErrors(TopK(&device_results, &results_pitch, distances, this->count,
                       count, distances_pitch, 12 /* FIXME: ??? */, neighbors));
  checkCudaErrors(cudaMemcpy2D(
      results, neighbors * sizeof(size_t), device_results, results_pitch,
      neighbors * sizeof(size_t), count, cudaMemcpyDeviceToHost));

  cudaFree(hashes);
  cudaFree(distances);
  cudaFree(device_results);
}

} // namespace caracal