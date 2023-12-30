#include <cassert>
#include <cstring>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../Hashes.h"

namespace caracal {

cudaError_t ComputeDots(float **dots, size_t *dots_pitch,
                        cublasHandle_t cublas_handle, float *vectors,
                        size_t vectors_count, size_t vectors_pitch,
                        float *planes, size_t planes_count, size_t planes_pitch,
                        size_t dimensions);

cudaError_t ConvertDotsToBits(uint64_t **bits, size_t *bits_pitch, float *dots,
                              size_t height, size_t width, size_t dots_pitch);

} // namespace caracal

void ComputeDotsTest(void) {
  cudaError_t error;

  const size_t dimensions = 3;

  const size_t vectors_count = 6;

  float *vectors;
  size_t vectors_pitch;
  error = cudaMallocPitch(&vectors, &vectors_pitch, dimensions * sizeof(float),
                          vectors_count);
  assert(error == cudaSuccess);

  const float host_vectors[vectors_count][dimensions] = {
      {1.0, 2.0, 3.0},    {4.0, 5.0, 6.0},    {7.0, 8.0, 9.0},
      {10.0, 11.0, 12.0}, {13.0, 14.0, 15.0}, {16.0, 17.0, 18.0}};
  error = cudaMemcpy2D(vectors, vectors_pitch, host_vectors,
                       dimensions * sizeof(float), dimensions * sizeof(float),
                       vectors_count, cudaMemcpyHostToDevice);
  assert(error == cudaSuccess);

  const size_t planes_count = 4;

  float *planes;
  size_t planes_pitch;
  error = cudaMallocPitch(&planes, &planes_pitch, dimensions * sizeof(float),
                          planes_count);
  assert(error == cudaSuccess);

  const float host_planes[planes_count][dimensions] = {{2.0, 3.0, 5.0},
                                                       {7.0, 11.0, 13.0},
                                                       {17.0, 19.0, 23.0},
                                                       {29.0, 31.0, 37.0}};
  error = cudaMemcpy2D(planes, planes_pitch, host_planes,
                       dimensions * sizeof(float), dimensions * sizeof(float),
                       planes_count, cudaMemcpyHostToDevice);
  assert(error == cudaSuccess);

  cublasHandle_t cublas_handle;
  assert(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);

  float *dots;
  size_t dots_pitch;
  error = caracal::ComputeDots(&dots, &dots_pitch, cublas_handle, vectors,
                               vectors_count, vectors_pitch, planes,
                               planes_count, planes_pitch, dimensions);
  assert(error == cudaSuccess);

  assert(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

  cudaFree(vectors);
  cudaFree(planes);

  float host_dots[vectors_count][planes_count];
  cudaMemcpy2D(host_dots, planes_count * sizeof(float), dots, dots_pitch,
               planes_count * sizeof(float), vectors_count,
               cudaMemcpyDeviceToHost);
  cudaFree(dots);

  const float expectation[vectors_count][planes_count] = {
      {23.0, 68.0, 124.0, 202.0},    {53.0, 161.0, 301.0, 493.0},
      {83.0, 254.0, 478.0, 784.0},   {113.0, 347.0, 655.0, 1075.0},
      {143.0, 440.0, 832.0, 1366.0}, {173.0, 533.0, 1009.0, 1657.0},
  };
  assert(memcmp(host_dots, expectation, sizeof(expectation)) == 0);
}

void ConvertDotsToBitsTest(void) {
  cudaError_t error;

  const size_t width = 100;
  const size_t height = 10;

  float host_dots[height][width];
  uint64_t expectation[height][(width + 64 - 1) / 64] = {};
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      host_dots[y][x] = -(float)(x + y + 1);
      if ((x + y) % 3 == 0) {
        host_dots[y][x] *= -1.0;
        expectation[y][x / 64] |= ((uint64_t)1) << (x % 64);
      }
    }
  }

  float *dots;
  size_t dots_pitch;
  error = cudaMallocPitch(&dots, &dots_pitch, width * sizeof(float), height);
  assert(error == cudaSuccess);

  error = cudaMemcpy2D(dots, dots_pitch, host_dots, width * sizeof(float),
                       width * sizeof(float), height, cudaMemcpyHostToDevice);
  assert(error == cudaSuccess);

  uint64_t *bits;
  size_t bits_pitch;
  caracal::ConvertDotsToBits(&bits, &bits_pitch, dots, height, width,
                             dots_pitch);
  cudaFree(dots);

  uint64_t host_bits[height][(width + 64 - 1) / 64];
  cudaMemcpy2D(host_bits, sizeof(host_bits[0]), bits, bits_pitch,
               sizeof(host_bits[0]), height, cudaMemcpyDeviceToHost);
  cudaFree(bits);

  assert(memcmp(host_bits, expectation, sizeof(expectation)) == 0);
}

void ComputeHashesTest(void) {
  cudaError_t error;

  const size_t dimensions = 3;

  const size_t vectors_count = 6;

  float *vectors;
  size_t vectors_pitch;
  error = cudaMallocPitch(&vectors, &vectors_pitch, dimensions * sizeof(float),
                          vectors_count);
  assert(error == cudaSuccess);

  const float host_vectors[vectors_count][dimensions] = {
      {-1.0, 2.0, -3.0},   {-4.0, 5.0, 6.0},   {7.0, 8.0, -9.0},
      {10.0, -11.0, 12.0}, {13.0, 14.0, 15.0}, {-16.0, -17.0, 18.0}};
  error = cudaMemcpy2D(vectors, vectors_pitch, host_vectors,
                       dimensions * sizeof(float), dimensions * sizeof(float),
                       vectors_count, cudaMemcpyHostToDevice);
  assert(error == cudaSuccess);

  const size_t planes_count = 2 * dimensions;

  float *planes;
  size_t planes_pitch;
  error = cudaMallocPitch(&planes, &planes_pitch, dimensions * sizeof(float),
                          planes_count);
  assert(error == cudaSuccess);

  const float host_planes[planes_count][dimensions] = {
      {1.0, 0.0, 0.0},  {0.0, 1.0, 0.0},  {0.0, 0.0, 1.0},
      {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0},
  };
  error = cudaMemcpy2D(planes, planes_pitch, host_planes,
                       dimensions * sizeof(float), dimensions * sizeof(float),
                       planes_count, cudaMemcpyHostToDevice);
  assert(error == cudaSuccess);

  cublasHandle_t cublas_handle;
  assert(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS);

  uint64_t *hashes;
  size_t hashes_pitch;
  error = caracal::ComputeHashes(&hashes, &hashes_pitch, cublas_handle, vectors,
                                 vectors_count, vectors_pitch, planes,
                                 planes_count, planes_pitch, dimensions);
  assert(error == cudaSuccess);

  assert(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);
  cudaFree(vectors);
  cudaFree(planes);

  uint64_t host_hashes[vectors_count][1];
  cudaMemcpy2D(host_hashes, sizeof(host_hashes[0]), hashes, hashes_pitch,
               sizeof(host_hashes[0]), vectors_count, cudaMemcpyDeviceToHost);
  cudaFree(hashes);

  const uint64_t expectation[vectors_count][1] = {
      {0b101010}, {0b001110}, {0b100011}, {0b010101}, {0b000111}, {0b011100},
  };
  assert(memcmp(host_hashes, expectation, sizeof(expectation)) == 0);
}

void ComputeHashDistancesTest(void) {
  /*
cudaError_t ComputeHashDistances(
  uint16_t **distances, size_t *distances_pitch, uint64_t *left_hashes,
  size_t left_hashes_count, size_t left_hashes_pitch, uint64_t *right_hashes,
  size_t right_hashes_count, size_t right_hashes_pitch, size_t hash_length);
  */
}

int main(void) {
  ComputeDotsTest();
  ConvertDotsToBitsTest();
  ComputeHashesTest();
  return 0;
}