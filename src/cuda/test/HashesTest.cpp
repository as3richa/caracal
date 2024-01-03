#undef NDEBUG

#include <bit>
#include <cassert>
#include <cstring>
#include <random>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../DevicePointer.h"
#include "../Hashes.h"

namespace caracal {

void ComputeDots(PitchedView<float> dots,
                 cublasHandle_t cublas_handle,
                 ConstPitchedView<float> vectors,
                 size_t vectors_count,
                 ConstPitchedView<float> planes,
                 size_t planes_count,
                 size_t dimensions);

void ConvertDotsToBits(PitchedView<uint64_t> bits,
                       ConstPitchedView<float> dots,
                       size_t width,
                       size_t height);

} // namespace caracal

void ComputeDotsTest(void) {
  const size_t dimensions = 3;

  const size_t vectors_count = 6;
  const float host_vectors[vectors_count][dimensions] = {{1.0, 2.0, 3.0},
                                                         {4.0, 5.0, 6.0},
                                                         {7.0, 8.0, 9.0},
                                                         {10.0, 11.0, 12.0},
                                                         {13.0, 14.0, 15.0},
                                                         {16.0, 17.0, 18.0}};
  caracal::PitchedDevicePointer<float> vectors =
      caracal::PitchedDevicePointer<float>::MemcpyPitch(
          &host_vectors[0][0], dimensions, vectors_count);

  const size_t planes_count = 4;
  const float host_planes[planes_count][dimensions] = {{2.0, 3.0, 5.0},
                                                       {7.0, 11.0, 13.0},
                                                       {17.0, 19.0, 23.0},
                                                       {29.0, 31.0, 37.0}};
  caracal::PitchedDevicePointer<float> planes =
      caracal::PitchedDevicePointer<float>::MemcpyPitch(
          &host_planes[0][0], dimensions, planes_count);

  cublasHandle_t cublas_handle;
  if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    // FIXME: ???
    CARACAL_CUDA_EXCEPTION_THROW_ON_ERROR(cudaErrorUnknown);
  }

  caracal::PitchedDevicePointer<float> dots =
      caracal::PitchedDevicePointer<float>::MallocPitch(planes_count,
                                                        vectors_count);
  caracal::ComputeDots(dots.View(),
                       cublas_handle,
                       vectors.ConstView(),
                       vectors_count,
                       planes.ConstView(),
                       planes_count,
                       dimensions);

  assert(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

  float host_dots[vectors_count][planes_count];
  cudaMemcpy2D(host_dots,
               planes_count * sizeof(float),
               dots.View().Ptr(),
               dots.View().Pitch(),
               planes_count * sizeof(float),
               vectors_count,
               cudaMemcpyDeviceToHost);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR(error);

  const float expectation[vectors_count][planes_count] = {
      {23.0, 68.0, 124.0, 202.0},
      {53.0, 161.0, 301.0, 493.0},
      {83.0, 254.0, 478.0, 784.0},
      {113.0, 347.0, 655.0, 1075.0},
      {143.0, 440.0, 832.0, 1366.0},
      {173.0, 533.0, 1009.0, 1657.0},
  };
  assert(memcmp(host_dots, expectation, sizeof(expectation)) == 0);
}

void ConvertDotsToBitsTest(void) {
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

  caracal::PitchedDevicePointer<float> dots =
      caracal::PitchedDevicePointer<float>::MemcpyPitch(
          &host_dots[0][0], width, height);

  // FIXME
  caracal::PitchedDevicePointer<uint64_t> bits =
      caracal::PitchedDevicePointer<uint64_t>::MallocPitch((width + 63) / 64,
                                                           height);
  caracal::ConvertDotsToBits(bits.View(), dots.ConstView(), width, height);

  uint64_t host_bits[height][(width + 64 - 1) / 64];
  cudaMemcpy2D(host_bits,
               sizeof(host_bits[0]),
               bits.View().Ptr(),
               bits.View().Pitch(),
               sizeof(host_bits[0]),
               height,
               cudaMemcpyDeviceToHost);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();

  assert(memcmp(host_bits, expectation, sizeof(expectation)) == 0);
}

void ComputeHashesTest(void) {
  const size_t dimensions = 3;

  const size_t vectors_count = 6;
  const float host_vectors[vectors_count][dimensions] = {{-1.0, 2.0, -3.0},
                                                         {-4.0, 5.0, 6.0},
                                                         {7.0, 8.0, -9.0},
                                                         {10.0, -11.0, 12.0},
                                                         {13.0, 14.0, 15.0},
                                                         {-16.0, -17.0, 18.0}};
  caracal::PitchedDevicePointer<float> vectors =
      caracal::PitchedDevicePointer<float>::MemcpyPitch(
          &host_vectors[0][0], dimensions, vectors_count);

  const size_t planes_count = 2 * dimensions;
  const float host_planes[planes_count][dimensions] = {
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {-1.0, 0.0, 0.0},
      {0.0, -1.0, 0.0},
      {0.0, 0.0, -1.0},
  };
  caracal::PitchedDevicePointer<float> planes =
      caracal::PitchedDevicePointer<float>::MemcpyPitch(
          &host_planes[0][0], dimensions, planes_count);

  cublasHandle_t cublas_handle;
  if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
    // FIXME: ???
    CARACAL_CUDA_EXCEPTION_THROW_ON_ERROR(cudaErrorUnknown);
  }

  // FIXME
  caracal::PitchedDevicePointer<uint64_t> hashes =
      caracal::PitchedDevicePointer<uint64_t>::MallocPitch(
          (planes_count + 63) / 64, vectors_count);

  caracal::ComputeHashes(hashes.View(),
                         cublas_handle,
                         vectors.ConstView(),
                         vectors_count,
                         planes.ConstView(),
                         planes_count,
                         dimensions);

  assert(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS);

  uint64_t host_hashes[vectors_count][1];
  cudaMemcpy2D(host_hashes,
               sizeof(host_hashes[0]),
               hashes.View().Ptr(),
               hashes.View().Pitch(),
               sizeof(host_hashes[0]),
               vectors_count,
               cudaMemcpyDeviceToHost);

  const uint64_t expectation[vectors_count][1] = {
      {0b101010},
      {0b001110},
      {0b100011},
      {0b010101},
      {0b000111},
      {0b011100},
  };
  assert(memcmp(host_hashes, expectation, sizeof(expectation)) == 0);
}

void ComputeHashDistancesTest(void) {
  const size_t hash_length = 10;

  std::default_random_engine rng(31337);
  std::uniform_int_distribution<uint64_t> distribution;

  for (size_t i = 0; i < 100; i++) {
    const size_t left_hashes_count = 1337;
    uint64_t host_left_hashes[left_hashes_count][hash_length];

    for (size_t y = 0; y < left_hashes_count; y++) {
      for (size_t x = 0; x < hash_length; x++) {
        host_left_hashes[y][x] = distribution(rng);
      }
    }

    caracal::PitchedDevicePointer<uint64_t> left_hashes =
        caracal::PitchedDevicePointer<uint64_t>::MemcpyPitch(
            &host_left_hashes[0][0], hash_length, left_hashes_count);

    const size_t right_hashes_count = 100;
    uint64_t host_right_hashes[right_hashes_count][hash_length];

    for (size_t y = 0; y < right_hashes_count; y++) {
      for (size_t x = 0; x < hash_length; x++) {
        host_right_hashes[y][x] = distribution(rng);
      }
    }

    caracal::PitchedDevicePointer<uint64_t> right_hashes =
        caracal::PitchedDevicePointer<uint64_t>::MemcpyPitch(
            &host_right_hashes[0][0], hash_length, right_hashes_count);

    uint16_t expectation[left_hashes_count][right_hashes_count];

    for (size_t y = 0; y < left_hashes_count; y++) {
      for (size_t x = 0; x < right_hashes_count; x++) {
        expectation[y][x] = 0;

        for (size_t i = 0; i < hash_length; i++) {
          expectation[y][x] +=
              std::popcount(host_left_hashes[y][i] ^ host_right_hashes[x][i]);
        }
      }
    }

    caracal::PitchedDevicePointer<uint16_t> distances =
        caracal::PitchedDevicePointer<uint16_t>::MallocPitch(right_hashes_count,
                                                             left_hashes_count);

    caracal::ComputeHashDistances(distances.View(),
                                  left_hashes.ConstView(),
                                  left_hashes_count,
                                  right_hashes.ConstView(),
                                  right_hashes_count,
                                  hash_length);

    uint16_t host_distances[left_hashes_count][right_hashes_count];
    cudaMemcpy2D(host_distances,
                 right_hashes_count * sizeof(uint16_t),
                 distances.View().Ptr(),
                 distances.View().Pitch(),
                 right_hashes_count * sizeof(uint16_t),
                 left_hashes_count,
                 cudaMemcpyDeviceToHost);
    CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();

    assert(memcmp(host_distances, expectation, sizeof(expectation)) == 0);
  }
}

int main(void) {
  ComputeDotsTest();
  ConvertDotsToBitsTest();
  ComputeHashesTest();
  ComputeHashDistancesTest();
  return 0;
}