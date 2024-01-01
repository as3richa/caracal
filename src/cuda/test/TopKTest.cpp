#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <vector>

#include "../TopK.h"

#include <chrono>
#include <iostream>
struct RandomTestCase {
  size_t count;
  size_t batches;
  size_t bits;
  size_t k;
};

void ExecuteRandomTestCase(const RandomTestCase &test_case) {
  cudaError_t error;

  assert(test_case.bits <= 12); // FIXME: ???
  assert(test_case.k <= test_case.count);

  std::default_random_engine rng(1337);
  std::uniform_int_distribution<uint16_t> distribution(
      0, (1 << test_case.bits) - 1);

  std::vector<uint16_t> values;

  for (size_t y = 0; y < test_case.batches; y++) {
    for (size_t x = 0; x < test_case.count; x++) {
      values.push_back(distribution(rng));
    }
  }

  uint16_t *device_values;
  size_t values_pitch;
  error =
      cudaMallocPitch(&device_values, &values_pitch,
                      test_case.count * sizeof(uint16_t), test_case.batches);
  assert(error == cudaSuccess);
  error = cudaMemcpy2D(device_values, values_pitch, values.data(),
                       test_case.count * sizeof(uint16_t),
                       test_case.count * sizeof(uint16_t), test_case.batches,
                       cudaMemcpyHostToDevice);
  assert(error == cudaSuccess);

  size_t *device_results;
  size_t results_pitch;
  std::chrono::high_resolution_clock::time_point p =
      std::chrono::high_resolution_clock::now();
  error = caracal::TopK(&device_results, &results_pitch, device_values,
                        test_case.count, test_case.batches, values_pitch,
                        test_case.bits, test_case.k);
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - p)
                   .count()
            << "\n";
  assert(error == cudaSuccess);

  std::vector<size_t> results(test_case.k * test_case.batches);
  error =
      cudaMemcpy2D(results.data(), test_case.k * sizeof(size_t), device_results,
                   results_pitch, test_case.k * sizeof(size_t),
                   test_case.batches, cudaMemcpyDeviceToHost);
  assert(error == cudaSuccess);
  cudaFree(device_results);

  for (size_t y = 0; y < test_case.batches; y++) {
    // printf("! %zu %zu %zu %zu %zu\n", y, test_case.count,
    // test_case.batches,test_case.bits, test_case.k);

    const size_t *result = results.data() + y * test_case.k;

    const uint16_t *batch_values = values.data() + y * test_case.count;

    std::vector<uint16_t> sorted_values(batch_values,
                                        batch_values + test_case.count);
    std::sort(sorted_values.begin(), sorted_values.end());

    for (size_t x = 0; x < std::min(test_case.count, test_case.k); x++) {
      if (!(batch_values[result[x]] == sorted_values[x])) {
        for (size_t x = 0; x < std::min(test_case.count, test_case.k); x++) {
          printf("== %zu %zu %zu %zu %zu\n", x, y, result[x],
                 batch_values[result[x]], sorted_values[x]);
        }
        break;
      }
      assert(result[x] < test_case.count);
      assert(batch_values[result[x]] == sorted_values[x]);
    }
  }
}

int main(void) {
  std::vector<RandomTestCase> test_cases{

      {100, 10, 8, 10},
      {1025, 100, 10, 300},
      {1000, 1000, 10, 300},
      {10000, 1000, 4, 100},
      /*{100, 10, 12, 10},*/ {10000, 10000, 10, 1000}
      //, // ,
  };

  for (const RandomTestCase &test_case : test_cases) {
    ExecuteRandomTestCase(test_case);
  }

  return 0;
}