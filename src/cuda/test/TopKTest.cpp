#undef NDEBUG

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <vector>

#include "../DevicePointer.h"
#include "../TopK.h"

struct RandomTestCase {
  size_t count;
  size_t batches;
  size_t bits;
  size_t k;
};

void ExecuteRandomTestCase(const RandomTestCase &test_case) {
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

  caracal::PitchedDevicePointer<uint16_t> device_values =
      caracal::PitchedDevicePointer<uint16_t>::MemcpyPitch(
          values.data(), test_case.count, test_case.batches);

  caracal::PitchedDevicePointer<size_t> device_results =
      caracal::PitchedDevicePointer<size_t>::MallocPitch(test_case.k,
                                                         test_case.batches);

  caracal::TopK(device_results.View(),
                device_values.ConstView(),
                test_case.count,
                test_case.batches,
                test_case.k);

  std::vector<size_t> results(test_case.k * test_case.batches);
  cudaMemcpy2D(results.data(),
               test_case.k * sizeof(size_t),
               device_results.View().Ptr(),
               device_results.View().Pitch(),
               test_case.k * sizeof(size_t),
               test_case.batches,
               cudaMemcpyDeviceToHost);
  CARACAL_CUDA_EXCEPTION_THROW_ON_LAST_ERROR();

  for (size_t y = 0; y < test_case.batches; y++) {
    const size_t *result = results.data() + y * test_case.k;

    const uint16_t *batch_values = values.data() + y * test_case.count;

    std::vector<uint16_t> sorted_values(batch_values,
                                        batch_values + test_case.count);
    std::sort(sorted_values.begin(), sorted_values.end());

    for (size_t x = 0; x < std::min(test_case.count, test_case.k); x++) {
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