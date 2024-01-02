#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "CudaLshAnnIndex.h"

int main(void) {
  const size_t dimensions = 2;
  const size_t count = 16;
  const size_t hash_bits = 512;
  const uint64_t seed = 1337;

  std::vector<float> vectors(count * dimensions);
  for (size_t i = 0; i < count; i++) {
    const float theta = 2 * M_PI * i / count;
    const float x = cos(theta);
    const float y = -sin(theta);
    vectors[2 * i] = x;
    vectors[2 * i + 1] = y;
  }

  caracal::CudaLshAnnIndex index(2, 16, vectors.data(), hash_bits, seed);

  for (;;) {
    float vector[2];
    scanf("%f %f", &vector[0], &vector[1]);

    size_t result[3];
    index.Query(result, 1, vector, 3);

    for (size_t i = 0; i < 3; i++) {
      printf("%f, %f (%zu)\n",
             vectors[2 * result[i]],
             vectors[2 * result[i] + 1],
             result[i]);
    }
  }

  return 0;
}