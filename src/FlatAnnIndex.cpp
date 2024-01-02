#include "FlatAnnIndex.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <queue>

namespace caracal {

FlatAnnIndex::FlatAnnIndex(size_t dimensions,
                           size_t count,
                           const float *vectors)
    : dimensions(dimensions), count(count), vectors(vectors) {}

void FlatAnnIndex::Query(size_t *results,
                         size_t count,
                         const float *vectors,
                         size_t neighbors) const {
  std::priority_queue<std::pair<float, size_t>> candidate_results;

  for (size_t i = 0; i < count; i++) {
    const float *vector = vectors + i * dimensions;

    for (size_t j = 0; j < this->count; j++) {
      const float *database_vector = this->vectors + j * dimensions;
      const float distance = CosineDistance(vector, database_vector);
      candidate_results.emplace(distance, j);

      if (candidate_results.size() > neighbors) {
        candidate_results.pop();
      }
    }

    size_t *result = results + i * neighbors;
    size_t result_count = std::min(neighbors, this->count);

    for (size_t i = 0; i < result_count; i++) {
      result[result_count - 1 - i] = candidate_results.top().second;
      candidate_results.pop();
    }
    assert(candidate_results.empty());
  }
}

float FlatAnnIndex::CosineDistance(const float *left,
                                   const float *right) const {
  float dot_product = 0.0;
  float left_norm_squared = 0.0;
  float right_norm_squared = 0.0;

  for (size_t i = 0; i < dimensions; i++) {
    dot_product += left[i] * right[i];
    left_norm_squared += left[i] * left[i];
    right_norm_squared += right[i] * right[i];
  }

  return -dot_product / sqrt(left_norm_squared * right_norm_squared);
}

} // namespace caracal
