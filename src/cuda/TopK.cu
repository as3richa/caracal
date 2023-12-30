#include "TopK.h"

#include <cuda.h>

#include <algorithm>
#include <vector>

#include <cstdio>
namespace caracal {

// FIXME
cudaError_t TopK(size_t *result, const uint16_t *distances, size_t count,
                 size_t k) {
  printf("%d %d\n", count, k);
  std::vector<uint16_t> host_distances(count);
  cudaMemcpy(host_distances.data(), distances, count * sizeof(uint16_t),
             cudaMemcpyDeviceToHost);

  std::vector<std::pair<uint16_t, size_t>> distances_with_indices;
  for (const uint16_t distance : host_distances) {
    printf("! %zu %zu\n", distance, distances_with_indices.size());
    distances_with_indices.push_back(
        std::make_pair(distance, distances_with_indices.size()));
  }

  std::nth_element(distances_with_indices.begin(),
                   distances_with_indices.begin() + std::min(k, count),
                   distances_with_indices.end());

  std::sort(distances_with_indices.begin(),
            distances_with_indices.begin() + std::min(k, count));

  for (size_t i = 0; i < std::min(k, count); i++) {
    result[i] = distances_with_indices[i].second;
  }

  return cudaSuccess;
}

} // namespace caracal