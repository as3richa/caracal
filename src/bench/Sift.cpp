#undef NDEBUG

#include "Sift.h"

#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../CudaLshAnnIndex.h"
#include "../FlatAnnIndex.h"
#include "../LshAnnIndex.h"

namespace caracal {

SiftDataset::SiftDataset(const SiftDatasetConfig &config) {
  ReadFvecs(database_vectors,
            database_vectors_count,
            dimensions,
            config.database_path);

  size_t query_dimensions;
  ReadFvecs(query_vectors, query_count, query_dimensions, config.query_path);
  assert(query_dimensions == dimensions);

  size_t ground_truth_vectors_count;
  size_t ground_truth_dimensions;
  ReadFvecs(ground_truth_vectors,
            ground_truth_vectors_count,
            ground_truth_dimensions,
            config.ground_truth_path);
  assert(ground_truth_vectors_count == query_count &&
         ground_truth_dimensions == dimensions);
}

void ReadFvecs(std::vector<float> &data,
               size_t &count,
               size_t &dimensions,
               const char *path) {
  static_assert(std::endian::native == std::endian::little);

  data.clear();
  count = 0;

  FILE *file = fopen(path, "rb");
  assert(file != nullptr);

  bool read_dimensions = false;

  for (;;) {
    uint32_t dimensions_u32;
    size_t read = fread(&dimensions_u32, 4, 1, file);
    assert(read <= 1);

    if (read == 0) {
      assert(feof(file));
      break;
    }

    if (read_dimensions) {
      assert(dimensions_u32 == dimensions);
    } else {
      dimensions = dimensions_u32;
    }

    const size_t offset = data.size();
    data.resize(offset + dimensions);
    assert(fread(data.data() + offset, 4, dimensions, file) == dimensions);

    count++;
  }

  fclose(file);
}

void WriteFvecs(const std::vector<float> &data,
                size_t count,
                size_t dimensions,
                const char *path) {
  static_assert(std::endian::native == std::endian::little);

  FILE *file = fopen(path, "wb");
  assert(file != nullptr);

  uint32_t dimensions_u32 = dimensions;

  for (size_t i = 0; i < count; i++) {
    assert(fwrite(&dimensions_u32, 4, 1, file) == 1);
    assert(fwrite(data.data() + i * dimensions, 4, dimensions, file) ==
           dimensions);
  }

  fclose(file);
}

void ComputeGroundTruth(const char *ground_truth_path,
                        const char *base_path,
                        const char *query_path) {
  printf("Reading base vectors from %s\n", base_path);
  std::vector<float> base_data;
  size_t base_count;
  size_t dimensions;
  ReadFvecs(base_data, base_count, dimensions, base_path);

  printf("Building index\n");
  caracal::FlatAnnIndex index(dimensions, base_count, base_data.data());

  printf("Reading query vectors from %s\n", query_path);
  std::vector<float> query_data;
  size_t query_dimensions;
  size_t query_count;
  ReadFvecs(query_data, query_count, query_dimensions, query_path);

  assert(query_dimensions == dimensions);

  printf("Querying index\n");
  std::vector<size_t> query_results(query_count);
  index.Query(query_results.data(), query_count, query_data.data(), 1);

  printf("Writing output ground truth vectors to %s\n", ground_truth_path);
  std::vector<float> ground_truth_data;
  for (const size_t index : query_results) {
    ground_truth_data.insert(ground_truth_data.end(),
                             base_data.data() + index * dimensions,
                             base_data.data() + (index + 1) * dimensions);
  }
  WriteFvecs(ground_truth_data, query_count, dimensions, ground_truth_path);
}

} // namespace caracal