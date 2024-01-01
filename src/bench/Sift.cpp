#include "Sift.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "../FlatAnnIndex.h"
#include "../LshAnnIndex.h"
#include "../CudaLshAnnIndex.h"

#undef NDEBUG
#include <cassert>

namespace caracal {

void ReadFvecs(std::vector<float> &data, size_t &count, size_t &dimensions,
               std::string path) {
  static_assert(std::endian::native == std::endian::little);

  data.clear();
  count = 0;

  FILE *file = fopen(path.c_str(), "rb");
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

void WriteFvecs(std::string path, const std::vector<float> &data, size_t count,
                size_t dimensions) {
  static_assert(std::endian::native == std::endian::little);

  FILE *file = fopen(path.c_str(), "wb");
  assert(file != nullptr);

  uint32_t dimensions_u32 = dimensions;

  for (size_t i = 0; i < count; i++) {
    assert(fwrite(&dimensions_u32, 4, 1, file) == 1);
    assert(fwrite(data.data() + i * dimensions, 4, dimensions, file) ==
           dimensions);
  }

  fclose(file);
}

void ComputeGroundTruth(std::string ground_truth_path, std::string base_path,
                        std::string query_path) {

  printf("Reading base vectors from %s\n", base_path.c_str());
  std::vector<float> base_data;
  size_t base_count;
  size_t dimensions;
  ReadFvecs(base_data, base_count, dimensions, base_path);

  printf("Building index\n");
  caracal::FlatAnnIndex index(dimensions, base_count, base_data.data());

  printf("Reading query vectors from %s\n", query_path.c_str());
  std::vector<float> query_data;
  size_t query_dimensions;
  size_t query_count;
  ReadFvecs(query_data, query_count, query_dimensions, query_path);

  assert(query_dimensions == dimensions);

  printf("Querying index\n");
  std::vector<size_t> query_results(query_count);
  index.Query(query_results.data(), query_count, query_data.data(), 1);

  printf("Writing output ground truth vectors to %s\n",
         ground_truth_path.c_str());
  std::vector<float> ground_truth_data;
  for (const size_t index : query_results) {
    ground_truth_data.insert(ground_truth_data.end(),
                             base_data.data() + index * dimensions,
                             base_data.data() + (index + 1) * dimensions);
  }
  /*for (size_t i = 0 ; i < 10; i++) {
    printf("==== %.3f\n", index.CosineDistance(&ground_truth_data[i*dimensions], &query_data[i*dimensions]));
    for (size_t j = 0; j < dimensions; j++) {
        printf("%.3f %.3f\n", ground_truth_data[i*dimensions +j], query_data[i*dimensions +j]);
    }
    puts("");
  }*/
  WriteFvecs(ground_truth_path, ground_truth_data, query_count, dimensions);
}

} // namespace caracal