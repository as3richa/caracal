#include <cassert>
#include <cstdio>
#include <cstddef>
#include <cstring>

#include "src/CudaLshAnnIndex.h"
#include "src/LshAnnIndex.h"
#include "src/bench/Sift.h"

struct Evaluation {
  double recallAt1;
  double recallAt5;
  double recallAt10;
  double recallAt100;
  double recallAt1000;
  double recallAt10000;
};

template <typename I>
Evaluation evaluate(const I &index, float *base_vectors, size_t base_count, float *vectors,
         size_t count, size_t dimensions, float *ground_truth) {
  // FIXME: assert index.count() >= 1000
  std::vector<size_t> results(count * 1000);
  index.Query(results.data(), count, vectors, 1000);

  Evaluation evaluation{};

  for (size_t i = 0; i < count; i++) {
    const size_t *result = results.data() + i * 1000;
    const float *ground_truth_vector = ground_truth + i * dimensions;

    size_t j;
    for (j = 0; j < 1000; j++) {
      const float *base_vector = base_vectors + result[j] * dimensions;
      if (memcmp(base_vector, ground_truth_vector,
                 dimensions * sizeof(float)) == 0) {
        break;
      }
    }

    if (j < 1) {
      evaluation.recallAt1 += 1;
    }
    if (j < 5) {
      evaluation.recallAt5 += 1;
    }
    if (j < 10) {
      evaluation.recallAt10 += 1;
    }
    if (j < 100) {
      evaluation.recallAt100 += 1;
    }
    if (j < 1000) {
      evaluation.recallAt1000 += 1;
    }
    if (j < 10000) {
      evaluation.recallAt10000 += 1;
    }
  }

  evaluation.recallAt1 /= count;
  evaluation.recallAt5 /= count;
  evaluation.recallAt10 /= count;
  evaluation.recallAt100 /= count;
  evaluation.recallAt1000 /= count;
  evaluation.recallAt10000 /= count;

  return evaluation;
}

int main(void) {
  const std::string base_path = "data/siftsmall/siftsmall_base.fvecs";
  const std::string query_path = "data/siftsmall/siftsmall_query.fvecs";
  const std::string ground_truth_path =
      "data/siftsmall/caracal_siftsmall_groundtruth.fvecs";

  size_t hash_bits;
  scanf("%zu", &hash_bits);

  printf("Reading base vectors from %s\n", base_path.c_str());
  std::vector<float> base_data;
  size_t base_count;
  size_t dimensions;
  caracal::ReadFvecs(base_data, base_count, dimensions, base_path);

  printf("Building index\n");
  //caracal::CudaLshAnnIndex index(dimensions, base_count, base_data.data(),
  //                               hash_bits, 1337);
caracal::LshAnnIndex index(dimensions, base_count, base_data.data(),
                                 hash_bits, 1337);

  printf("Reading query vectors from %s\n", query_path.c_str());
  std::vector<float> query_data;
  size_t query_dimensions;
  size_t query_count;
  caracal::ReadFvecs(query_data, query_count, query_dimensions, query_path);

  assert(query_dimensions == dimensions);

  printf("Reading ground truth vectors from %s\n", ground_truth_path.c_str());
  std::vector<float> ground_truth_data;
  size_t ground_truth_dimensions;
  size_t ground_truth_count;
  caracal::ReadFvecs(ground_truth_data, ground_truth_count, ground_truth_dimensions,
            ground_truth_path);

  assert(ground_truth_dimensions == dimensions);

  printf("Evaluating index\n");
  const Evaluation evaluation =
      evaluate(index, base_data.data(), base_count, query_data.data(),
               query_count, dimensions, ground_truth_data.data());

  printf("recalll@1: %.4f\nrecalll@5: %.4f\nrecalll@10: %.4f\nrecalll@100: "
         "%.4f\nrecalll@1000: %.4f\nrecalll@10000: %.4f\n",
         evaluation.recallAt1, evaluation.recallAt5, evaluation.recallAt10,
         evaluation.recallAt100, evaluation.recallAt1000,evaluation.recallAt10000);
}