#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
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
};

template <typename I>
Evaluation evaluate(const I &index,
                    float *base_vectors,
                    size_t base_count,
                    float *vectors,
                    size_t count,
                    size_t dimensions,
                    float *ground_truth) {
  // FIXME: assert index.count() >= 1000
  const size_t k = 1000;
  assert(base_count >= k);

  std::vector<size_t> results(count * k);
  index.Query(results.data(), count, vectors, k);

  Evaluation evaluation{};

  for (size_t i = 0; i < count; i++) {
    const size_t *result = results.data() + i * k;
    const float *ground_truth_vector = ground_truth + i * dimensions;

    size_t j;
    for (j = 0; j < k; j++) {
      const float *base_vector = base_vectors + result[j] * dimensions;
      if (memcmp(base_vector,
                 ground_truth_vector,
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
  }

  evaluation.recallAt1 /= count;
  evaluation.recallAt5 /= count;
  evaluation.recallAt10 /= count;
  evaluation.recallAt100 /= count;
  evaluation.recallAt1000 /= count;

  return evaluation;
}

int main(void) {
  const std::string base_path = "data/sift/sift_base.fvecs";
  const std::string query_path = "data/sift/sift_query.fvecs";
  const std::string ground_truth_path =
      "data/sift/caracal_sift_groundtruth.fvecs";

  printf("Reading base vectors from %s\n", base_path.c_str());
  std::vector<float> base_data;
  size_t base_count;
  size_t dimensions;
  caracal::ReadFvecs(base_data, base_count, dimensions, base_path);

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
  caracal::ReadFvecs(ground_truth_data,
                     ground_truth_count,
                     ground_truth_dimensions,
                     ground_truth_path);

  assert(ground_truth_dimensions == dimensions);

  printf("Evaluating index\n");

      caracal::LshAnnIndex index_warmup(
       dimensions, base_count, base_data.data(), 1024, 31337);

    auto build_start_time = std::chrono::high_resolution_clock::now();
    caracal::LshAnnIndex index(
       dimensions, base_count, base_data.data(), 1024, 31337);
    auto build_end_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::microseconds>(build_end_time - build_start_time);

    printf("Build: %zuus\n", build_duration.count());

    std::vector<size_t> results(query_count * 1000);
    index.Query(results.data(), query_count, query_data.data(), 1000);
    auto query_start_time = std::chrono::high_resolution_clock::now();
    index.Query(results.data(), 10, query_data.data(), 1000);
    auto query_end_time = std::chrono::high_resolution_clock::now();
    auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end_time - query_start_time);

    printf("Query: %zuus\n", query_duration.count());

  puts("hash_bits,recall1,recall5,recall10,recall100,recall1000");

  for (size_t hash_bits = 1; hash_bits <= 4096;) {
    caracal::CudaLshAnnIndex index(
       dimensions, base_count, base_data.data(), hash_bits, 31337);
    //caracal::LshAnnIndex index(
    //    dimensions, base_count, base_data.data(), hash_bits, 31337);

    const Evaluation evaluation = evaluate(index,
                                           base_data.data(),
                                           base_count,
                                           query_data.data(),
                                           query_count,
                                           dimensions,
                                           ground_truth_data.data());

    printf("%zu,%.4f,%.4f,%.4f,%.4f,%.4f\n",
           hash_bits,
           evaluation.recallAt1,
           evaluation.recallAt5,
           evaluation.recallAt10,
           evaluation.recallAt100,
           evaluation.recallAt1000);

    if (hash_bits < 16) {
      hash_bits++;
    } else if (hash_bits < 128) {
      hash_bits += 16;
    } else {
      hash_bits += 128;
    }
  }
}