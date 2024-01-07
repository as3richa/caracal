#undef NDEBUG

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "../CudaLshAnnIndex.h"
#include "Sift.h"

using namespace caracal;

struct Recall {
  double at1, at5, at10, at100, at1000;
};

Recall EvaluateRecall(size_t hash_bits,
                      const float *database_vectors,
                      size_t database_vectors_count,
                      const float *query_vectors,
                      const float *ground_truth_vectors,
                      size_t query_count,
                      size_t dimensions) {
  const size_t neighbors = 1000;
  assert(database_vectors_count >= neighbors);

  CudaLshAnnIndex index(
      dimensions, database_vectors_count, database_vectors, hash_bits, 31337);

  std::vector<size_t> results(query_count * neighbors);
  index.Query(results.data(), query_count, query_vectors, neighbors);

  Recall recall{};

  for (size_t i = 0; i < query_count; i++) {
    const size_t *result = results.data() + i * neighbors;
    const float *ground_truth_vector = ground_truth_vectors + i * dimensions;

    for (size_t j = 0; j < neighbors; j++) {
      const float *database_vector = database_vectors + result[j] * dimensions;

      const int cmp = memcmp(
          database_vector, ground_truth_vector, dimensions * sizeof(float));
      if (cmp == 0) {
        recall.at1 += j < 1;
        recall.at5 += j < 5;
        recall.at10 += j < 10;
        recall.at100 += j < 100;
        recall.at1000 += j < 1000;
        break;
      }
    }
  }

  recall.at1 /= query_count;
  recall.at5 /= query_count;
  recall.at10 /= query_count;
  recall.at100 /= query_count;
  recall.at1000 /= query_count;

  return recall;
}

std::vector<size_t> BuildHashBitsSchedule() {
  std::vector<size_t> schedule;

  for (size_t i = 1; i < 16; i++) {
    schedule.push_back(i);
  }

  for (size_t i = 16; i < 128; i += 4) {
    schedule.push_back(i);
  }

  for (size_t i = 128; i < 1024; i += 32) {
    schedule.push_back(i);
  }

  for (size_t i = 1024; i < 4096; i += 128) {
    schedule.push_back(i);
  }

  return schedule;
}

void EmitCsvHeader() { puts("dataset,hash_bits,at1,at5,at10,at100,at1000"); }

void EmitCsvRow(const char *dataset, size_t hash_bits, const Recall &recall) {
  printf("%s,%zu,%f,%f,%f,%f,%f\n",
         dataset,
         hash_bits,
         recall.at1,
         recall.at5,
         recall.at10,
         recall.at100,
         recall.at1000);
}

int main(void) {
  EmitCsvHeader();

  const std::vector<size_t> schedule = BuildHashBitsSchedule();

  for (const SiftDatasetConfig &config : sift_dataset_configs) {
    SiftDataset dataset(config);

    for (const size_t hash_bits : schedule) {
      fprintf(stderr,
              "Evaluating index recall with %zu hash bits on dataset %s\n",
              hash_bits,
              config.name);

      const Recall recall = EvaluateRecall(hash_bits,
                                           dataset.database_vectors.data(),
                                           dataset.database_vectors_count,
                                           dataset.query_vectors.data(),
                                           dataset.ground_truth_vectors.data(),
                                           dataset.query_count,
                                           dataset.dimensions);
      EmitCsvRow(config.name, hash_bits, recall);
    }
  }
}