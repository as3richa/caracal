#ifndef CARACAL_BENCH_SIFT_H_
#define CARACAL_BENCH_SIFT_H_

#include <string>
#include <vector>

namespace caracal {

struct SiftDatasetConfig {
  const char *name;
  const char *database_path;
  const char *query_path;
  const char *ground_truth_path;
};

static const std::vector<SiftDatasetConfig> sift_dataset_configs{
    {"siftsmall",
     "data/siftsmall/siftsmall_base.fvecs",
     "data/siftsmall/siftsmall_query.fvecs",
     "data/siftsmall/caracal_siftsmall_groundtruth.fvecs"},
    {"sift",
     "data/sift/sift_base.fvecs",
     "data/sift/sift_query.fvecs",
     "data/sift/caracal_sift_groundtruth.fvecs"}};

struct SiftDataset {
  std::vector<float> database_vectors;
  size_t database_vectors_count;

  std::vector<float> query_vectors;
  std::vector<float> ground_truth_vectors;
  size_t query_count;

  size_t dimensions;

  SiftDataset(const SiftDatasetConfig &config);
};

void ReadFvecs(std::vector<float> &data,
               size_t &count,
               size_t &dimensions,
               const char *path);
void WriteFvecs(const std::vector<float> &data,
                size_t count,
                size_t dimensions,
                const char *path);

void ComputeGroundTruth(std::string ground_truth_path,
                        std::string base_path,
                        std::string query_path);

} // namespace caracal

#endif