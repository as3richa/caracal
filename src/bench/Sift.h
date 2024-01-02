#ifndef CARACAL_BENCH_SIFT_H_
#define CARACAL_BENCH_SIFT_H_

#include <string>
#include <vector>

namespace caracal {

void ReadFvecs(std::vector<float> &data,
               size_t &count,
               size_t &dimensions,
               std::string path);
void WriteFvecs(const std::vector<float> &data,
                size_t count,
                size_t dimensions,
                std::string path);
void ComputeGroundTruth(std::string ground_truth_path,
                        std::string base_path,
                        std::string query_path);

} // namespace caracal

#endif