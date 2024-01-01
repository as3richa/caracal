#include "src/bench/Sift.h"

int main(void) {
  caracal::ComputeGroundTruth(
      "data/siftsmall/caracal_siftsmall_groundtruth.fvecs",
      "data/siftsmall/siftsmall_base.fvecs",
      "data/siftsmall/siftsmall_query.fvecs");
  caracal::ComputeGroundTruth("data/sift/caracal_sift_groundtruth.fvecs",
                              "data/sift/sift_base.fvecs",
                              "data/sift/sift_query.fvecs");
}