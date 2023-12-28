#ifndef CARACAL_FLAT_ANN_INDEX_H_
#define CARACAL_FLAT_ANN_INDEX_H_

#include <cstddef>

namespace caracal {

class FlatAnnIndex {
public:
  FlatAnnIndex(size_t dimensions, size_t count, const float *vectors);

  void Query(size_t *results, size_t count, const float *vectors,
             size_t neighbors) const;

private:
  size_t dimensions;
  size_t count;
  const float *vectors;

  float CosineDistance(const float *left, const float *right) const;
};

} // namespace caracal

#endif