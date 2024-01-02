#ifndef CARACAL_LSH_ANN_INDEX_H_
#define CARACAL_LSH_ANN_INDEX_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace caracal {

class LshAnnIndex {
public:
  LshAnnIndex(size_t dimensions,
              size_t count,
              const float *vectors,
              size_t hash_bits,
              size_t seed);

  void Query(size_t *results,
             size_t count,
             const float *vectors,
             size_t neighbors) const;

private:
  size_t dimensions;
  size_t count;
  size_t hash_bits;
  std::vector<float> planes;
  std::vector<std::byte> hashes;

  float *Plane(size_t index);
  const float *Plane(size_t index) const;

  std::byte *Hash(size_t index);
  const std::byte *Hash(size_t index) const;

  size_t HashBytes(void) const;

  void ComputeHash(std::byte *hash, const float *vector) const;
  bool ComputeHashBit(const float *plane, const float *vector) const;
  size_t ComputeHashDistance(const std::byte *left,
                             const std::byte *right) const;
};

} // namespace caracal

#endif