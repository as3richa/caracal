#include "LshAnnIndex.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <queue>
#include <random>
#include <vector>

namespace caracal {

template <typename R>
void GenerateRandomPlane(float *plane, size_t dimensions, R &rng) {
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);

  float norm_squared = 0.0;

  for (size_t i = 0; i < dimensions; i++) {
    float component = distribution(rng);
    norm_squared += component * component;
    plane[i] = component;
  }

  float norm = sqrt(norm_squared);

  for (size_t i = 0; i < dimensions; i++) {
    plane[i] /= norm;
  }
}

LshAnnIndex::LshAnnIndex(size_t dimensions, size_t count, const float *vectors,
                         size_t hash_bits, size_t seed)
    : dimensions(dimensions), count(count), hash_bits(hash_bits),
      planes(dimensions * hash_bits), hashes(HashBytes() * count) {
  std::default_random_engine rng(seed);
  for (size_t i = 0; i < hash_bits; i++) {
    GenerateRandomPlane(Plane(i), dimensions, rng);
  }

  for (size_t i = 0; i < count; i++) {
    const float *vector = vectors + i * dimensions;
    ComputeHash(Hash(i), vector);
  }
}

void LshAnnIndex::Query(size_t *results, size_t count, const float *vectors,
                        size_t neighbors) const {
  std::vector<std::byte> hash(HashBytes());
  std::priority_queue<std::pair<size_t, size_t>> candidate_results;

  for (size_t i = 0; i < count; i++) {
    const float *vector = vectors + i * dimensions;
    ComputeHash(hash.data(), vector);

    for (size_t j = 0; j < this->count; j++) {
      assert(candidate_results.size() <= neighbors);

      const size_t distance = ComputeHashDistance(hash.data(), Hash(j));
      candidate_results.push(std::make_pair(distance, j));

      if (candidate_results.size() > neighbors) {
        candidate_results.pop();
      }
    }

    size_t *result = results + i * dimensions;
    size_t result_count = std::min(neighbors, this->count);

    for (size_t i = 0; i < result_count; i++) {
      result[result_count - 1 - i] = candidate_results.top().second;
      candidate_results.pop();
    }
    assert(candidate_results.empty());
  }
}

float *LshAnnIndex::Plane(size_t index) {
  return planes.data() + index * dimensions;
}

const float *LshAnnIndex::Plane(size_t index) const {
  return planes.data() + index * dimensions;
}

std::byte *LshAnnIndex::Hash(size_t index) {
  return hashes.data() + index * HashBytes();
}

const std::byte *LshAnnIndex::Hash(size_t index) const {
  return hashes.data() + index * HashBytes();
}

size_t LshAnnIndex::HashBytes(void) const { return (hash_bits + 8 - 1) / 8; }

void LshAnnIndex::ComputeHash(std::byte *hash, const float *vector) const {
  for (size_t i = 0; i < HashBytes(); i++) {
    std::byte hash_byte{0};

    for (size_t j = 0; j < 8; j++) {
      const size_t index = 8 * i + j;
      if (index >= hash_bits) {
        break;
      }

      hash_byte |= static_cast<std::byte>(ComputeHashBit(Plane(index), vector))
                   << j;
    }

    hash[i] = hash_byte;
  }
}

bool LshAnnIndex::ComputeHashBit(const float *plane,
                                 const float *vector) const {
  float dot = 0.0;

  for (size_t i = 0; i < dimensions; i++) {
    dot += plane[i] * vector[i];
  }

  return dot >= 0.0;
}

size_t LshAnnIndex::ComputeHashDistance(const std::byte *left,
                                        const std::byte *right) const {
  size_t distance = 0;

  size_t i;

  for (i = 0; i + 8 < HashBytes(); i += 8) {
    distance +=
        std::popcount(*(uint64_t *)(left + i) ^ *(uint64_t *)(right + i));
  }

  for (; i < HashBytes(); i++) {
    distance += std::popcount(std::to_integer<uint8_t>(left[i] ^ right[i]));
  }

  return distance;
}

} // namespace caracal