// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once
#include "distance.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace quiverdb {

enum class DistanceMetric { L2, COSINE, DOT };

struct SearchResult {
  uint64_t id;
  float distance;
  bool operator<(const SearchResult& o) const { return distance < o.distance; }
};

class VectorStore {
public:
  explicit VectorStore(size_t dimension, DistanceMetric metric = DistanceMetric::L2)
      : dim_(dimension), metric_(metric) {
    if (dimension == 0) throw std::invalid_argument("Dimension must be > 0");
  }

  void add(uint64_t id, const float* vector) {
    if (!vector) throw std::invalid_argument("Vector must not be null");
    std::unique_lock lock(mutex_);
    if (id_to_index_.count(id)) throw std::invalid_argument("ID " + std::to_string(id) + " exists");
    vectors_data_.insert(vectors_data_.end(), vector, vector + dim_);
    ids_.push_back(id);
    id_to_index_[id] = ids_.size() - 1;
  }

  bool remove(uint64_t id) {
    std::unique_lock lock(mutex_);
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return false;
    size_t idx = it->second, last = ids_.size() - 1;
    if (idx != last) {
      std::copy_n(vectors_data_.data() + last * dim_, dim_, vectors_data_.data() + idx * dim_);
      ids_[idx] = ids_[last];
      id_to_index_[ids_[idx]] = idx;
    }
    vectors_data_.resize(vectors_data_.size() - dim_);
    ids_.pop_back();
    id_to_index_.erase(it);
    return true;
  }

  // WARNING: Returned pointer invalidated by any write operation
  const float* get(uint64_t id) const {
    std::shared_lock lock(mutex_);
    auto it = id_to_index_.find(id);
    return it == id_to_index_.end() ? nullptr : vectors_data_.data() + it->second * dim_;
  }

  // Thread-safe: returns a copy of the vector (safe for concurrent access)
  std::vector<float> get_copy(uint64_t id) const {
    std::shared_lock lock(mutex_);
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return {};
    const float* ptr = vectors_data_.data() + it->second * dim_;
    return std::vector<float>(ptr, ptr + dim_);
  }

  std::vector<SearchResult> search(const float* query, size_t k) const {
    if (!query) throw std::invalid_argument("Query must not be null");
    if (k == 0) throw std::invalid_argument("k must be > 0");
    std::shared_lock lock(mutex_);
    std::vector<SearchResult> results;
    results.reserve(ids_.size());
    for (size_t i = 0; i < ids_.size(); ++i)
      results.push_back({ids_[i], compute_distance(query, vectors_data_.data() + i * dim_)});
    size_t n = std::min(k, results.size());
    std::partial_sort(results.begin(), results.begin() + n, results.end());
    results.resize(n);
    return results;
  }

  size_t size() const { std::shared_lock lock(mutex_); return ids_.size(); }
  size_t dimension() const { return dim_; }
  DistanceMetric metric() const { return metric_; }

  void clear() {
    std::unique_lock lock(mutex_);
    vectors_data_.clear();
    ids_.clear();
    id_to_index_.clear();
  }

  bool contains(uint64_t id) const {
    std::shared_lock lock(mutex_);
    return id_to_index_.count(id);
  }

  void reserve(size_t capacity) {
    std::unique_lock lock(mutex_);
    vectors_data_.reserve(capacity * dim_);
    ids_.reserve(capacity);
    id_to_index_.reserve(capacity);
  }

  bool update(uint64_t id, const float* vector) {
    if (!vector) throw std::invalid_argument("Vector must not be null");
    std::unique_lock lock(mutex_);
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return false;
    std::copy_n(vector, dim_, vectors_data_.data() + it->second * dim_);
    return true;
  }

private:
  float compute_distance(const float* a, const float* b) const {
    switch (metric_) {
      case DistanceMetric::L2: return l2_sq(a, b, dim_);
      case DistanceMetric::COSINE: return cosine_distance(a, b, dim_);
      case DistanceMetric::DOT: return -dot_product(a, b, dim_);
      default: return std::numeric_limits<float>::infinity();
    }
  }

  size_t dim_;
  DistanceMetric metric_;
  std::vector<float> vectors_data_;
  std::vector<uint64_t> ids_;
  std::unordered_map<uint64_t, size_t> id_to_index_;
  mutable std::shared_mutex mutex_;
};

} // namespace quiverdb
