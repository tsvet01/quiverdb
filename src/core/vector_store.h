#pragma once
#include "distance.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace quiverdb {

/**
 * @brief Distance metric type for vector search
 */
enum class DistanceMetric {
  L2,      ///< Squared L2 (Euclidean) distance
  COSINE,  ///< Cosine distance
  DOT      ///< Negative dot product (for maximum inner product search)
};

/**
 * @brief Search result containing vector ID and distance
 */
struct SearchResult {
  uint64_t id;      ///< Vector ID
  float distance;   ///< Distance to query vector

  bool operator<(const SearchResult& other) const {
    return distance < other.distance;
  }
};

/**
 * @brief In-memory vector database with k-NN search
 *
 * Stores vectors with associated IDs and supports efficient k-nearest neighbor
 * search using various distance metrics.
 *
 * Example usage:
 * @code
 * VectorStore store(768, DistanceMetric::COSINE);
 *
 * float vec1[768] = {...};
 * float vec2[768] = {...};
 * store.add(1, vec1);
 * store.add(2, vec2);
 *
 * float query[768] = {...};
 * auto results = store.search(query, 5); // Find 5 nearest neighbors
 * @endcode
 */
class VectorStore {
public:
  /**
   * @brief Constructs a new vector store
   *
   * @param dimension Dimension of vectors to store
   * @param metric Distance metric to use for search
   * @throws std::invalid_argument if dimension is 0
   */
  explicit VectorStore(size_t dimension, DistanceMetric metric = DistanceMetric::L2)
      : dim_(dimension), metric_(metric) {
    if (dimension == 0) {
      throw std::invalid_argument("Dimension must be greater than 0");
    }
  }

  /**
   * @brief Adds a vector to the store
   *
   * @param id Unique identifier for the vector
   * @param vector Pointer to vector data (must have dim elements)
   * @throws std::invalid_argument if vector is null or ID already exists
   */
  void add(uint64_t id, const float* vector) {
    if (vector == nullptr) {
      throw std::invalid_argument("Vector must not be null");
    }

    if (id_to_index_.find(id) != id_to_index_.end()) {
      throw std::invalid_argument("Vector with ID " + std::to_string(id) + " already exists");
    }

    // Allocate and copy vector data
    size_t index = vectors_.size();
    vectors_.push_back(std::vector<float>(vector, vector + dim_));
    ids_.push_back(id);
    id_to_index_[id] = index;
  }

  /**
   * @brief Removes a vector from the store
   *
   * @param id ID of vector to remove
   * @return true if vector was removed, false if not found
   */
  bool remove(uint64_t id) {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
      return false;
    }

    size_t index = it->second;
    size_t last_index = vectors_.size() - 1;

    // Swap with last element and pop
    if (index != last_index) {
      vectors_[index] = std::move(vectors_[last_index]);
      ids_[index] = ids_[last_index];
      id_to_index_[ids_[index]] = index;
    }

    vectors_.pop_back();
    ids_.pop_back();
    id_to_index_.erase(it);

    return true;
  }

  /**
   * @brief Gets a vector by ID
   *
   * @param id Vector ID
   * @return Pointer to vector data, or nullptr if not found
   */
  const float* get(uint64_t id) const {
    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
      return nullptr;
    }
    return vectors_[it->second].data();
  }

  /**
   * @brief Searches for k nearest neighbors
   *
   * @param query Query vector (must have dim elements)
   * @param k Number of nearest neighbors to return
   * @return Vector of search results, sorted by distance (closest first)
   * @throws std::invalid_argument if query is null or k is 0
   */
  std::vector<SearchResult> search(const float* query, size_t k) const {
    if (query == nullptr) {
      throw std::invalid_argument("Query vector must not be null");
    }
    if (k == 0) {
      throw std::invalid_argument("k must be greater than 0");
    }

    // Compute all distances
    std::vector<SearchResult> results;
    results.reserve(vectors_.size());

    for (size_t i = 0; i < vectors_.size(); ++i) {
      float dist = compute_distance(query, vectors_[i].data());
      results.push_back({ids_[i], dist});
    }

    // Partial sort to get k smallest distances
    size_t num_results = std::min(k, results.size());
    std::partial_sort(results.begin(),
                     results.begin() + num_results,
                     results.end());

    // Return only k results
    results.resize(num_results);
    return results;
  }

  /**
   * @brief Returns the number of vectors in the store
   */
  size_t size() const {
    return vectors_.size();
  }

  /**
   * @brief Returns the dimension of stored vectors
   */
  size_t dimension() const {
    return dim_;
  }

  /**
   * @brief Returns the distance metric being used
   */
  DistanceMetric metric() const {
    return metric_;
  }

  /**
   * @brief Clears all vectors from the store
   */
  void clear() {
    vectors_.clear();
    ids_.clear();
    id_to_index_.clear();
  }

  /**
   * @brief Checks if a vector with given ID exists
   */
  bool contains(uint64_t id) const {
    return id_to_index_.find(id) != id_to_index_.end();
  }

  /**
   * @brief Reserves space for a given number of vectors
   *
   * Pre-allocates memory to avoid reallocations during insertion.
   * Useful when you know the approximate number of vectors in advance.
   *
   * @param capacity Number of vectors to reserve space for
   */
  void reserve(size_t capacity) {
    vectors_.reserve(capacity);
    ids_.reserve(capacity);
    id_to_index_.reserve(capacity);
  }

  /**
   * @brief Updates an existing vector
   *
   * @param id ID of vector to update
   * @param vector New vector data (must have dim elements)
   * @return true if vector was updated, false if ID not found
   * @throws std::invalid_argument if vector is null
   */
  bool update(uint64_t id, const float* vector) {
    if (vector == nullptr) {
      throw std::invalid_argument("Vector must not be null");
    }

    auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) {
      return false;
    }

    size_t index = it->second;
    std::copy(vector, vector + dim_, vectors_[index].begin());
    return true;
  }

private:
  float compute_distance(const float* a, const float* b) const {
    switch (metric_) {
      case DistanceMetric::L2:
        return l2_sq(a, b, dim_);
      case DistanceMetric::COSINE:
        return cosine_distance(a, b, dim_);
      case DistanceMetric::DOT:
        // For maximum inner product search, we negate the dot product
        // so that larger dot products have smaller "distances"
        return -dot_product(a, b, dim_);
      default:
        return std::numeric_limits<float>::infinity();
    }
  }

  size_t dim_;
  DistanceMetric metric_;
  std::vector<std::vector<float>> vectors_;
  std::vector<uint64_t> ids_;
  std::unordered_map<uint64_t, size_t> id_to_index_;
};

} // namespace quiverdb
