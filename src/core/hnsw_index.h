#pragma once
#include "distance.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream> // Added for file operations
#include <filesystem> // Added for file operations
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace quiverdb {

// Utility functions for binary serialization
namespace detail {
template <typename T>
void write_binary(std::ofstream& ofs, const T& value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void read_binary(std::ifstream& ifs, T& value) {
    ifs.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
void write_vector(std::ofstream& ofs, const std::vector<T>& vec) {
    write_binary(ofs, vec.size());
    if (!vec.empty()) {
        ofs.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
    }
}

template <typename T>
void read_vector(std::ifstream& ifs, std::vector<T>& vec) {
    size_t size;
    read_binary(ifs, size);
    
    // Sanity check: Arbitrary limit of 100GB for a single vector to prevent OOM on bad data
    // Realistically, this depends on T, but 10 billion elements is a safe upper bound for sanity
    if (size > 10000000000ULL) { 
        throw std::runtime_error("File corrupted: Vector size too large (" + std::to_string(size) + ")");
    }

    vec.resize(size);
    if (!vec.empty()) {
        ifs.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
    }
}
} // namespace detail

/**
 * @brief Distance metric type for HNSW index
 */
enum class HNSWDistanceMetric {
  L2,      ///< Squared L2 (Euclidean) distance
  COSINE,  ///< Cosine distance
  DOT      ///< Negative dot product (for maximum inner product search)
};

/**
 * @brief Search result containing vector ID and distance
 */
struct HNSWSearchResult {
  uint64_t id;      ///< External vector ID
  float distance;   ///< Distance to query vector

  bool operator<(const HNSWSearchResult& other) const {
    return distance < other.distance;
  }

  bool operator>(const HNSWSearchResult& other) const {
    return distance > other.distance;
  }
};

/**
 * @brief HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search
 *
 * Implements the HNSW algorithm as described in:
 * "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
 * by Malkov and Yashunin (2018)
 *
 * Key parameters:
 * - M: Maximum number of connections per node (default: 16)
 * - ef_construction: Size of dynamic candidate list during construction (default: 200)
 * - ef_search: Size of dynamic candidate list during search (default: 50)
 *
 * @note Thread-safety model:
 * - Multiple threads can call search() concurrently
 * - Write operations (add) acquire exclusive locks
 *
 * Example usage:
 * @code
 * HNSWIndex index(768, HNSWDistanceMetric::COSINE, 100000);
 *
 * float vec[768] = {...};
 * index.add(1, vec);
 *
 * float query[768] = {...};
 * auto results = index.search(query, 10);  // Find 10 nearest neighbors
 * @endcode
 */
class HNSWIndex {
public:
  /**
   * @brief Constructs a new HNSW index
   *
   * @param dimension Dimension of vectors to store
   * @param metric Distance metric to use
   * @param max_elements Maximum number of elements (must be known upfront)
   * @param M Number of connections per node (higher = better recall, more memory)
   * @param ef_construction Construction-time search parameter (higher = better index quality)
   * @param random_seed Seed for random level generation
   * @throws std::invalid_argument if dimension is 0 or max_elements is 0
   */
  explicit HNSWIndex(
      size_t dimension,
      HNSWDistanceMetric metric = HNSWDistanceMetric::L2,
      size_t max_elements = 100000,
      size_t M = 16,
      size_t ef_construction = 200,
      uint32_t random_seed = 42)
      : dim_(dimension),
        metric_(metric),
        max_elements_(max_elements),
        M_(M),
        M_max_(M),
        M_max0_(M * 2),  // Level 0 has 2x connections
        ef_construction_(std::max(ef_construction, M)),
        ef_search_(50),
        mult_(1.0 / std::log(static_cast<double>(M))),
        level_generator_(random_seed),
        enterpoint_(-1),
        max_level_(-1) {

    if (dimension == 0) {
      throw std::invalid_argument("Dimension must be greater than 0");
    }
    if (max_elements == 0) {
      throw std::invalid_argument("max_elements must be greater than 0");
    }

    // Pre-allocate storage
    vectors_.resize(max_elements * dim_);
    external_ids_.resize(max_elements);
    levels_.resize(max_elements, 0);
    neighbors_.resize(max_elements);

    // Per-element locks for concurrent modifications
    // Using unique_ptr since mutex is not movable
    node_locks_.reserve(max_elements);
    for (size_t i = 0; i < max_elements; ++i) {
      node_locks_.push_back(std::make_unique<std::shared_mutex>());
    }
  }

  /**
   * @brief Adds a vector to the index
   *
   * @param id External unique identifier for the vector
   * @param vector Pointer to vector data (must have dim elements)
   * @throws std::invalid_argument if vector is null or ID already exists
   * @throws std::runtime_error if index is full
   * @note Thread-safe: acquires exclusive lock
   */
  void add(uint64_t id, const float* vector) {
    if (vector == nullptr) {
      throw std::invalid_argument("Vector must not be null");
    }

    std::unique_lock<std::shared_mutex> global_lock(global_mutex_);

    // Check for duplicate ID
    if (id_to_internal_.find(id) != id_to_internal_.end()) {
      throw std::invalid_argument("Vector with ID " + std::to_string(id) + " already exists");
    }

    if (cur_element_count_ >= max_elements_) {
      throw std::runtime_error("Index is full");
    }

    // Allocate internal ID
    size_t internal_id = cur_element_count_++;
    id_to_internal_[id] = internal_id;
    external_ids_[internal_id] = id;

    // Copy vector data
    std::copy(vector, vector + dim_, vectors_.begin() + internal_id * dim_);

    // Determine level for this node
    int level = get_random_level();
    levels_[internal_id] = level;

    // Initialize neighbor lists for all levels
    neighbors_[internal_id].resize(level + 1);
    for (int l = 0; l <= level; ++l) {
      size_t max_conn = (l == 0) ? M_max0_ : M_max_;
      neighbors_[internal_id][l].reserve(max_conn);
    }

    // First element - just set as entry point
    if (enterpoint_ == static_cast<size_t>(-1)) {
      enterpoint_ = internal_id;
      max_level_ = level;
      return;
    }

    // Search for entry point from top to the level above the new node's level
    size_t curr_obj = enterpoint_;

    if (level < max_level_) {
      float curr_dist = compute_distance(vector, get_vector(curr_obj));

      for (int l = max_level_; l > level; --l) {
        bool changed = true;
        while (changed) {
          changed = false;
          std::shared_lock<std::shared_mutex> lock(*node_locks_[curr_obj]);
          const auto& neighbors = neighbors_[curr_obj][l];

          for (size_t neighbor_id : neighbors) {
            float dist = compute_distance(vector, get_vector(neighbor_id));
            if (dist < curr_dist) {
              curr_dist = dist;
              curr_obj = neighbor_id;
              changed = true;
            }
          }
        }
      }
    }

    // Insert at each level from min(level, max_level_) down to 0
    for (int l = std::min(level, max_level_); l >= 0; --l) {
      // Search for ef_construction nearest neighbors at this level
      auto top_candidates = search_layer(vector, curr_obj, ef_construction_, l);

      // Select M best neighbors using the heuristic
      auto selected = select_neighbors(vector, top_candidates, M_, l);

      // Connect the new node to selected neighbors
      {
        std::unique_lock<std::shared_mutex> lock(*node_locks_[internal_id]);
        neighbors_[internal_id][l] = std::move(selected);
      }

      // Add reverse connections
      size_t max_conn = (l == 0) ? M_max0_ : M_max_;
      for (size_t neighbor_id : neighbors_[internal_id][l]) {
        std::unique_lock<std::shared_mutex> lock(*node_locks_[neighbor_id]);

        auto& neighbor_connections = neighbors_[neighbor_id][l];

        if (neighbor_connections.size() < max_conn) {
          neighbor_connections.push_back(internal_id);
        } else {
          // Need to prune - find the worst connection and potentially replace
          float dist_to_new = compute_distance(get_vector(neighbor_id), vector);

          // Build candidate list with all existing + new
          std::vector<std::pair<float, size_t>> candidates;
          candidates.reserve(neighbor_connections.size() + 1);

          for (size_t conn : neighbor_connections) {
            float d = compute_distance(get_vector(neighbor_id), get_vector(conn));
            candidates.emplace_back(d, conn);
          }
          candidates.emplace_back(dist_to_new, internal_id);

          // Sort by distance
          std::sort(candidates.begin(), candidates.end());

          // Keep only max_conn
          neighbor_connections.clear();
          for (size_t i = 0; i < max_conn && i < candidates.size(); ++i) {
            neighbor_connections.push_back(candidates[i].second);
          }
        }
      }

      // Update entry point for next level
      if (!top_candidates.empty()) {
        curr_obj = top_candidates.top().second;
      }
    }

    // Update global entry point if new node has higher level
    if (level > max_level_) {
      enterpoint_ = internal_id;
      max_level_ = level;
    }
  }

  /**
   * @brief Searches for k nearest neighbors
   *
   * @param query Query vector (must have dim elements)
   * @param k Number of nearest neighbors to return
   * @return Vector of search results, sorted by distance (closest first)
   * @throws std::invalid_argument if query is null or k is 0
   * @note Thread-safe: acquires shared lock, multiple searches can run concurrently
   */
  std::vector<HNSWSearchResult> search(const float* query, size_t k) const {
    if (query == nullptr) {
      throw std::invalid_argument("Query vector must not be null");
    }
    if (k == 0) {
      throw std::invalid_argument("k must be greater than 0");
    }

    std::shared_lock<std::shared_mutex> global_lock(global_mutex_);

    if (cur_element_count_ == 0) {
      return {};
    }

    // Start from entry point
    size_t curr_obj = enterpoint_;
    float curr_dist = compute_distance(query, get_vector(curr_obj));

    // Traverse from top level down to level 1
    for (int l = max_level_; l > 0; --l) {
      bool changed = true;
      while (changed) {
        changed = false;
        std::shared_lock<std::shared_mutex> lock(*node_locks_[curr_obj]);

        if (static_cast<int>(neighbors_[curr_obj].size()) <= l) continue;

        const auto& neighbors = neighbors_[curr_obj][l];
        for (size_t neighbor_id : neighbors) {
          float dist = compute_distance(query, get_vector(neighbor_id));
          if (dist < curr_dist) {
            curr_dist = dist;
            curr_obj = neighbor_id;
            changed = true;
          }
        }
      }
    }

    // Search at level 0 with ef_search
    size_t ef = std::max(ef_search_, k);
    auto top_candidates = search_layer(query, curr_obj, ef, 0);

    // Extract k nearest
    std::vector<HNSWSearchResult> results;
    results.reserve(std::min(k, top_candidates.size()));

    // top_candidates is a max-heap, so we need to reverse
    std::vector<std::pair<float, size_t>> temp;
    while (!top_candidates.empty()) {
      temp.push_back(top_candidates.top());
      top_candidates.pop();
    }

    // Sort by distance ascending
    std::sort(temp.begin(), temp.end());

    for (size_t i = 0; i < k && i < temp.size(); ++i) {
      results.push_back({external_ids_[temp[i].second], temp[i].first});
    }

    return results;
  }

  /**
   * @brief Sets the ef parameter for search (controls accuracy/speed tradeoff)
   * @param ef New ef value (higher = better recall, slower search)
   */
  void set_ef_search(size_t ef) {
    ef_search_ = ef;
  }

  /**
   * @brief Returns the current ef_search parameter
   */
  size_t get_ef_search() const {
    return ef_search_;
  }

  /**
   * @brief Returns the number of vectors in the index
   */
  size_t size() const {
    std::shared_lock<std::shared_mutex> lock(global_mutex_);
    return cur_element_count_;
  }

  /**
   * @brief Returns the dimension of stored vectors
   */
  size_t dimension() const {
    return dim_;
  }

  /**
   * @brief Returns the maximum capacity of the index
   */
  size_t capacity() const {
    return max_elements_;
  }

  /**
   * @brief Checks if a vector with given ID exists
   */
  bool contains(uint64_t id) const {
    std::shared_lock<std::shared_mutex> lock(global_mutex_);
    return id_to_internal_.find(id) != id_to_internal_.end();
  }

  /**
   * @brief Retrieves the vector data for a given ID
   * 
   * @param id External ID of the vector to retrieve
   * @return A copy of the vector data
   * @throws std::runtime_error if the ID is not found
   */
  std::vector<float> get_vector(uint64_t id) const {
    std::shared_lock<std::shared_mutex> lock(global_mutex_);
    auto it = id_to_internal_.find(id);
    if (it == id_to_internal_.end()) {
        throw std::runtime_error("Vector ID not found: " + std::to_string(id));
    }
    size_t internal_id = it->second;
    const float* data = vectors_.data() + internal_id * dim_;
    return std::vector<float>(data, data + dim_);
  }

  // --- Serialization ---
  /**
   * @brief Saves the index to a binary file.
   *
   * @param filename Path to the file where the index will be saved.
   * @throws std::runtime_error if the file cannot be opened.
   * @note Thread-safe: acquires exclusive lock.
   */
  void save(const std::string& filename) const {
    std::shared_lock<std::shared_mutex> global_lock(global_mutex_); // Use shared_lock for const method

    // Write to a temporary file first
    std::string temp_filename = filename + ".tmp";
    std::ofstream ofs(temp_filename, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + temp_filename);
    }

    try {
        // Write header (magic number, version, key parameters)
        const uint32_t magic = 0x51565244; // "QVRD"
        detail::write_binary(ofs, magic);
        const uint32_t version = 1;
        detail::write_binary(ofs, version);

        // Write configuration
        detail::write_binary(ofs, dim_);
        detail::write_binary(ofs, metric_);
        detail::write_binary(ofs, max_elements_);
        detail::write_binary(ofs, M_);
        detail::write_binary(ofs, ef_construction_);
        detail::write_binary(ofs, ef_search_);
        detail::write_binary(ofs, mult_);
        
        // Random engine state (can be ignored for loading if not critical for reproducibility post-load)
        // For now, let's skip to keep it simpler, as reconstructing graph structure is enough

        // Write graph state
        detail::write_binary(ofs, cur_element_count_.load());
        detail::write_binary(ofs, enterpoint_);
        detail::write_binary(ofs, max_level_);

        // Write main data vectors
        detail::write_vector(ofs, vectors_);
        detail::write_vector(ofs, external_ids_);
        detail::write_vector(ofs, levels_);

        // Write id_to_internal_ map
        detail::write_binary(ofs, id_to_internal_.size());
        for (const auto& pair : id_to_internal_) {
            detail::write_binary(ofs, pair.first);  // uint64_t id
            detail::write_binary(ofs, pair.second); // size_t internal_id
        }

        // Write neighbors_ (vector of vectors of vectors)
        detail::write_binary(ofs, neighbors_.size()); // Should be max_elements
        for (size_t i = 0; i < neighbors_.size(); ++i) {
            detail::write_binary(ofs, neighbors_[i].size()); // Number of levels for this node
            for (size_t l = 0; l < neighbors_[i].size(); ++l) {
                detail::write_vector(ofs, neighbors_[i][l]);
            }
        }
        ofs.close();

        // Atomic rename
        std::filesystem::rename(temp_filename, filename);
    } catch (...) {
        // Try to clean up temp file on failure
        ofs.close();
        std::filesystem::remove(temp_filename);
        throw;
    }
  }

  /**
   * @brief Loads the index from a binary file.
   *
   * @param filename Path to the file from which the index will be loaded.
   * @return A unique_ptr to a new HNSWIndex instance.
   * @throws std::runtime_error if the file cannot be opened, or if the format is invalid.
   * @note Not thread-safe for the current instance. It constructs a new instance.
   */
  static std::unique_ptr<HNSWIndex> load(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    // Read header
    uint32_t magic;
    detail::read_binary(ifs, magic);
    if (magic != 0x51565244) {
        throw std::runtime_error("Invalid magic number in file: " + filename);
    }
    uint32_t version;
    detail::read_binary(ifs, version);
    if (version != 1) {
        throw std::runtime_error("Unsupported file version: " + std::to_string(version));
    }

    // Read configuration
    size_t dim;
    HNSWDistanceMetric metric;
    size_t max_elements_val;
    size_t M, ef_construction, ef_search;
    double mult;

    detail::read_binary(ifs, dim);
    detail::read_binary(ifs, metric);
    detail::read_binary(ifs, max_elements_val);
    detail::read_binary(ifs, M);
    detail::read_binary(ifs, ef_construction);
    detail::read_binary(ifs, ef_search);
    detail::read_binary(ifs, mult);

    // Construct a new index object with the loaded configuration
    // Note: random_seed is not serialized, will use default 42
    auto index = std::make_unique<HNSWIndex>(dim, metric, max_elements_val, M, ef_construction);
    index->ef_search_ = ef_search; // Directly set as it might be different from ctor default
    index->mult_ = mult; // Restore mult_ as it depends on M, but M might be changed by ctor

    // Read graph state
    size_t cur_element_count_val;
    detail::read_binary(ifs, cur_element_count_val);
    index->cur_element_count_ = cur_element_count_val; // Restore atomic
    detail::read_binary(ifs, index->enterpoint_);
    detail::read_binary(ifs, index->max_level_);

    // Read main data vectors
    detail::read_vector(ifs, index->vectors_);
    detail::read_vector(ifs, index->external_ids_);
    detail::read_vector(ifs, index->levels_);

    // Read id_to_internal_ map
    size_t map_size;
    detail::read_binary(ifs, map_size);
    index->id_to_internal_.reserve(map_size);
    for (size_t i = 0; i < map_size; ++i) {
        uint64_t id;
        size_t internal_id;
        detail::read_binary(ifs, id);
        detail::read_binary(ifs, internal_id);
        index->id_to_internal_[id] = internal_id;
    }

    // Read neighbors_
    size_t neighbors_outer_size; // Should be max_elements_
    detail::read_binary(ifs, neighbors_outer_size);
    index->neighbors_.resize(neighbors_outer_size);
    for (size_t i = 0; i < neighbors_outer_size; ++i) {
        size_t node_levels_size;
        detail::read_binary(ifs, node_levels_size);
        index->neighbors_[i].resize(node_levels_size);
        for (size_t l = 0; l < node_levels_size; ++l) {
            detail::read_vector(ifs, index->neighbors_[i][l]);
        }
    }
    
    // Re-initialize node_locks_ for the new instance
    index->node_locks_.clear();
    index->node_locks_.reserve(max_elements_val);
    for (size_t i = 0; i < max_elements_val; ++i) {
      index->node_locks_.push_back(std::make_unique<std::shared_mutex>());
    }

    ifs.close();
    return index;
  }
  // --- End Serialization ---

private:
  // Max-heap comparator (we want smallest distances at top after popping all)
  using MaxHeap = std::priority_queue<
      std::pair<float, size_t>,
      std::vector<std::pair<float, size_t>>,
      std::less<std::pair<float, size_t>>>;

  /**
   * @brief Generates a random level for a new node
   *
   * The level is drawn from an exponential distribution, which results in
   * an expected number of elements at each level decreasing exponentially.
   */
  int get_random_level() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = -std::log(dist(level_generator_)) * mult_;
    return static_cast<int>(r);
  }

  /**
   * @brief Gets pointer to vector data by internal ID
   */
  const float* get_vector(size_t internal_id) const {
    return vectors_.data() + internal_id * dim_;
  }

  /**
   * @brief Computes distance between two vectors
   */
  float compute_distance(const float* a, const float* b) const {
    switch (metric_) {
      case HNSWDistanceMetric::L2:
        return l2_sq(a, b, dim_);
      case HNSWDistanceMetric::COSINE:
        return cosine_distance(a, b, dim_);
      case HNSWDistanceMetric::DOT:
        return -dot_product(a, b, dim_);
      default:
        return std::numeric_limits<float>::infinity();
    }
  }

  /**
   * @brief Searches a single layer for nearest neighbors
   *
   * Returns a max-heap of (distance, internal_id) pairs
   */
  MaxHeap search_layer(const float* query, size_t entry_point, size_t ef, int level) const {
    std::unordered_set<size_t> visited;
    visited.insert(entry_point);

    // Candidate set: min-heap (negated distances for min behavior)
    std::priority_queue<
        std::pair<float, size_t>,
        std::vector<std::pair<float, size_t>>,
        std::greater<std::pair<float, size_t>>> candidates;

    // Result set: max-heap
    MaxHeap results;

    float dist = compute_distance(query, get_vector(entry_point));
    candidates.emplace(dist, entry_point);
    results.emplace(dist, entry_point);

    float lower_bound = dist;

    while (!candidates.empty()) {
      auto [candidate_dist, candidate_id] = candidates.top();

      if (candidate_dist > lower_bound && results.size() >= ef) {
        break;
      }
      candidates.pop();

      // Get neighbors
      std::shared_lock<std::shared_mutex> lock(*node_locks_[candidate_id]);

      if (static_cast<int>(neighbors_[candidate_id].size()) <= level) continue;

      const auto& neighbors = neighbors_[candidate_id][level];

      for (size_t neighbor_id : neighbors) {
        if (visited.count(neighbor_id)) continue;
        visited.insert(neighbor_id);

        float neighbor_dist = compute_distance(query, get_vector(neighbor_id));

        if (results.size() < ef || neighbor_dist < lower_bound) {
          candidates.emplace(neighbor_dist, neighbor_id);
          results.emplace(neighbor_dist, neighbor_id);

          if (results.size() > ef) {
            results.pop();
          }

          if (!results.empty()) {
            lower_bound = results.top().first;
          }
        }
      }
    }

    return results;
  }

  /**
   * @brief Selects M neighbors using the heuristic
   *
   * This implements a simple heuristic that prefers diverse neighbors.
   */
  std::vector<size_t> select_neighbors(
      [[maybe_unused]] const float* query,
      MaxHeap& candidates,
      size_t M,
      [[maybe_unused]] int level) const {

    if (candidates.size() <= M) {
      std::vector<size_t> result;
      result.reserve(candidates.size());
      while (!candidates.empty()) {
        result.push_back(candidates.top().second);
        candidates.pop();
      }
      return result;
    }

    // Convert to sorted vector (closest first)
    std::vector<std::pair<float, size_t>> sorted;
    sorted.reserve(candidates.size());
    while (!candidates.empty()) {
      sorted.push_back(candidates.top());
      candidates.pop();
    }
    std::sort(sorted.begin(), sorted.end());

    // Select using heuristic: prefer neighbors that are not too close to each other
    std::vector<size_t> result;
    result.reserve(M);

    for (const auto& [dist_to_query, candidate_id] : sorted) {
      if (result.size() >= M) break;

      bool good = true;
      for (size_t selected_id : result) {
        float dist_between = compute_distance(get_vector(candidate_id), get_vector(selected_id));
        if (dist_between < dist_to_query) {
          good = false;
          break;
        }
      }

      if (good) {
        result.push_back(candidate_id);
      }
    }

    // If heuristic didn't select enough, add closest remaining
    if (result.size() < M) {
      std::unordered_set<size_t> selected_set(result.begin(), result.end());
      for (const auto& [dist, id] : sorted) {
        if (result.size() >= M) break;
        if (selected_set.count(id) == 0) {
          result.push_back(id);
        }
      }
    }

    return result;
  }

  // Configuration
  size_t dim_;
  HNSWDistanceMetric metric_;
  size_t max_elements_;
  size_t M_;           // Number of connections to make during construction
  size_t M_max_;       // Maximum connections per node (levels > 0)
  size_t M_max0_;      // Maximum connections per node (level 0)
  size_t ef_construction_;
  size_t ef_search_;
  double mult_;        // Level multiplier for random level generation

  // Random number generator for level selection
  mutable std::mt19937 level_generator_;

  // Data storage
  std::vector<float> vectors_;                           // Flat vector storage
  std::vector<uint64_t> external_ids_;                   // Internal ID -> External ID
  std::unordered_map<uint64_t, size_t> id_to_internal_;  // External ID -> Internal ID
  std::vector<int> levels_;                              // Level of each node
  std::vector<std::vector<std::vector<size_t>>> neighbors_;  // neighbors_[node][level] = list of neighbor IDs

  // Graph structure
  size_t enterpoint_;
  int max_level_;
  std::atomic<size_t> cur_element_count_{0};

  // Thread safety
  mutable std::shared_mutex global_mutex_;
  mutable std::vector<std::unique_ptr<std::shared_mutex>> node_locks_;
};

} // namespace quiverdb
