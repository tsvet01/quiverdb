// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once
#include "distance.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace quiverdb {

namespace detail {
template <typename T> void write_bin(std::ofstream& f, const T& v) {
  f.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T> void read_bin(std::ifstream& f, T& v) {
  if (!f.read(reinterpret_cast<char*>(&v), sizeof(T)))
    throw std::runtime_error("Unexpected end of file or read error");
}
template <typename T> void write_vec(std::ofstream& f, const std::vector<T>& v) {
  write_bin(f, v.size());
  if (!v.empty()) f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(T));
}
constexpr size_t MAX_VEC_SIZE = 100000000ULL;
constexpr size_t MAX_RNG_STATE_SIZE = 10000;  // Reasonable upper bound for serialized RNG state
template <typename T> void read_vec(std::ifstream& f, std::vector<T>& v) {
  size_t sz; read_bin(f, sz);
  if (sz > MAX_VEC_SIZE || sz > SIZE_MAX / sizeof(T))
    throw std::runtime_error("Corrupted file: vector too large");
  v.resize(sz);
  if (!v.empty() && !f.read(reinterpret_cast<char*>(v.data()), sz * sizeof(T)))
    throw std::runtime_error("Unexpected end of file or read error");
}
} // namespace detail

enum class HNSWDistanceMetric { L2, COSINE, DOT };

struct HNSWSearchResult {
  uint64_t id;
  float distance;
  bool operator<(const HNSWSearchResult& o) const { return distance < o.distance; }
  bool operator>(const HNSWSearchResult& o) const { return distance > o.distance; }
};

class HNSWIndex {
public:
  static constexpr uint32_t MAGIC = 0x51565244;  // "QVRD" (QuiverDB) in little-endian
  static constexpr uint32_t VERSION = 2;  // v2: added RNG state serialization
  static constexpr int MAX_LEVEL = 32;  // Reasonable upper bound for HNSW levels
  static constexpr size_t INVALID_ID = static_cast<size_t>(-1);  // Sentinel for empty entry point

  explicit HNSWIndex(size_t dimension, HNSWDistanceMetric metric = HNSWDistanceMetric::L2,
      size_t max_elements = 100000, size_t M = 16, size_t ef_construction = 200, uint32_t seed = 42)
      : dim_(dimension), metric_(metric), max_elements_(max_elements), M_(M), M_max_(M),
        M_max0_(M * 2), ef_construction_(std::max(ef_construction, M)), ef_search_(50),
        mult_(M > 1 ? 1.0 / std::log(static_cast<double>(M)) : 1.0), level_gen_(seed) {
    if (dimension == 0) throw std::invalid_argument("Dimension must be > 0");
    if (max_elements == 0) throw std::invalid_argument("max_elements must be > 0");
    if (M < 2) throw std::invalid_argument("M must be >= 2");
    if (max_elements > SIZE_MAX / dim_) throw std::invalid_argument("max_elements * dimension overflow");
    vectors_.resize(max_elements * dim_);
    ext_ids_.resize(max_elements);
    levels_.resize(max_elements, 0);
    neighbors_.resize(max_elements);
    locks_.reserve(max_elements);
    for (size_t i = 0; i < max_elements; ++i) locks_.push_back(std::make_unique<std::shared_mutex>());
  }

  void add(uint64_t id, const float* vec) {
    if (!vec) throw std::invalid_argument("Vector must not be null");
    std::unique_lock glock(global_mtx_);
    if (id_map_.count(id)) throw std::invalid_argument("ID " + std::to_string(id) + " exists");
    if (count_ >= max_elements_) throw std::runtime_error("Index full");

    size_t iid = count_++;
    id_map_[id] = iid;
    ext_ids_[iid] = id;
    std::copy_n(vec, dim_, vectors_.begin() + iid * dim_);

    int level = get_level();
    levels_[iid] = level;
    neighbors_[iid].resize(level + 1);
    for (int l = 0; l <= level; ++l)
      neighbors_[iid][l].reserve(l == 0 ? M_max0_ : M_max_);

    if (ep_.load() == INVALID_ID) { ep_.store(iid); max_level_.store(level); return; }

    size_t curr = ep_.load();
    int cur_max_level = max_level_.load();
    if (level < cur_max_level) {
      float d = dist(vec, get_vec(curr));
      for (int l = cur_max_level; l > level; --l) {
        bool changed = true;
        while (changed) {
          changed = false;
          std::shared_lock lk(*locks_[curr]);
          for (size_t n : neighbors_[curr][l]) {
            float nd = dist(vec, get_vec(n));
            if (nd < d) { d = nd; curr = n; changed = true; }
          }
        }
      }
    }

    for (int l = std::min(level, cur_max_level); l >= 0; --l) {
      auto top = search_layer(vec, curr, ef_construction_, l);
      auto sel = select_neighbors(top, M_, l);
      { std::unique_lock lk(*locks_[iid]); neighbors_[iid][l] = std::move(sel); }

      size_t max_conn = l == 0 ? M_max0_ : M_max_;
      for (size_t nid : neighbors_[iid][l]) {
        std::unique_lock lk(*locks_[nid]);
        auto& nc = neighbors_[nid][l];
        if (nc.size() < max_conn) { nc.push_back(iid); }
        else {
          float d2new = dist(get_vec(nid), vec);
          std::vector<std::pair<float, size_t>> cands;
          cands.reserve(nc.size() + 1);
          for (size_t c : nc) cands.emplace_back(dist(get_vec(nid), get_vec(c)), c);
          cands.emplace_back(d2new, iid);
          std::sort(cands.begin(), cands.end());
          nc.clear();
          for (size_t i = 0; i < max_conn && i < cands.size(); ++i) nc.push_back(cands[i].second);
        }
      }
      // Use closest candidate (min distance) for next layer entry point
      if (!top.empty()) {
        std::pair<float, size_t> best = top.top();
        while (!top.empty()) {
          if (top.top().first < best.first) best = top.top();
          top.pop();
        }
        curr = best.second;
      }
    }
    if (level > cur_max_level) { ep_.store(iid); max_level_.store(level); }
  }

  std::vector<HNSWSearchResult> search(const float* query, size_t k) const {
    if (!query) throw std::invalid_argument("Query must not be null");
    if (k == 0) throw std::invalid_argument("k must be > 0");
    std::shared_lock glock(global_mtx_);
    if (count_ == 0) return {};

    size_t curr = ep_.load();
    float d = dist(query, get_vec(curr));
    for (int l = max_level_.load(); l > 0; --l) {
      bool changed = true;
      while (changed) {
        changed = false;
        std::shared_lock lk(*locks_[curr]);
        if (static_cast<int>(neighbors_[curr].size()) <= l) continue;
        for (size_t n : neighbors_[curr][l]) {
          float nd = dist(query, get_vec(n));
          if (nd < d) { d = nd; curr = n; changed = true; }
        }
      }
    }

    auto top = search_layer(query, curr, std::max(ef_search_, k), 0);
    std::vector<std::pair<float, size_t>> temp;
    while (!top.empty()) { temp.push_back(top.top()); top.pop(); }
    std::sort(temp.begin(), temp.end());

    std::vector<HNSWSearchResult> res;
    res.reserve(std::min(k, temp.size()));
    for (size_t i = 0; i < k && i < temp.size(); ++i)
      res.push_back({ext_ids_[temp[i].second], temp[i].first});
    return res;
  }

  void set_ef_search(size_t ef) {
    if (ef == 0) throw std::invalid_argument("ef_search must be > 0");
    ef_search_ = ef;
  }
  size_t get_ef_search() const { return ef_search_; }
  size_t size() const { std::shared_lock lk(global_mtx_); return count_; }
  size_t dimension() const { return dim_; }
  size_t capacity() const { return max_elements_; }
  bool contains(uint64_t id) const { std::shared_lock lk(global_mtx_); return id_map_.count(id); }

  std::vector<float> get_vector(uint64_t id) const {
    std::shared_lock lk(global_mtx_);
    auto it = id_map_.find(id);
    if (it == id_map_.end()) throw std::runtime_error("ID not found: " + std::to_string(id));
    const float* p = vectors_.data() + it->second * dim_;
    return std::vector<float>(p, p + dim_);
  }

  void save(const std::string& filename) const {
    std::shared_lock glock(global_mtx_);
    std::string tmp = filename + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + tmp);
    try {
      detail::write_bin(f, MAGIC);
      detail::write_bin(f, VERSION);
      detail::write_bin(f, dim_);
      detail::write_bin(f, static_cast<uint32_t>(metric_));
      detail::write_bin(f, max_elements_);
      detail::write_bin(f, M_);
      detail::write_bin(f, ef_construction_);
      detail::write_bin(f, ef_search_);
      detail::write_bin(f, mult_);
      detail::write_bin(f, count_.load());
      detail::write_bin(f, ep_.load());
      detail::write_bin(f, max_level_.load());
      detail::write_vec(f, vectors_);
      detail::write_vec(f, ext_ids_);
      detail::write_vec(f, levels_);
      detail::write_bin(f, id_map_.size());
      for (const auto& [k, v] : id_map_) { detail::write_bin(f, k); detail::write_bin(f, v); }
      detail::write_bin(f, neighbors_.size());
      for (size_t i = 0; i < neighbors_.size(); ++i) {
        detail::write_bin(f, neighbors_[i].size());
        for (size_t l = 0; l < neighbors_[i].size(); ++l) detail::write_vec(f, neighbors_[i][l]);
      }
      // Save RNG state for deterministic behavior after load
      std::stringstream rng_ss;
      rng_ss << level_gen_;
      std::string rng_state = rng_ss.str();
      detail::write_bin(f, rng_state.size());
      f.write(rng_state.data(), rng_state.size());
      f.flush();
      if (!f) { std::filesystem::remove(tmp); throw std::runtime_error("Write failed: " + tmp); }
#if defined(_WIN32) || defined(_WIN64)
      // FlushFileBuffers for durability before atomic rename
      HANDLE hFile = CreateFileA(tmp.c_str(), GENERIC_WRITE, 0, NULL,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if (hFile != INVALID_HANDLE_VALUE) { FlushFileBuffers(hFile); CloseHandle(hFile); }
#elif defined(__unix__) || defined(__APPLE__)
      // fsync for durability before atomic rename
      int fd = open(tmp.c_str(), O_RDONLY);
      if (fd >= 0) { fsync(fd); close(fd); }
#endif
      f.close();
      std::filesystem::rename(tmp, filename);
    } catch (...) { f.close(); std::filesystem::remove(tmp); throw; }
  }

  static std::unique_ptr<HNSWIndex> load(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + filename);
    uint32_t magic, ver;
    detail::read_bin(f, magic);
    if (magic != MAGIC) throw std::runtime_error("Invalid magic");
    detail::read_bin(f, ver);
    if (ver != VERSION && ver != 1) throw std::runtime_error("Unsupported version");

    size_t dim, max_el, M, ef_con, ef_s; uint32_t met; double mult;
    detail::read_bin(f, dim);
    detail::read_bin(f, met);
    detail::read_bin(f, max_el);
    detail::read_bin(f, M);
    detail::read_bin(f, ef_con);
    detail::read_bin(f, ef_s);
    detail::read_bin(f, mult);

    auto idx = std::make_unique<HNSWIndex>(dim, static_cast<HNSWDistanceMetric>(met), max_el, M, ef_con);
    idx->ef_search_ = ef_s;
    idx->mult_ = mult;

    size_t cnt, ep_val;
    int max_level_val;
    detail::read_bin(f, cnt);
    if (cnt > max_el) throw std::runtime_error("Corrupted file: count exceeds max_elements");
    idx->count_.store(cnt);
    detail::read_bin(f, ep_val);
    detail::read_bin(f, max_level_val);
    // Validate ep_ and max_level_
    if (cnt > 0) {
      if (ep_val >= cnt) throw std::runtime_error("Corrupted file: invalid entry point");
      if (max_level_val < 0 || max_level_val > MAX_LEVEL)
        throw std::runtime_error("Corrupted file: invalid max_level");
    } else {
      // Empty index must have invalid entry point
      if (ep_val != INVALID_ID)
        throw std::runtime_error("Corrupted file: non-empty entry point for empty index");
    }
    idx->ep_.store(ep_val);
    idx->max_level_.store(max_level_val);
    detail::read_vec(f, idx->vectors_);
    detail::read_vec(f, idx->ext_ids_);
    detail::read_vec(f, idx->levels_);

    size_t msz;
    detail::read_bin(f, msz);
    if (msz > cnt) throw std::runtime_error("Corrupted file: id_map size exceeds count");
    idx->id_map_.reserve(msz);
    for (size_t i = 0; i < msz; ++i) {
      uint64_t k; size_t v;
      detail::read_bin(f, k);
      detail::read_bin(f, v);
      if (v >= cnt) throw std::runtime_error("Corrupted file: invalid internal index in id_map");
      idx->id_map_[k] = v;
    }

    size_t nsz;
    detail::read_bin(f, nsz);
    if (nsz > max_el) throw std::runtime_error("Corrupted file: neighbors size exceeds max_elements");
    idx->neighbors_.resize(nsz);
    for (size_t i = 0; i < nsz; ++i) {
      size_t lsz;
      detail::read_bin(f, lsz);
      if (lsz > static_cast<size_t>(MAX_LEVEL) + 1) throw std::runtime_error("Corrupted file: too many levels");
      idx->neighbors_[i].resize(lsz);
      for (size_t l = 0; l < lsz; ++l) {
        detail::read_vec(f, idx->neighbors_[i][l]);
        // Validate neighbor indices are within bounds
        for (size_t nid : idx->neighbors_[i][l]) {
          if (nid >= cnt) throw std::runtime_error("Corrupted file: invalid neighbor index");
        }
      }
    }

    // Restore RNG state for deterministic behavior (v2+)
    if (ver >= 2) {
      size_t rng_state_size;
      detail::read_bin(f, rng_state_size);
      if (rng_state_size > detail::MAX_RNG_STATE_SIZE)
        throw std::runtime_error("Corrupted file: RNG state too large");
      std::string rng_state(rng_state_size, '\0');
      if (!f.read(rng_state.data(), rng_state_size))
        throw std::runtime_error("Unexpected end of file or read error");
      std::stringstream rng_ss(rng_state);
      rng_ss >> idx->level_gen_;
      if (rng_ss.fail()) throw std::runtime_error("Corrupted file: invalid RNG state");
    }
    // Note: v1 files don't have RNG state, level_gen_ keeps default initialization

    idx->locks_.clear();
    idx->locks_.reserve(max_el);
    for (size_t i = 0; i < max_el; ++i) idx->locks_.push_back(std::make_unique<std::shared_mutex>());
    return idx;
  }

private:
  using MaxHeap = std::priority_queue<std::pair<float, size_t>>;

  int get_level() {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    double r = std::max(d(level_gen_), 1e-9);  // Clamp to prevent log overflow
    int level = static_cast<int>(-std::log(r) * mult_);
    return std::min(level, MAX_LEVEL);
  }

  const float* get_vec(size_t iid) const { return vectors_.data() + iid * dim_; }

  float dist(const float* a, const float* b) const {
    switch (metric_) {
      case HNSWDistanceMetric::L2: return l2_sq(a, b, dim_);
      case HNSWDistanceMetric::COSINE: return cosine_distance(a, b, dim_);
      case HNSWDistanceMetric::DOT: return -dot_product(a, b, dim_);
      default: return std::numeric_limits<float>::infinity();
    }
  }

  MaxHeap search_layer(const float* q, size_t ep, size_t ef, int level) const {
    std::unordered_set<size_t> vis;
    vis.insert(ep);
    std::priority_queue<std::pair<float, size_t>, std::vector<std::pair<float, size_t>>,
                        std::greater<std::pair<float, size_t>>> cands;
    MaxHeap res;
    float d = dist(q, get_vec(ep));
    cands.emplace(d, ep);
    res.emplace(d, ep);
    float lb = d;

    while (!cands.empty()) {
      auto [cd, cid] = cands.top();
      if (cd > lb && res.size() >= ef) break;
      cands.pop();
      std::shared_lock lk(*locks_[cid]);
      if (static_cast<int>(neighbors_[cid].size()) <= level) continue;
      for (size_t n : neighbors_[cid][level]) {
        if (vis.count(n)) continue;
        vis.insert(n);
        float nd = dist(q, get_vec(n));
        if (res.size() < ef || nd < lb) {
          cands.emplace(nd, n);
          res.emplace(nd, n);
          if (res.size() > ef) res.pop();
          if (!res.empty()) lb = res.top().first;
        }
      }
    }
    return res;
  }

  std::vector<size_t> select_neighbors(MaxHeap& cands, size_t M, int /*level*/) const {
    if (cands.size() <= M) {
      std::vector<size_t> r;
      r.reserve(cands.size());
      while (!cands.empty()) { r.push_back(cands.top().second); cands.pop(); }
      return r;
    }
    std::vector<std::pair<float, size_t>> sorted;
    sorted.reserve(cands.size());
    while (!cands.empty()) { sorted.push_back(cands.top()); cands.pop(); }
    std::sort(sorted.begin(), sorted.end());

    std::vector<size_t> r;
    r.reserve(M);
    for (auto& [dq, cid] : sorted) {
      if (r.size() >= M) break;
      bool ok = true;
      for (size_t s : r)
        if (dist(get_vec(cid), get_vec(s)) < dq) { ok = false; break; }
      if (ok) r.push_back(cid);
    }
    if (r.size() < M) {
      std::unordered_set<size_t> sel(r.begin(), r.end());
      for (auto& p : sorted) {
        if (r.size() >= M) break;
        if (!sel.count(p.second)) r.push_back(p.second);
      }
    }
    return r;
  }

  size_t dim_;
  HNSWDistanceMetric metric_;
  size_t max_elements_, M_, M_max_, M_max0_, ef_construction_, ef_search_;
  double mult_;
  std::mt19937 level_gen_;
  std::vector<float> vectors_;
  std::vector<uint64_t> ext_ids_;
  std::unordered_map<uint64_t, size_t> id_map_;
  std::vector<int> levels_;
  std::vector<std::vector<std::vector<size_t>>> neighbors_;
  std::atomic<size_t> ep_{INVALID_ID};
  std::atomic<int> max_level_{-1};
  std::atomic<size_t> count_{0};
  mutable std::shared_mutex global_mtx_;
  mutable std::vector<std::unique_ptr<std::shared_mutex>> locks_;
};

} // namespace quiverdb
