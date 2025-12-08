// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once

#if defined(_WIN32) || defined(_WIN64)
#define QUIVERDB_WINDOWS 1
#include <windows.h>
#else
#define QUIVERDB_POSIX 1
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "distance.h"
#include "vector_store.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace quiverdb {

class MMapVectorStore {
public:
  static constexpr uint32_t MAGIC = 0x42445651;
  static constexpr uint32_t VERSION = 1;
  static constexpr size_t HEADER_SIZE = 32;

  explicit MMapVectorStore(const std::string& filename) {
#ifdef QUIVERDB_WINDOWS
    file_handle_ = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open: " + filename);
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(file_handle_, &sz)) { CloseHandle(file_handle_); file_handle_ = INVALID_HANDLE_VALUE;
      throw std::runtime_error("Cannot get file size"); }
    file_size_ = static_cast<size_t>(sz.QuadPart);
    if (file_size_ < HEADER_SIZE) { CloseHandle(file_handle_); file_handle_ = INVALID_HANDLE_VALUE;
      throw std::runtime_error("File too small"); }
    mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_handle_) { CloseHandle(file_handle_); file_handle_ = INVALID_HANDLE_VALUE;
      throw std::runtime_error("Cannot create mapping"); }
    mapped_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
    if (!mapped_) { CloseHandle(mapping_handle_); CloseHandle(file_handle_);
      mapping_handle_ = nullptr; file_handle_ = INVALID_HANDLE_VALUE;
      throw std::runtime_error("Cannot map file"); }
#else
    fd_ = open(filename.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("Cannot open: " + filename);
    struct stat sb;
    if (fstat(fd_, &sb) < 0) { close(fd_); fd_ = -1; throw std::runtime_error("Cannot stat file"); }
    file_size_ = static_cast<size_t>(sb.st_size);
    if (file_size_ < HEADER_SIZE) { close(fd_); fd_ = -1; throw std::runtime_error("File too small"); }
    mapped_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) { close(fd_); fd_ = -1; mapped_ = nullptr;
      throw std::runtime_error("Cannot mmap file"); }
#endif
    const uint8_t* p = static_cast<const uint8_t*>(mapped_);
    uint32_t magic; std::memcpy(&magic, p, 4);
    if (magic != MAGIC) { cleanup(); throw std::runtime_error("Invalid magic"); }
    p += 4;
    uint32_t ver; std::memcpy(&ver, p, 4);
    if (ver != VERSION) { cleanup(); throw std::runtime_error("Unsupported version"); }
    p += 4;
    std::memcpy(&dim_, p, 8); p += 8;
    std::memcpy(&num_vectors_, p, 8); p += 8;
    uint32_t met; std::memcpy(&met, p, 4);
    if (met > 2) { cleanup(); throw std::runtime_error("Invalid metric"); }
    metric_ = static_cast<DistanceMetric>(met);

    if (num_vectors_ > SIZE_MAX / sizeof(uint64_t) ||
        dim_ > SIZE_MAX / sizeof(float) ||
        (num_vectors_ > 0 && num_vectors_ > SIZE_MAX / (dim_ * sizeof(float)))) {
      cleanup(); throw std::runtime_error("File corrupted: size overflow");
    }
    size_t expected = HEADER_SIZE + num_vectors_ * sizeof(uint64_t) + num_vectors_ * dim_ * sizeof(float);
    if (file_size_ < expected) { cleanup(); throw std::runtime_error("File truncated"); }

    ids_ptr_ = reinterpret_cast<const uint64_t*>(static_cast<const uint8_t*>(mapped_) + HEADER_SIZE);
    vectors_ptr_ = reinterpret_cast<const float*>(
        static_cast<const uint8_t*>(mapped_) + HEADER_SIZE + num_vectors_ * sizeof(uint64_t));
    for (size_t i = 0; i < num_vectors_; ++i) id_map_[ids_ptr_[i]] = i;
  }

  ~MMapVectorStore() { cleanup(); }
  MMapVectorStore(const MMapVectorStore&) = delete;
  MMapVectorStore& operator=(const MMapVectorStore&) = delete;

  MMapVectorStore(MMapVectorStore&& o) noexcept :
#ifdef QUIVERDB_WINDOWS
    file_handle_(o.file_handle_), mapping_handle_(o.mapping_handle_),
#else
    fd_(o.fd_),
#endif
    mapped_(o.mapped_), file_size_(o.file_size_), dim_(o.dim_), num_vectors_(o.num_vectors_),
    metric_(o.metric_), ids_ptr_(o.ids_ptr_), vectors_ptr_(o.vectors_ptr_), id_map_(std::move(o.id_map_)) {
#ifdef QUIVERDB_WINDOWS
    o.file_handle_ = INVALID_HANDLE_VALUE; o.mapping_handle_ = nullptr;
#else
    o.fd_ = -1;
#endif
    o.mapped_ = nullptr;
  }

  MMapVectorStore& operator=(MMapVectorStore&& o) noexcept {
    if (this != &o) {
      cleanup();
#ifdef QUIVERDB_WINDOWS
      file_handle_ = o.file_handle_; mapping_handle_ = o.mapping_handle_;
      o.file_handle_ = INVALID_HANDLE_VALUE; o.mapping_handle_ = nullptr;
#else
      fd_ = o.fd_; o.fd_ = -1;
#endif
      mapped_ = o.mapped_; file_size_ = o.file_size_; dim_ = o.dim_; num_vectors_ = o.num_vectors_;
      metric_ = o.metric_; ids_ptr_ = o.ids_ptr_; vectors_ptr_ = o.vectors_ptr_;
      id_map_ = std::move(o.id_map_); o.mapped_ = nullptr;
    }
    return *this;
  }

  const float* get(uint64_t id) const {
    auto it = id_map_.find(id);
    return it == id_map_.end() ? nullptr : vectors_ptr_ + it->second * dim_;
  }

  bool contains(uint64_t id) const { return id_map_.count(id); }

  std::vector<SearchResult> search(const float* query, size_t k) const {
    if (!query) throw std::invalid_argument("Query must not be null");
    if (k == 0) throw std::invalid_argument("k must be > 0");
    std::vector<SearchResult> res;
    res.reserve(num_vectors_);
    for (size_t i = 0; i < num_vectors_; ++i)
      res.push_back({ids_ptr_[i], dist(query, vectors_ptr_ + i * dim_)});
    size_t n = std::min(k, res.size());
    std::partial_sort(res.begin(), res.begin() + n, res.end());
    res.resize(n);
    return res;
  }

  size_t size() const { return num_vectors_; }
  size_t dimension() const { return dim_; }
  DistanceMetric metric() const { return metric_; }

private:
  void cleanup() {
#ifdef QUIVERDB_WINDOWS
    if (mapped_) { UnmapViewOfFile(mapped_); mapped_ = nullptr; }
    if (mapping_handle_) { CloseHandle(mapping_handle_); mapping_handle_ = nullptr; }
    if (file_handle_ != INVALID_HANDLE_VALUE) { CloseHandle(file_handle_); file_handle_ = INVALID_HANDLE_VALUE; }
#else
    if (mapped_ && mapped_ != MAP_FAILED) { munmap(mapped_, file_size_); mapped_ = nullptr; }
    if (fd_ >= 0) { close(fd_); fd_ = -1; }
#endif
  }

  float dist(const float* a, const float* b) const {
    switch (metric_) {
      case DistanceMetric::L2: return l2_sq(a, b, dim_);
      case DistanceMetric::COSINE: return cosine_distance(a, b, dim_);
      case DistanceMetric::DOT: return -dot_product(a, b, dim_);
      default: return std::numeric_limits<float>::infinity();
    }
  }

#ifdef QUIVERDB_WINDOWS
  HANDLE file_handle_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_handle_ = nullptr;
#else
  int fd_ = -1;
#endif
  void* mapped_ = nullptr;
  size_t file_size_ = 0, dim_ = 0, num_vectors_ = 0;
  DistanceMetric metric_ = DistanceMetric::L2;
  const uint64_t* ids_ptr_ = nullptr;
  const float* vectors_ptr_ = nullptr;
  std::unordered_map<uint64_t, size_t> id_map_;
};

class MMapVectorStoreBuilder {
public:
  explicit MMapVectorStoreBuilder(size_t dimension, DistanceMetric metric = DistanceMetric::L2)
      : dim_(dimension), metric_(metric) {
    if (dimension == 0) throw std::invalid_argument("Dimension must be > 0");
  }

  void add(uint64_t id, const float* vec) {
    if (!vec) throw std::invalid_argument("Vector must not be null");
    if (id_set_.count(id)) throw std::invalid_argument("Duplicate ID: " + std::to_string(id));
    ids_.push_back(id);
    vectors_.insert(vectors_.end(), vec, vec + dim_);
    id_set_.insert(id);
  }

  void reserve(size_t cap) { ids_.reserve(cap); vectors_.reserve(cap * dim_); }

  void save(const std::string& filename) const {
    std::string tmp = filename + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + tmp);
    uint32_t magic = MMapVectorStore::MAGIC, ver = MMapVectorStore::VERSION;
    uint64_t dim = dim_, nv = ids_.size();
    uint32_t met = static_cast<uint32_t>(metric_), reserved = 0;
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&ver), 4);
    f.write(reinterpret_cast<const char*>(&dim), 8);
    f.write(reinterpret_cast<const char*>(&nv), 8);
    f.write(reinterpret_cast<const char*>(&met), 4);
    f.write(reinterpret_cast<const char*>(&reserved), 4);
    f.write(reinterpret_cast<const char*>(ids_.data()), ids_.size() * sizeof(uint64_t));
    f.write(reinterpret_cast<const char*>(vectors_.data()), vectors_.size() * sizeof(float));
    if (!f) { std::remove(tmp.c_str()); throw std::runtime_error("Write failed"); }
    f.close();
    if (std::rename(tmp.c_str(), filename.c_str()) != 0) {
      std::remove(tmp.c_str()); throw std::runtime_error("Rename failed");
    }
  }

  size_t size() const { return ids_.size(); }
  size_t dimension() const { return dim_; }

private:
  size_t dim_;
  DistanceMetric metric_;
  std::vector<uint64_t> ids_;
  std::vector<float> vectors_;
  std::unordered_set<uint64_t> id_set_;
};

} // namespace quiverdb
