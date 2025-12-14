// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once
#include <cstddef>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_MAC || TARGET_OS_IPHONE
#define QUIVER_HAS_METAL 1
#endif
#endif

#if defined(__CUDACC__) || defined(QUIVER_CUDA_ENABLED)
#define QUIVER_HAS_CUDA 1
#endif

namespace quiverdb {
namespace gpu {

enum class Backend { NONE, METAL, CUDA };

inline Backend available() {
#if defined(QUIVER_HAS_METAL)
  return Backend::METAL;
#elif defined(QUIVER_HAS_CUDA)
  return Backend::CUDA;
#else
  return Backend::NONE;
#endif
}

constexpr size_t GPU_THRESHOLD = 50000;
inline bool use_gpu(size_t n, size_t d) { return available() != Backend::NONE && n * d >= GPU_THRESHOLD; }

} // namespace gpu
} // namespace quiverdb
