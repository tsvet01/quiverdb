# quiverdb

Embeddable vector database for edge AI. Lightning-fast semantic search that runs anywhere.

## Features (coming soon)
- **Compact**: <1MB binary, runs on mobile/edge devices
- **Fast**: SIMD-optimized distance calculations
- **Zero dependencies**: Single-header C++ library
- **Cross-platform**: Linux, macOS, Windows, iOS, Android
- **Local-first**: No network required, complete privacy

## Quick Start
```cpp
#include "core/distance.h"

// 768-dimensional vectors (e.g., OpenAI embeddings)
float vec_a[768] = {/* ... */};
float vec_b[768] = {/* ... */};

// Compute L2 squared distance (auto-selects SIMD implementation)
float distance = quiverdb::l2_sq(vec_a, vec_b, 768);
```

## Building and Running
Prerequisites
- CMake 3.20+
- Conan 2.0+
- C++20 compiler (Clang 17+ / GCC 11+ / MSVC 19.30+)

```bash
# One-time setup
conan profile detect --force

# Build
conan install . --build=missing -s build_type=Release -s compiler.cppstd=20
cmake --preset conan-release
cmake --build build/Release

# Run tests
./build/Release/test_distance

# Run benchmarks
./build/Release/bench_distance
```