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
#include "quiverdb.h"

quiverdb::Index index(768);  // 768-dimensional vectors
index.add(vector_id, embedding);
auto results = index.search(query, k=10);
```

## Building and Running
Prerequisites
- CMake 4.1+
- Conan 2.0+
- C++20 compiler (Clang 17+ / GCC 11+ / MSVC 19.30+)

```bash
# One-time setup
conan profile detect --force

# Build
conan install . --build=missing -s build_type=Release
cmake --preset conan-release
cmake --build build/Release

### Run
# Run tests
./build/Release/test_distance

# Run benchmarks
./build/Release/bench_distance
```