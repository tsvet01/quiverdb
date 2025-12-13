<div align="center">

# 🏹 QuiverDB

**Embeddable vector database for edge AI**

*Lightning-fast semantic search that runs anywhere*

[![Build and Test](https://github.com/tsvet01/quiverdb/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/tsvet01/quiverdb/actions/workflows/build-and-test.yml)
[![codecov](https://codecov.io/gh/tsvet01/quiverdb/branch/main/graph/badge.svg)](https://codecov.io/gh/tsvet01/quiverdb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C.svg?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Header Only](https://img.shields.io/badge/header--only-yes-success.svg)](https://en.wikipedia.org/wiki/Header-only)

[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows%20%7C%20iOS%20%7C%20Android-lightgrey.svg)]()
[![GitHub stars](https://img.shields.io/github/stars/tsvet01/quiverdb?style=social)](https://github.com/tsvet01/quiverdb/stargazers)

</div>

---

## 📑 Table of Contents

- [Why QuiverDB?](#why-quiverdb)
- [Features](#-features)
- [Performance](#-performance)
- [Quick Start](#-quick-start)
- [Python Bindings](#-python-bindings)
- [Building and Running](#-building-and-running)
- [Mobile Development](#-mobile-development)
- [Architecture](#-architecture)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Roadmap](#-roadmap)

---

## Why QuiverDB?

| Feature | QuiverDB | FAISS | hnswlib | Pinecone |
|---------|----------|-------|---------|----------|
| Header-only | Yes | No | No | N/A |
| Mobile/Edge | Native | No | Partial | No |
| Dependencies | Zero | Many | Few | Cloud |
| Binary size | <100KB | 200MB+ | ~1MB | N/A |
| GPU (Metal) | Yes | No | No | N/A |

**Perfect for**: Mobile AI apps, offline-first applications, Obsidian/Logseq plugins, edge devices, privacy-sensitive workloads.

## ✨ Features

- **📦 Compact**: Header-only C++ library (~1,300 lines), runs on mobile/edge devices
- **⚡ Fast**: SIMD-optimized distance calculations (ARM NEON, x86 AVX2)
- **📐 Multiple distance metrics**: L2, cosine similarity, dot product
- **🗂️ Multiple index types**: Brute-force (exact), HNSW (approximate), Memory-mapped (large datasets)
- **🎮 GPU acceleration**: Metal (Apple Silicon), CUDA (NVIDIA)
- **🌍 Cross-platform**: Linux, macOS, Windows, iOS, Android (ARM64 and x86_64)
- **🔒 Thread-safe**: Concurrent reads with `std::shared_mutex`
- **🐍 Python bindings**: NumPy integration, GIL-safe operations
- **🏠 Local-first**: No network required, complete privacy

## ⚡ Performance

| Metric | Value | Platform |
|--------|-------|----------|
| L2 Distance (768d) | ~100ns | Apple Silicon |
| Dot Product (768d) | ~95ns | Apple Silicon |
| Cosine Distance (768d) | ~115ns | Apple Silicon |
| SIMD Speedup | 3.8x vs scalar | ARM NEON |
| GPU Speedup | 4.7x at 500K vectors | Metal |
| Throughput | 10M+ ops/sec | M-series Mac |

## 🚀 Quick Start

### Distance Calculations
```cpp
#include "core/distance.h"

// 768-dimensional vectors (e.g., OpenAI embeddings)
float vec_a[768] = {/* ... */};
float vec_b[768] = {/* ... */};

// L2 squared distance (auto-selects SIMD implementation)
float l2_dist = quiverdb::l2_sq(vec_a, vec_b, 768);

// Cosine distance (best for embeddings)
float cos_dist = quiverdb::cosine_distance(vec_a, vec_b, 768);

// Dot product (for maximum inner product search)
float dot = quiverdb::dot_product(vec_a, vec_b, 768);
```

### Vector Store with k-NN Search
```cpp
#include "core/vector_store.h"

// Create a store for 768-dimensional vectors using cosine distance
quiverdb::VectorStore store(768, quiverdb::DistanceMetric::COSINE);

// Add vectors with unique IDs
float doc1[768] = {/* ... */};
float doc2[768] = {/* ... */};
store.add(1, doc1);
store.add(2, doc2);

// Search for 5 nearest neighbors
float query[768] = {/* ... */};
auto results = store.search(query, 5);

for (const auto& result : results) {
    std::cout << "ID: " << result.id
              << " Distance: " << result.distance << "\n";
}
```

### HNSW Index (Approximate Nearest Neighbor)
For large datasets, use `HNSWIndex` for much faster search:

```cpp
#include "core/hnsw_index.h"

// Create HNSW index
quiverdb::HNSWIndex index(768, quiverdb::HNSWDistanceMetric::COSINE, 100000);

// Add vectors
index.add(1, doc1);

// Search
auto results = index.search(query, 5);

// Save and Load
index.save("my_index.bin");
auto loaded_index = quiverdb::HNSWIndex::load("my_index.bin");
```

### Memory-Mapped Vector Store
For datasets larger than RAM, use `MMapVectorStore` for zero-copy file access:

```cpp
#include "core/mmap_vector_store.h"

// Build and save vectors to disk
quiverdb::MMapVectorStoreBuilder builder(768, quiverdb::DistanceMetric::COSINE);
builder.add(1, doc1);
builder.add(2, doc2);
builder.save("vectors.bin");

// Load with memory-mapping (zero-copy, instant load)
quiverdb::MMapVectorStore store("vectors.bin");
auto results = store.search(query, 5);
```

## 🐍 Python Bindings

QuiverDB includes Python bindings for all index types.

### Installation (from source)
```bash
# Clone and build
git clone https://github.com/tsvet01/quiverdb.git
cd quiverdb
cmake -B build -DCMAKE_BUILD_TYPE=Release -DQUIVERDB_BUILD_PYTHON=ON
cmake --build build --parallel

# Add to Python path
export PYTHONPATH=$PWD/build:$PYTHONPATH
```

### Usage
```python
import quiverdb_py as quiverdb
import numpy as np

# Check version
print(quiverdb.__version__)  # "0.1.0"

# === HNSW Index (approximate, fastest for large datasets) ===
index = quiverdb.HNSWIndex(128, quiverdb.HNSWDistanceMetric.COSINE)
vec = np.random.rand(128).astype(np.float32)
index.add(1, vec)
ids, dists = index.search(vec, 10)
index.save("index.bin")

# === VectorStore (exact k-NN, thread-safe) ===
store = quiverdb.VectorStore(128, quiverdb.DistanceMetric.COSINE)
store.add(1, vec)
store.add(2, np.random.rand(128).astype(np.float32))
ids, dists = store.search(vec, 5)

# === MMapVectorStore (memory-mapped, for large datasets) ===
builder = quiverdb.MMapVectorStoreBuilder(128, quiverdb.DistanceMetric.L2)
for i in range(1000):
    builder.add(i, np.random.rand(128).astype(np.float32))
builder.save("vectors.bin")

mmap_store = quiverdb.MMapVectorStore("vectors.bin")  # Instant load
ids, dists = mmap_store.search(vec, 10)
```

## 🔧 Building and Running

### Prerequisites
- CMake 3.20+
- C++20 compiler (Clang 17+ / GCC 11+ / MSVC 19.30+)
- Python 3.9+ (optional, for bindings)
- Git (for fetching dependencies)

```bash
# Build C++ library and Python bindings
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Run tests
ctest --output-on-failure

# Run benchmarks
./bench_distance --benchmark_min_time=0.1s
./bench_vector_store --benchmark_min_time=0.1s
./bench_hnsw_index --benchmark_min_time=0.1s
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `QUIVERDB_BUILD_TESTS` | ON | Build test suite |
| `QUIVERDB_BUILD_BENCHMARKS` | ON | Build benchmarks |
| `QUIVERDB_BUILD_PYTHON` | ON | Build Python bindings |
| `QUIVERDB_BUILD_EXAMPLES` | ON | Build examples |
| `QUIVERDB_BUILD_METAL` | OFF | Build Metal GPU support (macOS) |
| `QUIVERDB_BUILD_CUDA` | OFF | Build CUDA GPU support |

## 📱 Mobile Development

### iOS

```bash
# Build for iOS (requires Xcode)
cmake -B build-ios \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_BUILD_TYPE=Release \
  -GXcode

cmake --build build-ios --config Release -- -sdk iphoneos -arch arm64
```

### Android

```bash
# Build for Android (requires NDK)
export ANDROID_NDK=/path/to/android-ndk

cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-android --parallel
```

Supported Android ABIs: `arm64-v8a` (ARM NEON), `x86_64` (AVX2).

## 🏗️ Architecture

```
quiverdb/
├── src/core/
│   ├── distance.h          # SIMD distance functions (132 lines)
│   ├── vector_store.h      # Thread-safe brute-force store (127 lines)
│   ├── hnsw_index.h        # HNSW approximate search (368 lines)
│   ├── mmap_vector_store.h # Memory-mapped store (239 lines)
│   └── gpu/
│       ├── metal_distance.h # Metal compute shaders
│       └── cuda_distance.cuh # CUDA kernels
├── tests/                   # 31 C++ tests, 28 Python tests
├── benchmarks/              # Google Benchmark suite
└── python/                  # pybind11 bindings
```

## 🔄 CI/CD Pipeline

| Job | Platform | Description |
|-----|----------|-------------|
| build-and-test | Linux (GCC, Clang), macOS, Windows | Core build + tests |
| python-tests | All platforms, Python 3.9/3.11 | Python binding tests |
| sanitizers | Linux | AddressSanitizer, UBSan |
| coverage | Linux | Code coverage + Codecov |
| linux-arm64 | Linux ARM64 (QEMU) | ARM NEON validation |
| ios-build | macOS | iOS arm64 build |
| android-build | Linux | Android arm64-v8a + x86_64 |

## ⚠️ Known Limitations

Current v0.1.0 limitations (documented for transparency):

- **VectorStore pointer lifetime**: `get()` returns a pointer that is invalidated by write operations. Copy data if persistence needed.
- **Brute-force search**: VectorStore uses O(n) brute-force search. Use HNSWIndex for large datasets.
- **No deletion in HNSW**: HNSWIndex doesn't support removing vectors (common HNSW limitation).
- **Single-file persistence**: Each index is a single file; no sharding for very large datasets.
- **GPU requires dim%4==0**: Metal/CUDA kernels use float4 vectorization, requiring dimensions divisible by 4.

## 🗺️ Roadmap

- [ ] PyPI package distribution
- [ ] npm/WebAssembly bindings
- [ ] Product quantization (PQ) for memory efficiency
- [ ] Incremental index updates
- [ ] Multi-vector queries (batch search)
- [ ] Filtering/metadata support

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [hnswlib](https://github.com/nmslib/hnswlib) - HNSW algorithm reference
- [Google Benchmark](https://github.com/google/benchmark) - Benchmarking framework
- [Catch2](https://github.com/catchorg/Catch2) - Testing framework
- [pybind11](https://github.com/pybind/pybind11) - Python bindings

---

See [CHANGELOG.md](CHANGELOG.md) for version history.
