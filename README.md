# quiverdb

Embeddable vector database for edge AI. Lightning-fast semantic search that runs anywhere.

## Features
- **Compact**: Header-only C++ library, runs on mobile/edge devices
- **Fast**: SIMD-optimized distance calculations (ARM NEON, x86 AVX2)
- **Multiple distance metrics**: L2, cosine similarity, dot product
- **In-memory vector store**: Add, search, and manage vectors with k-NN search
- **Cross-platform**: Linux, macOS, Windows, iOS, Android
- **Local-first**: No network required, complete privacy

## Quick Start

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

## Building and Running
Prerequisites
- CMake 3.20+
- C++20 compiler (Clang 17+ / GCC 11+ / MSVC 19.30+)
- Git (for fetching dependencies)

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Run tests
./test_distance
./test_vector_store

# Run benchmarks
./bench_distance --benchmark_min_time=0.1s
./bench_vector_store --benchmark_min_time=0.1s
```

## Performance

Benchmarks on x86_64 (Intel/AMD) with AVX2:
- **Dot Product**: ~14-16 billion elements/second (768d)
- **L2 Distance**: ~10-11 billion elements/second (768d)
- **Cosine Distance**: ~7-8 billion elements/second (768d)

Performance scales linearly with vector dimensions and benefits from SIMD optimizations.