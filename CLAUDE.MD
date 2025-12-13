# quiverdb

## Overview
Embeddable vector database for edge AI. Header-only C++20, SIMD-optimized, cross-platform.

## Current Status: v0.1.0

### Features
| Feature | Status |
|---------|--------|
| Distance Functions | L2, Cosine, Dot (ARM NEON, x86 AVX2) |
| VectorStore | In-memory k-NN, thread-safe |
| HNSWIndex | Approximate NN with save/load |
| MMapVectorStore | Memory-mapped zero-copy |
| GPU Acceleration | Metal (Apple Silicon), CUDA (NVIDIA) |
| Python Bindings | pybind11 + NumPy |
| Mobile | iOS arm64, Android arm64-v8a/x86_64 |

### Test Coverage
- 31 C++ test cases (131k+ assertions)
- 28 Python tests
- 3 GPU tests (Metal)
- Sanitizers: ASan + UBSan clean

### Performance
- CPU SIMD: 3.8x speedup vs scalar (768d)
- L2 distance: ~100ns (768d, Apple Silicon)
- GPU: 3.9x speedup at 500k vectors (persistent buffers)

## Structure
```
src/core/
├── distance.h           # SIMD distance functions
├── vector_store.h       # Brute-force k-NN
├── hnsw_index.h         # HNSW approximate NN
├── mmap_vector_store.h  # Memory-mapped store
└── gpu/
    ├── metal_distance.h # Metal compute
    └── cuda_distance.cuh # CUDA kernels
```
~1,000 lines of core code total.

## Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cd build && ctest --output-on-failure
```

## CI/CD (10 jobs)
- Linux: GCC, Clang
- macOS: Apple Clang (ARM NEON)
- Windows: MSVC (AVX2)
- ARM64: QEMU emulation
- iOS: Xcode arm64
- Android: NDK arm64-v8a, x86_64
- Python: 3 platforms x 2 versions
- Sanitizers: ASan, UBSan
- Coverage: Codecov

## API
```cpp
// Distance
float d = quiverdb::l2_sq(a, b, dim);

// VectorStore
quiverdb::VectorStore store(768, quiverdb::DistanceMetric::COSINE);
store.add(id, vec);
auto results = store.search(query, k);

// HNSWIndex
quiverdb::HNSWIndex idx(768, quiverdb::HNSWDistanceMetric::COSINE, 100000);
idx.add(id, vec);
idx.save("index.bin");

// GPU (Metal)
auto& gpu = quiverdb::gpu::MetalCompute::get();
auto buf = gpu.upload(vectors, n, dim);
auto dists = gpu.search(query, buf, dim, n, quiverdb::gpu::MetalMetric::L2);
```

## Demo: Obsidian Semantic Search
Location: `/Users/anton/code/search/`

A working semantic search tool built on QuiverDB demonstrating real-world usage:
```bash
python3 search.py index ~/path/to/vault   # Index notes
python3 search.py find "your query"       # Search
python3 search.py interactive             # REPL mode
```

## Roadmap

### Near-term (v0.2.0)
- [ ] PyPI package distribution (`pip install quiverdb`)
- [ ] Incremental index updates (add/remove without full rebuild)
- [ ] Batch search API (multiple queries in one call)
- [ ] Metadata/filtering support (search with constraints)

### Medium-term (v0.3.0)
- [ ] Product quantization (PQ) for 4-8x memory reduction
- [ ] npm/WebAssembly bindings for browser/Node.js
- [ ] Disk-based HNSW (mmap + HNSW hybrid)
- [ ] Multi-vector queries (document retrieval)

### Long-term
- [ ] Distributed sharding for billion-scale
- [ ] CUDA optimizations (tensor cores)
- [ ] Swift package for native iOS/macOS
- [ ] Real-time index updates (concurrent insert/search)

## Known Limitations
- No deletion in HNSWIndex (rebuild required)
- GPU requires dim % 4 == 0
- VectorStore `get()` pointer invalidated by writes
- Single-file persistence (no sharding)
