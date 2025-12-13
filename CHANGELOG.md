# Changelog

All notable changes to QuiverDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-07

### Added
- Core distance functions with SIMD optimization (ARM NEON, x86 AVX2)
  - L2 squared distance
  - Cosine similarity/distance
  - Dot product
- GPU acceleration (Metal for Apple Silicon, CUDA for NVIDIA)
  - Persistent buffer API for zero-copy repeated queries
  - 3.9x speedup at 500k vectors
- In-memory VectorStore with k-NN brute-force search
- HNSW index for approximate nearest neighbor search
  - Configurable M, ef_construction, ef_search parameters
  - Binary serialization (save/load)
- Memory-mapped VectorStore for large datasets
  - Zero-copy file access
  - Atomic save operations
- Python bindings via pybind11
  - NumPy array support
  - All index types and distance metrics
- Comprehensive test suite (35 C++ tests, 28 Python tests)
- Google Benchmark performance tests
- Multi-platform CI/CD (Linux, macOS, Windows, iOS, Android)
  - GCC, Clang, MSVC compilers
  - Native ARM64 testing
  - iOS arm64 builds (Xcode)
  - Android arm64-v8a and x86_64 builds (NDK)
  - AddressSanitizer and UBSan checks
  - Code coverage reporting

### Performance
- 3.8x speedup with ARM NEON vs scalar (768d vectors)
- ~100ns L2 distance per operation (768d, Apple Silicon)
- Sub-millisecond search latency for 10k vectors
