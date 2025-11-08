#include "core/distance.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

// Benchmark scalar implementation
static void BM_L2_Scalar(benchmark::State &state) {
  const size_t dim = state.range(0);
  std::vector<float> a(dim);
  std::vector<float> b(dim);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (size_t i = 0; i < dim; ++i) {
    a[i] = dis(gen);
    b[i] = dis(gen);
  }

  for (auto _ : state) {
    float result = quiverdb::l2_sq_scalar(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * dim);
  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

#ifdef QUIVER_ARM_NEON
// Benchmark NEON implementation
static void BM_L2_NEON(benchmark::State &state) {
  const size_t dim = state.range(0);
  std::vector<float> a(dim);
  std::vector<float> b(dim);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (size_t i = 0; i < dim; ++i) {
    a[i] = dis(gen);
    b[i] = dis(gen);
  }

  for (auto _ : state) {
    float result = quiverdb::l2_sq_neon(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * dim);
  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}
#endif

// Benchmark public API (auto-selects best implementation)
static void BM_L2_API(benchmark::State &state) {
  const size_t dim = state.range(0);
  std::vector<float> a(dim);
  std::vector<float> b(dim);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (size_t i = 0; i < dim; ++i) {
    a[i] = dis(gen);
    b[i] = dis(gen);
  }

  for (auto _ : state) {
    float result = quiverdb::l2_sq(a.data(), b.data(), dim);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * dim);
  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

// Register benchmarks with common embedding dimensions
BENCHMARK(BM_L2_Scalar)
    ->Arg(128)
    ->Arg(256)
    ->Arg(384)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1536);

#ifdef QUIVER_ARM_NEON
BENCHMARK(BM_L2_NEON)
    ->Arg(128)
    ->Arg(256)
    ->Arg(384)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1536);
#endif

BENCHMARK(BM_L2_API)
    ->Arg(128)
    ->Arg(256)
    ->Arg(384)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1536);

BENCHMARK_MAIN();
