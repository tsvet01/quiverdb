// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#include "core/distance.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

static void BM_L2(benchmark::State& state) {
  const size_t dim = state.range(0);
  std::vector<float> a(dim), b(dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < dim; ++i) { a[i] = dis(gen); b[i] = dis(gen); }

  for (auto _ : state) {
    benchmark::DoNotOptimize(quiverdb::l2_sq(a.data(), b.data(), dim));
  }
  state.SetItemsProcessed(state.iterations() * dim);
  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_DotProduct(benchmark::State& state) {
  const size_t dim = state.range(0);
  std::vector<float> a(dim), b(dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < dim; ++i) { a[i] = dis(gen); b[i] = dis(gen); }

  for (auto _ : state) {
    benchmark::DoNotOptimize(quiverdb::dot_product(a.data(), b.data(), dim));
  }
  state.SetItemsProcessed(state.iterations() * dim);
  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

static void BM_Cosine(benchmark::State& state) {
  const size_t dim = state.range(0);
  std::vector<float> a(dim), b(dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < dim; ++i) { a[i] = dis(gen); b[i] = dis(gen); }

  for (auto _ : state) {
    benchmark::DoNotOptimize(quiverdb::cosine_distance(a.data(), b.data(), dim));
  }
  state.SetItemsProcessed(state.iterations() * dim);
  state.SetBytesProcessed(state.iterations() * dim * sizeof(float) * 2);
}

BENCHMARK(BM_L2)->Arg(128)->Arg(256)->Arg(384)->Arg(512)->Arg(768)->Arg(1536);
BENCHMARK(BM_DotProduct)->Arg(128)->Arg(256)->Arg(384)->Arg(512)->Arg(768)->Arg(1536);
BENCHMARK(BM_Cosine)->Arg(128)->Arg(256)->Arg(384)->Arg(512)->Arg(768)->Arg(1536);

BENCHMARK_MAIN();
