#include "core/hnsw_index.h"
#include "core/vector_store.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

// Random data generator
class RandomData {
public:
  RandomData(size_t dim, size_t count, uint32_t seed = 42)
      : dim_(dim), gen_(seed), dis_(-1.0f, 1.0f) {
    vectors_.resize(count);
    for (auto& vec : vectors_) {
      vec.resize(dim_);
      for (float& v : vec) {
        v = dis_(gen_);
      }
    }
  }

  const float* vector(size_t idx) const { return vectors_[idx].data(); }
  size_t count() const { return vectors_.size(); }
  size_t dim() const { return dim_; }

  std::vector<float> random_query() {
    std::vector<float> q(dim_);
    for (float& v : q) {
      v = dis_(gen_);
    }
    return q;
  }

private:
  size_t dim_;
  std::mt19937 gen_;
  std::uniform_real_distribution<float> dis_;
  std::vector<std::vector<float>> vectors_;
};

// Benchmark: HNSW insertion
static void BM_HNSW_Insert(benchmark::State& state) {
  const size_t dim = 128;
  const size_t count = state.range(0);

  for (auto _ : state) {
    state.PauseTiming();
    RandomData data(dim, count);
    quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, count);
    state.ResumeTiming();

    for (size_t i = 0; i < count; ++i) {
      index.add(i, data.vector(i));
    }
  }

  state.SetItemsProcessed(state.iterations() * count);
}

BENCHMARK(BM_HNSW_Insert)
    ->Arg(1000)
    ->Arg(5000)
    ->Arg(10000)
    ->Unit(benchmark::kMillisecond);

// Benchmark: HNSW search with varying ef
static void BM_HNSW_Search(benchmark::State& state) {
  const size_t dim = 128;
  const size_t count = 10000;
  const size_t k = 10;
  const size_t ef = state.range(0);

  // Build index once
  RandomData data(dim, count);
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, count, 16, 200);
  for (size_t i = 0; i < count; ++i) {
    index.add(i, data.vector(i));
  }
  index.set_ef_search(ef);

  for (auto _ : state) {
    auto query = data.random_query();
    auto results = index.search(query.data(), k);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_HNSW_Search)
    ->Arg(10)   // Low ef (fast, lower recall)
    ->Arg(50)   // Medium ef
    ->Arg(100)  // High ef (slower, better recall)
    ->Arg(200)  // Very high ef
    ->Unit(benchmark::kMicrosecond);

// Benchmark: HNSW vs Brute Force comparison
static void BM_BruteForce_Search(benchmark::State& state) {
  const size_t dim = 128;
  const size_t count = state.range(0);
  const size_t k = 10;

  // Build index once
  RandomData data(dim, count);
  quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::L2);
  for (size_t i = 0; i < count; ++i) {
    store.add(i, data.vector(i));
  }

  for (auto _ : state) {
    auto query = data.random_query();
    auto results = store.search(query.data(), k);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_BruteForce_Search)
    ->Arg(1000)
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(50000)
    ->Unit(benchmark::kMicrosecond);

static void BM_HNSW_Search_VaryingN(benchmark::State& state) {
  const size_t dim = 128;
  const size_t count = state.range(0);
  const size_t k = 10;
  const size_t ef = 50;

  // Build index once
  RandomData data(dim, count);
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, count, 16, 200);
  for (size_t i = 0; i < count; ++i) {
    index.add(i, data.vector(i));
  }
  index.set_ef_search(ef);

  for (auto _ : state) {
    auto query = data.random_query();
    auto results = index.search(query.data(), k);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_HNSW_Search_VaryingN)
    ->Arg(1000)
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(50000)
    ->Unit(benchmark::kMicrosecond);

// Benchmark: Different dimensions
static void BM_HNSW_Search_Dimension(benchmark::State& state) {
  const size_t dim = state.range(0);
  const size_t count = 10000;
  const size_t k = 10;
  const size_t ef = 50;

  // Build index once
  RandomData data(dim, count);
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, count, 16, 200);
  for (size_t i = 0; i < count; ++i) {
    index.add(i, data.vector(i));
  }
  index.set_ef_search(ef);

  for (auto _ : state) {
    auto query = data.random_query();
    auto results = index.search(query.data(), k);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_HNSW_Search_Dimension)
    ->Arg(64)    // Small embeddings
    ->Arg(128)   // Medium
    ->Arg(384)   // SentenceTransformers
    ->Arg(768)   // BERT/OpenAI
    ->Arg(1536)  // OpenAI ada-002
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
