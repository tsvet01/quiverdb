#include "core/vector_store.h"
#include <benchmark/benchmark.h>
#include <random>
#include <vector>

// ============================================================================
// VectorStore Add Benchmarks
// ============================================================================

static void BM_VectorStore_Add(benchmark::State &state) {
  const size_t dim = state.range(0);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  for (auto _ : state) {
    state.PauseTiming();
    quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::L2);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
      vec[i] = dis(gen);
    }
    state.ResumeTiming();

    store.add(1, vec.data());
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_VectorStore_Add)
    ->Arg(128)
    ->Arg(256)
    ->Arg(384)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1536);

// ============================================================================
// VectorStore Search Benchmarks - L2 Distance
// ============================================================================

static void BM_VectorStore_Search_L2(benchmark::State &state) {
  const size_t dim = 768;
  const size_t num_vectors = state.range(0);
  const size_t k = 10;

  quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::L2);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Add vectors
  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dis(gen);
    }
    store.add(i, vec.data());
  }

  // Prepare query
  std::vector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = dis(gen);
  }

  for (auto _ : state) {
    auto results = store.search(query.data(), k);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetLabel(std::to_string(num_vectors) + " vectors, k=" + std::to_string(k));
}

BENCHMARK(BM_VectorStore_Search_L2)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(5000)
    ->Arg(10000);

// ============================================================================
// VectorStore Search Benchmarks - Cosine Distance
// ============================================================================

static void BM_VectorStore_Search_Cosine(benchmark::State &state) {
  const size_t dim = 768;
  const size_t num_vectors = state.range(0);
  const size_t k = 10;

  quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::COSINE);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Add vectors
  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dis(gen);
    }
    store.add(i, vec.data());
  }

  // Prepare query
  std::vector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = dis(gen);
  }

  for (auto _ : state) {
    auto results = store.search(query.data(), k);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetLabel(std::to_string(num_vectors) + " vectors, k=" + std::to_string(k));
}

BENCHMARK(BM_VectorStore_Search_Cosine)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(5000)
    ->Arg(10000);

// ============================================================================
// VectorStore Search Benchmarks - Varying k
// ============================================================================

static void BM_VectorStore_Search_VaryingK(benchmark::State &state) {
  const size_t dim = 768;
  const size_t num_vectors = 1000;
  const size_t k = state.range(0);

  quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::L2);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Add vectors
  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dis(gen);
    }
    store.add(i, vec.data());
  }

  // Prepare query
  std::vector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = dis(gen);
  }

  for (auto _ : state) {
    auto results = store.search(query.data(), k);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetLabel("1000 vectors, k=" + std::to_string(k));
}

BENCHMARK(BM_VectorStore_Search_VaryingK)
    ->Arg(1)
    ->Arg(5)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100);

// ============================================================================
// VectorStore Search Benchmarks - Varying Dimensions
// ============================================================================

static void BM_VectorStore_Search_VaryingDim(benchmark::State &state) {
  const size_t dim = state.range(0);
  const size_t num_vectors = 1000;
  const size_t k = 10;

  quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::L2);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Add vectors
  for (size_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dis(gen);
    }
    store.add(i, vec.data());
  }

  // Prepare query
  std::vector<float> query(dim);
  for (size_t j = 0; j < dim; ++j) {
    query[j] = dis(gen);
  }

  for (auto _ : state) {
    auto results = store.search(query.data(), k);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * num_vectors);
  state.SetLabel(std::to_string(dim) + "d, 1000 vectors, k=10");
}

BENCHMARK(BM_VectorStore_Search_VaryingDim)
    ->Arg(128)
    ->Arg(256)
    ->Arg(384)
    ->Arg(512)
    ->Arg(768)
    ->Arg(1536);

BENCHMARK_MAIN();
