#include "core/mmap_vector_store.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <random>
#include <vector>

using Catch::Approx;

TEST_CASE("MMapVectorStoreBuilder - construction", "[mmap][builder]") {
  SECTION("Valid dimension") {
    REQUIRE_NOTHROW(quiverdb::MMapVectorStoreBuilder(768));
    REQUIRE_NOTHROW(quiverdb::MMapVectorStoreBuilder(128, quiverdb::DistanceMetric::COSINE));
  }

  SECTION("Zero dimension throws") {
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStoreBuilder(0), std::invalid_argument);
  }
}

TEST_CASE("MMapVectorStoreBuilder - add vectors", "[mmap][builder]") {
  quiverdb::MMapVectorStoreBuilder builder(3);

  SECTION("Add single vector") {
    float vec[] = {1.0f, 2.0f, 3.0f};
    REQUIRE_NOTHROW(builder.add(1, vec));
    REQUIRE(builder.size() == 1);
  }

  SECTION("Add multiple vectors") {
    float vec1[] = {1.0f, 2.0f, 3.0f};
    float vec2[] = {4.0f, 5.0f, 6.0f};

    builder.add(1, vec1);
    builder.add(2, vec2);

    REQUIRE(builder.size() == 2);
  }

  SECTION("Null vector throws") {
    REQUIRE_THROWS_AS(builder.add(1, nullptr), std::invalid_argument);
  }

  SECTION("Duplicate ID throws") {
    float vec1[] = {1.0f, 2.0f, 3.0f};
    float vec2[] = {4.0f, 5.0f, 6.0f};

    builder.add(1, vec1);
    REQUIRE_THROWS_AS(builder.add(1, vec2), std::invalid_argument);
  }
}

TEST_CASE("MMapVectorStore - save and load", "[mmap]") {
  const std::string filename = "test_mmap_store.bin";

  // Cleanup before test
  std::filesystem::remove(filename);

  constexpr size_t dim = 4;
  quiverdb::MMapVectorStoreBuilder builder(dim, quiverdb::DistanceMetric::L2);

  // Add test vectors
  float vec1[] = {1.0f, 0.0f, 0.0f, 0.0f};
  float vec2[] = {0.0f, 1.0f, 0.0f, 0.0f};
  float vec3[] = {0.0f, 0.0f, 1.0f, 0.0f};

  builder.add(10, vec1);
  builder.add(20, vec2);
  builder.add(30, vec3);

  SECTION("Save and load successfully") {
    REQUIRE_NOTHROW(builder.save(filename));

    quiverdb::MMapVectorStore store(filename);

    REQUIRE(store.size() == 3);
    REQUIRE(store.dimension() == dim);
    REQUIRE(store.metric() == quiverdb::DistanceMetric::L2);
  }

  SECTION("Get vector by ID") {
    builder.save(filename);
    quiverdb::MMapVectorStore store(filename);

    const float* retrieved = store.get(10);
    REQUIRE(retrieved != nullptr);
    REQUIRE(retrieved[0] == 1.0f);
    REQUIRE(retrieved[1] == 0.0f);
    REQUIRE(retrieved[2] == 0.0f);
    REQUIRE(retrieved[3] == 0.0f);
  }

  SECTION("Get non-existent ID returns nullptr") {
    builder.save(filename);
    quiverdb::MMapVectorStore store(filename);

    REQUIRE(store.get(999) == nullptr);
  }

  SECTION("Contains works correctly") {
    builder.save(filename);
    quiverdb::MMapVectorStore store(filename);

    REQUIRE(store.contains(10));
    REQUIRE(store.contains(20));
    REQUIRE(store.contains(30));
    REQUIRE_FALSE(store.contains(999));
  }

  SECTION("Search finds nearest neighbors") {
    builder.save(filename);
    quiverdb::MMapVectorStore store(filename);

    float query[] = {0.9f, 0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 1);

    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 10);  // Closest to vec1
  }

  SECTION("Search with k > size returns all") {
    builder.save(filename);
    quiverdb::MMapVectorStore store(filename);

    float query[] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 100);

    REQUIRE(results.size() == 3);
  }

  // Cleanup
  std::filesystem::remove(filename);
}

TEST_CASE("MMapVectorStore - error handling", "[mmap]") {
  SECTION("Non-existent file throws") {
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStore("nonexistent_file.bin"), std::runtime_error);
  }

  SECTION("Invalid magic throws") {
    const std::string filename = "test_bad_magic.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t bad_magic = 0xDEADBEEF;
      ofs.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
    }
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStore(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Truncated file throws") {
    const std::string filename = "test_truncated.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      // Write only partial header
      uint32_t magic = quiverdb::MMapVectorStore::MAGIC;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    }
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStore(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Invalid metric value throws") {
    const std::string filename = "test_bad_metric.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = quiverdb::MMapVectorStore::MAGIC;
      uint32_t version = 1;
      uint32_t dim = 3;
      uint64_t num_vectors = 0;
      uint32_t bad_metric = 999; // Invalid metric value
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&bad_metric), sizeof(bad_metric));
    }
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStore(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }
}

TEST_CASE("MMapVectorStore - large scale", "[mmap][stress]") {
  const std::string filename = "test_mmap_large.bin";
  std::filesystem::remove(filename);

  constexpr size_t dim = 128;
  constexpr size_t num_vectors = 1000;

  quiverdb::MMapVectorStoreBuilder builder(dim, quiverdb::DistanceMetric::COSINE);
  builder.reserve(num_vectors);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<std::vector<float>> all_vectors(num_vectors);
  for (uint64_t i = 0; i < num_vectors; ++i) {
    all_vectors[i].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      all_vectors[i][j] = dis(gen);
    }
    builder.add(i, all_vectors[i].data());
  }

  builder.save(filename);

  SECTION("Load and search") {
    quiverdb::MMapVectorStore store(filename);

    REQUIRE(store.size() == num_vectors);
    REQUIRE(store.dimension() == dim);

    // Search should find exact matches
    for (size_t i = 0; i < 10; ++i) {
      auto results = store.search(all_vectors[i].data(), 1);
      REQUIRE(results.size() == 1);
      REQUIRE(results[0].id == i);
      REQUIRE(results[0].distance == Approx(0.0f).margin(1e-5f));
    }
  }

  SECTION("Vectors are preserved") {
    quiverdb::MMapVectorStore store(filename);

    for (size_t i = 0; i < num_vectors; ++i) {
      const float* retrieved = store.get(i);
      REQUIRE(retrieved != nullptr);
      for (size_t j = 0; j < dim; ++j) {
        REQUIRE(retrieved[j] == all_vectors[i][j]);
      }
    }
  }

  std::filesystem::remove(filename);
}

TEST_CASE("MMapVectorStore - cosine metric", "[mmap]") {
  const std::string filename = "test_mmap_cosine.bin";
  std::filesystem::remove(filename);

  quiverdb::MMapVectorStoreBuilder builder(4, quiverdb::DistanceMetric::COSINE);

  // Same direction, different magnitudes
  float vec1[] = {1.0f, 0.0f, 0.0f, 0.0f};
  float vec2[] = {2.0f, 0.0f, 0.0f, 0.0f};
  float vec3[] = {0.0f, 1.0f, 0.0f, 0.0f};  // Orthogonal

  builder.add(1, vec1);
  builder.add(2, vec2);
  builder.add(3, vec3);
  builder.save(filename);

  quiverdb::MMapVectorStore store(filename);

  float query[] = {3.0f, 0.0f, 0.0f, 0.0f};
  auto results = store.search(query, 2);

  // vec1 and vec2 should be closest (same direction)
  REQUIRE(results.size() == 2);
  REQUIRE((results[0].id == 1 || results[0].id == 2));
  REQUIRE((results[1].id == 1 || results[1].id == 2));
  REQUIRE(results[0].distance == Approx(0.0f).margin(1e-5f));

  std::filesystem::remove(filename);
}

TEST_CASE("MMapVectorStore - dot product metric", "[mmap]") {
  const std::string filename = "test_mmap_dot.bin";
  std::filesystem::remove(filename);

  quiverdb::MMapVectorStoreBuilder builder(4, quiverdb::DistanceMetric::DOT);

  float vec1[] = {1.0f, 0.0f, 0.0f, 0.0f};
  float vec2[] = {2.0f, 0.0f, 0.0f, 0.0f};  // Higher dot product
  float vec3[] = {0.0f, 1.0f, 0.0f, 0.0f};

  builder.add(1, vec1);
  builder.add(2, vec2);
  builder.add(3, vec3);
  builder.save(filename);

  quiverdb::MMapVectorStore store(filename);

  float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
  auto results = store.search(query, 1);

  // vec2 should be first (highest dot product = smallest negative distance)
  REQUIRE(results.size() == 1);
  REQUIRE(results[0].id == 2);

  std::filesystem::remove(filename);
}
