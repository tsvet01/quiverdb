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
      uint64_t dim = 3;  // Fixed: was uint32_t, should be uint64_t per file format
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

  SECTION("Unsupported version throws") {
    const std::string filename = "test_bad_version.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = quiverdb::MMapVectorStore::MAGIC;
      uint32_t version = 99; // Unsupported version
      uint64_t dim = 3;
      uint64_t num_vectors = 0;
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStore(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("File truncated mid-data throws") {
    const std::string filename = "test_truncated_data.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = quiverdb::MMapVectorStore::MAGIC;
      uint32_t version = 1;
      uint64_t dim = 4;
      uint64_t num_vectors = 10; // Claims 10 vectors but won't provide them
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
      // Write only one ID and one vector (should have 10)
      uint64_t id = 1;
      float vec[4] = {1.0f, 2.0f, 3.0f, 4.0f};
      ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
      ofs.write(reinterpret_cast<const char*>(vec), sizeof(vec));
    }
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStore(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Size overflow - huge num_vectors throws") {
    const std::string filename = "test_overflow_num.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = quiverdb::MMapVectorStore::MAGIC;
      uint32_t version = 1;
      uint64_t dim = 4;
      uint64_t num_vectors = SIZE_MAX; // Causes overflow
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStore(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Size overflow - huge dimension throws") {
    const std::string filename = "test_overflow_dim.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = quiverdb::MMapVectorStore::MAGIC;
      uint32_t version = 1;
      uint64_t dim = SIZE_MAX; // Causes overflow
      uint64_t num_vectors = 1;
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStore(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }

  SECTION("Size overflow - combined dim*num_vectors overflow throws") {
    // Test the combined overflow check: num_vectors * dim * sizeof(float) overflows
    // even though each value individually passes earlier checks
    const std::string filename = "test_overflow_combined.bin";
    {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t magic = quiverdb::MMapVectorStore::MAGIC;
      uint32_t version = 1;
      // dim = SIZE_MAX/8 passes dim check (< SIZE_MAX/sizeof(float))
      // num_vectors = 3 passes num_vectors check (< SIZE_MAX/sizeof(uint64_t))
      // But dim * sizeof(float) * num_vectors = (SIZE_MAX/2) * 3 overflows
      uint64_t dim = SIZE_MAX / 8;
      uint64_t num_vectors = 3;
      uint32_t metric = 0;
      uint32_t reserved = 0;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
      ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      ofs.write(reinterpret_cast<const char*>(&num_vectors), sizeof(num_vectors));
      ofs.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
      ofs.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    }
    REQUIRE_THROWS_AS(quiverdb::MMapVectorStore(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }
}

TEST_CASE("MMapVectorStore - search validation", "[mmap]") {
  const std::string filename = "test_mmap_search_validation.bin";
  std::filesystem::remove(filename);

  quiverdb::MMapVectorStoreBuilder builder(4, quiverdb::DistanceMetric::L2);
  float vec[] = {1.0f, 0.0f, 0.0f, 0.0f};
  builder.add(1, vec);
  builder.save(filename);

  {
    quiverdb::MMapVectorStore store(filename);

    SECTION("Search with null query throws") {
      REQUIRE_THROWS_AS(store.search(nullptr, 1), std::invalid_argument);
    }

    SECTION("Search with k=0 throws") {
      float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
      REQUIRE_THROWS_AS(store.search(query, 0), std::invalid_argument);
    }
  }  // Scope ensures store is destroyed and file unmapped before removal (Windows file locking)

  std::filesystem::remove(filename);
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

  {
    quiverdb::MMapVectorStore store(filename);

    float query[] = {3.0f, 0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 2);

    // vec1 and vec2 should be closest (same direction)
    REQUIRE(results.size() == 2);
    REQUIRE((results[0].id == 1 || results[0].id == 2));
    REQUIRE((results[1].id == 1 || results[1].id == 2));
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-5f));
  }  // store destructor unmaps file before removal

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

  {
    quiverdb::MMapVectorStore store(filename);

    float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 1);

    // vec2 should be first (highest dot product = smallest negative distance)
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 2);
  }  // store destructor unmaps file before removal

  std::filesystem::remove(filename);
}
