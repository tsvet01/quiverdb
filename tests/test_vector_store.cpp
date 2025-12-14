#include "core/vector_store.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <thread>
#include <vector>

using Catch::Approx;

TEST_CASE("VectorStore - construction", "[vector_store]") {
  SECTION("Valid dimension") {
    REQUIRE_NOTHROW(quiverdb::VectorStore(768));
    REQUIRE_NOTHROW(quiverdb::VectorStore(128, quiverdb::DistanceMetric::COSINE));
  }

  SECTION("Zero dimension throws") {
    REQUIRE_THROWS_AS(quiverdb::VectorStore(0), std::invalid_argument);
  }

  SECTION("Check initial state") {
    quiverdb::VectorStore store(768);
    REQUIRE(store.size() == 0);
    REQUIRE(store.dimension() == 768);
    REQUIRE(store.metric() == quiverdb::DistanceMetric::L2);
  }
}

TEST_CASE("VectorStore - add vectors", "[vector_store]") {
  quiverdb::VectorStore store(3);

  SECTION("Add single vector") {
    float vec[] = {1.0f, 2.0f, 3.0f};
    REQUIRE_NOTHROW(store.add(1, vec));
    REQUIRE(store.size() == 1);
    REQUIRE(store.contains(1));
  }

  SECTION("Add multiple vectors") {
    float vec1[] = {1.0f, 2.0f, 3.0f};
    float vec2[] = {4.0f, 5.0f, 6.0f};
    float vec3[] = {7.0f, 8.0f, 9.0f};

    store.add(1, vec1);
    store.add(2, vec2);
    store.add(3, vec3);

    REQUIRE(store.size() == 3);
    REQUIRE(store.contains(1));
    REQUIRE(store.contains(2));
    REQUIRE(store.contains(3));
  }

  SECTION("Add with null vector throws") {
    REQUIRE_THROWS_AS(store.add(1, nullptr), std::invalid_argument);
  }

  SECTION("Add duplicate ID throws") {
    float vec1[] = {1.0f, 2.0f, 3.0f};
    float vec2[] = {4.0f, 5.0f, 6.0f};

    store.add(1, vec1);
    REQUIRE_THROWS_AS(store.add(1, vec2), std::invalid_argument);
  }
}

TEST_CASE("VectorStore - get vectors", "[vector_store]") {
  quiverdb::VectorStore store(3);

  SECTION("Get existing vector") {
    float vec[] = {1.0f, 2.0f, 3.0f};
    store.add(42, vec);

    const float* retrieved = store.get(42);
    REQUIRE(retrieved != nullptr);
    REQUIRE(retrieved[0] == 1.0f);
    REQUIRE(retrieved[1] == 2.0f);
    REQUIRE(retrieved[2] == 3.0f);
  }

  SECTION("Get non-existent vector") {
    const float* retrieved = store.get(999);
    REQUIRE(retrieved == nullptr);
  }

  SECTION("Vector data is independent") {
    float vec[] = {1.0f, 2.0f, 3.0f};
    store.add(1, vec);

    // Modify original
    vec[0] = 999.0f;

    // Stored vector should be unchanged
    const float* retrieved = store.get(1);
    REQUIRE(retrieved[0] == 1.0f);
  }
}

TEST_CASE("VectorStore - remove vectors", "[vector_store]") {
  quiverdb::VectorStore store(3);
  float vec1[] = {1.0f, 2.0f, 3.0f};
  float vec2[] = {4.0f, 5.0f, 6.0f};
  float vec3[] = {7.0f, 8.0f, 9.0f};

  store.add(1, vec1);
  store.add(2, vec2);
  store.add(3, vec3);

  SECTION("Remove existing vector") {
    REQUIRE(store.remove(2) == true);
    REQUIRE(store.size() == 2);
    REQUIRE(!store.contains(2));
    REQUIRE(store.contains(1));
    REQUIRE(store.contains(3));
  }

  SECTION("Remove non-existent vector") {
    REQUIRE(store.remove(999) == false);
    REQUIRE(store.size() == 3);
  }

  SECTION("Remove all vectors") {
    store.remove(1);
    store.remove(2);
    store.remove(3);
    REQUIRE(store.size() == 0);
  }
}

TEST_CASE("VectorStore - clear", "[vector_store]") {
  quiverdb::VectorStore store(3);
  float vec1[] = {1.0f, 2.0f, 3.0f};
  float vec2[] = {4.0f, 5.0f, 6.0f};

  store.add(1, vec1);
  store.add(2, vec2);

  store.clear();
  REQUIRE(store.size() == 0);
  REQUIRE(!store.contains(1));
  REQUIRE(!store.contains(2));
}

TEST_CASE("VectorStore - search with L2 distance", "[vector_store][search]") {
  quiverdb::VectorStore store(3, quiverdb::DistanceMetric::L2);

  // Add test vectors
  float vec1[] = {1.0f, 0.0f, 0.0f};
  float vec2[] = {0.0f, 1.0f, 0.0f};
  float vec3[] = {0.0f, 0.0f, 1.0f};
  float vec4[] = {1.0f, 1.0f, 0.0f};

  store.add(1, vec1);
  store.add(2, vec2);
  store.add(3, vec3);
  store.add(4, vec4);

  SECTION("Search for nearest neighbor") {
    float query[] = {0.9f, 0.0f, 0.0f};
    auto results = store.search(query, 1);

    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 1); // Closest to vec1
  }

  SECTION("Search for k nearest neighbors") {
    float query[] = {0.5f, 0.5f, 0.0f};
    auto results = store.search(query, 2);

    REQUIRE(results.size() == 2);
    // vec1, vec2, and vec4 all have same distance 0.5 to query
    // So any two of them could be returned
    REQUIRE((results[0].id == 1 || results[0].id == 2 || results[0].id == 4));
    REQUIRE((results[1].id == 1 || results[1].id == 2 || results[1].id == 4));
    REQUIRE(results[0].distance == Approx(0.5f).epsilon(0.001));
    REQUIRE(results[1].distance == Approx(0.5f).epsilon(0.001));
  }

  SECTION("Search with k larger than store size") {
    float query[] = {0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 100);

    REQUIRE(results.size() == 4);
  }

  SECTION("Search with null query throws") {
    REQUIRE_THROWS_AS(store.search(nullptr, 1), std::invalid_argument);
  }

  SECTION("Search with k=0 throws") {
    float query[] = {0.0f, 0.0f, 0.0f};
    REQUIRE_THROWS_AS(store.search(query, 0), std::invalid_argument);
  }

  SECTION("Results are sorted by distance") {
    float query[] = {0.0f, 0.0f, 0.0f};
    auto results = store.search(query, 4);

    REQUIRE(results.size() == 4);
    for (size_t i = 1; i < results.size(); ++i) {
      REQUIRE(results[i-1].distance <= results[i].distance);
    }
  }
}

TEST_CASE("VectorStore - search with cosine distance", "[vector_store][search]") {
  quiverdb::VectorStore store(3, quiverdb::DistanceMetric::COSINE);

  // Add test vectors
  float vec1[] = {1.0f, 0.0f, 0.0f};
  float vec2[] = {2.0f, 0.0f, 0.0f}; // Same direction as vec1
  float vec3[] = {0.0f, 1.0f, 0.0f}; // Orthogonal to vec1
  float vec4[] = {-1.0f, 0.0f, 0.0f}; // Opposite to vec1

  store.add(1, vec1);
  store.add(2, vec2);
  store.add(3, vec3);
  store.add(4, vec4);

  SECTION("Search finds same direction vectors") {
    float query[] = {3.0f, 0.0f, 0.0f};
    auto results = store.search(query, 2);

    REQUIRE(results.size() == 2);
    // Should find vec1 and vec2 (same direction)
    REQUIRE((results[0].id == 1 || results[0].id == 2));
    REQUIRE((results[1].id == 1 || results[1].id == 2));
  }

  SECTION("Cosine distance is magnitude-independent") {
    float query[] = {10.0f, 0.0f, 0.0f};
    auto results = store.search(query, 1);

    // Both vec1 and vec2 point in same direction, should get either
    REQUIRE((results[0].id == 1 || results[0].id == 2));
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-6));
  }
}

TEST_CASE("VectorStore - search with dot product", "[vector_store][search]") {
  quiverdb::VectorStore store(3, quiverdb::DistanceMetric::DOT);

  // Add test vectors
  float vec1[] = {1.0f, 0.0f, 0.0f};
  float vec2[] = {2.0f, 0.0f, 0.0f};
  float vec3[] = {0.0f, 1.0f, 0.0f};

  store.add(1, vec1);
  store.add(2, vec2);
  store.add(3, vec3);

  SECTION("Maximum inner product search") {
    float query[] = {1.0f, 0.0f, 0.0f};
    auto results = store.search(query, 1);

    // vec2 has largest dot product with query
    REQUIRE(results[0].id == 2);
    REQUIRE(results[0].distance == Approx(-2.0f).epsilon(0.001));
  }
}

TEST_CASE("VectorStore - high dimensional vectors", "[vector_store][search]") {
  constexpr size_t dim = 768;
  quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::COSINE);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  // Add 100 random vectors
  for (uint64_t i = 0; i < 100; ++i) {
    std::vector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dis(gen);
    }
    store.add(i, vec.data());
  }

  REQUIRE(store.size() == 100);

  SECTION("Search returns correct number of results") {
    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
      query[j] = dis(gen);
    }

    auto results = store.search(query.data(), 10);
    REQUIRE(results.size() == 10);

    // Results should be sorted
    for (size_t i = 1; i < results.size(); ++i) {
      REQUIRE(results[i-1].distance <= results[i].distance);
    }
  }

  SECTION("Search for exact match") {
    // Query with a vector that exists in the store
    const float* vec = store.get(42);
    REQUIRE(vec != nullptr);

    std::vector<float> query(vec, vec + dim);
    auto results = store.search(query.data(), 1);

    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 42);
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-5));
  }
}

TEST_CASE("VectorStore - update vector", "[vector_store]") {
  quiverdb::VectorStore store(3);
  float vec1[] = {1.0f, 2.0f, 3.0f};
  float vec2[] = {4.0f, 5.0f, 6.0f};

  store.add(1, vec1);

  SECTION("Update existing vector") {
    REQUIRE(store.update(1, vec2) == true);

    const float* retrieved = store.get(1);
    REQUIRE(retrieved != nullptr);
    REQUIRE(retrieved[0] == 4.0f);
    REQUIRE(retrieved[1] == 5.0f);
    REQUIRE(retrieved[2] == 6.0f);
  }

  SECTION("Update non-existent vector") {
    REQUIRE(store.update(999, vec2) == false);
  }

  SECTION("Update with null vector throws") {
    REQUIRE_THROWS_AS(store.update(1, nullptr), std::invalid_argument);
  }
}

TEST_CASE("VectorStore - reserve", "[vector_store]") {
  quiverdb::VectorStore store(3);

  SECTION("Reserve space") {
    REQUIRE_NOTHROW(store.reserve(100));
    REQUIRE(store.size() == 0);

    // Add vectors after reserve
    float vec[] = {1.0f, 2.0f, 3.0f};
    for (uint64_t i = 0; i < 10; ++i) {
      store.add(i, vec);
    }

    REQUIRE(store.size() == 10);
  }
}

TEST_CASE("VectorStore - stress test", "[vector_store][stress]") {
  constexpr size_t dim = 128;
  quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::L2);

  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  SECTION("Add and remove many vectors") {
    // Add 1000 vectors
    for (uint64_t i = 0; i < 1000; ++i) {
      std::vector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = dis(gen);
      }
      store.add(i, vec.data());
    }

    REQUIRE(store.size() == 1000);

    // Remove half of them
    for (uint64_t i = 0; i < 500; ++i) {
      store.remove(i * 2);
    }

    REQUIRE(store.size() == 500);

    // Search still works
    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
      query[j] = dis(gen);
    }

    auto results = store.search(query.data(), 10);
    REQUIRE(results.size() == 10);
  }
}

TEST_CASE("VectorStore - concurrent access", "[vector_store][thread]") {
  constexpr size_t dim = 64;
  quiverdb::VectorStore store(dim, quiverdb::DistanceMetric::L2);

  SECTION("Concurrent reads are safe") {
    // Pre-populate store
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (uint64_t i = 0; i < 100; ++i) {
      std::vector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = dis(gen);
      }
      store.add(i, vec.data());
    }

    // Launch multiple reader threads
    std::vector<std::thread> readers;
    std::atomic<int> successful_reads{0};

    for (int t = 0; t < 4; ++t) {
      readers.emplace_back([&store, &successful_reads]() {
        constexpr size_t dim = 64;
        std::mt19937 local_gen(std::random_device{}());
        std::uniform_real_distribution<float> local_dis(-1.0f, 1.0f);

        for (int i = 0; i < 100; ++i) {
          // Concurrent search
          std::vector<float> query(dim);
          for (size_t j = 0; j < dim; ++j) {
            query[j] = local_dis(local_gen);
          }
          auto results = store.search(query.data(), 5);
          if (results.size() == 5) {
            successful_reads++;
          }

          // Concurrent contains check
          store.contains(static_cast<uint64_t>(i));

          // Concurrent size check
          store.size();
        }
      });
    }

    for (auto& t : readers) {
      t.join();
    }

    REQUIRE(successful_reads == 400);
  }

  SECTION("Concurrent writes are safe") {
    std::vector<std::thread> writers;
    std::atomic<int> successful_writes{0};

    for (int t = 0; t < 4; ++t) {
      writers.emplace_back([&store, &successful_writes, t]() {
        constexpr size_t dim = 64;
        std::mt19937 local_gen(42 + t);
        std::uniform_real_distribution<float> local_dis(-1.0f, 1.0f);

        for (int i = 0; i < 25; ++i) {
          std::vector<float> vec(dim);
          for (size_t j = 0; j < dim; ++j) {
            vec[j] = local_dis(local_gen);
          }
          // Each thread uses unique IDs: thread 0 -> 0-24, thread 1 -> 25-49, etc.
          uint64_t id = static_cast<uint64_t>(t * 25 + i);
          store.add(id, vec.data());
          successful_writes++;
        }
      });
    }

    for (auto& t : writers) {
      t.join();
    }

    REQUIRE(successful_writes == 100);
    REQUIRE(store.size() == 100);
  }

  SECTION("Concurrent read-write is safe") {
    // Pre-populate with some vectors
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (uint64_t i = 0; i < 50; ++i) {
      std::vector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = dis(gen);
      }
      store.add(i, vec.data());
    }

    std::atomic<bool> stop{false};
    std::atomic<int> read_count{0};
    std::atomic<int> write_count{0};

    // Reader thread
    std::thread reader([&]() {
      std::mt19937 local_gen(std::random_device{}());
      std::uniform_real_distribution<float> local_dis(-1.0f, 1.0f);

      while (!stop) {
        std::vector<float> query(dim);
        for (size_t j = 0; j < dim; ++j) {
          query[j] = local_dis(local_gen);
        }
        auto results = store.search(query.data(), 5);
        read_count++;
      }
    });

    // Writer thread
    std::thread writer([&]() {
      std::mt19937 local_gen(std::random_device{}());
      std::uniform_real_distribution<float> local_dis(-1.0f, 1.0f);

      for (uint64_t i = 50; i < 100; ++i) {
        std::vector<float> vec(dim);
        for (size_t j = 0; j < dim; ++j) {
          vec[j] = local_dis(local_gen);
        }
        store.add(i, vec.data());
        write_count++;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    });

    writer.join();
    stop = true;
    reader.join();

    REQUIRE(write_count == 50);
    REQUIRE(read_count > 0);
    REQUIRE(store.size() == 100);
  }
}
