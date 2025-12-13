#include "core/hnsw_index.h"
#include <atomic>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>
#include <vector>

using Catch::Approx;

TEST_CASE("HNSWIndex - construction", "[hnsw]") {
  SECTION("Valid construction") {
    REQUIRE_NOTHROW(quiverdb::HNSWIndex(768));
    REQUIRE_NOTHROW(quiverdb::HNSWIndex(128, quiverdb::HNSWDistanceMetric::COSINE));
    REQUIRE_NOTHROW(quiverdb::HNSWIndex(64, quiverdb::HNSWDistanceMetric::L2, 10000, 32, 400));
  }

  SECTION("Zero dimension throws") {
    REQUIRE_THROWS_AS(quiverdb::HNSWIndex(0), std::invalid_argument);
  }

  SECTION("Zero max_elements throws") {
    REQUIRE_THROWS_AS(quiverdb::HNSWIndex(768, quiverdb::HNSWDistanceMetric::L2, 0), std::invalid_argument);
  }

  SECTION("M < 2 throws") {
    // M=0 should throw
    REQUIRE_THROWS_AS(quiverdb::HNSWIndex(768, quiverdb::HNSWDistanceMetric::L2, 1000, 0), std::invalid_argument);
    // M=1 should throw
    REQUIRE_THROWS_AS(quiverdb::HNSWIndex(768, quiverdb::HNSWDistanceMetric::L2, 1000, 1), std::invalid_argument);
    // M=2 should succeed
    REQUIRE_NOTHROW(quiverdb::HNSWIndex(768, quiverdb::HNSWDistanceMetric::L2, 1000, 2));
  }

  SECTION("Check initial state") {
    quiverdb::HNSWIndex index(768);
    REQUIRE(index.size() == 0);
    REQUIRE(index.dimension() == 768);
    REQUIRE(index.capacity() == 100000);  // default
  }
}

TEST_CASE("HNSWIndex - add and search", "[hnsw]") {
  constexpr size_t dim = 64;
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, 1000);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  SECTION("Add single vector and search") {
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
      vec[i] = dis(gen);
    }

    index.add(1, vec.data());
    REQUIRE(index.size() == 1);
    REQUIRE(index.contains(1));

    // Search should return the same vector
    auto results = index.search(vec.data(), 1);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 1);
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-6f));
  }

  SECTION("Add multiple vectors") {
    for (uint64_t i = 0; i < 100; ++i) {
      std::vector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = dis(gen);
      }
      index.add(i, vec.data());
    }

    REQUIRE(index.size() == 100);
  }

  SECTION("Duplicate ID throws") {
    std::vector<float> vec1(dim, 1.0f);
    std::vector<float> vec2(dim, 2.0f);

    index.add(42, vec1.data());
    REQUIRE_THROWS_AS(index.add(42, vec2.data()), std::invalid_argument);
  }

  SECTION("Null vector throws") {
    REQUIRE_THROWS_AS(index.add(1, nullptr), std::invalid_argument);

    std::vector<float> vec(dim, 1.0f);
    index.add(1, vec.data());
    REQUIRE_THROWS_AS(index.search(nullptr, 1), std::invalid_argument);
  }

  SECTION("k=0 throws") {
    std::vector<float> vec(dim, 1.0f);
    index.add(1, vec.data());
    REQUIRE_THROWS_AS(index.search(vec.data(), 0), std::invalid_argument);
  }

  SECTION("Index full throws") {
    constexpr size_t small_capacity = 5;
    quiverdb::HNSWIndex small_index(dim, quiverdb::HNSWDistanceMetric::L2, small_capacity);

    std::vector<float> vec(dim, 1.0f);
    for (uint64_t i = 0; i < small_capacity; ++i) {
      small_index.add(i, vec.data());
    }
    REQUIRE(small_index.size() == small_capacity);

    // Adding one more should throw
    REQUIRE_THROWS_AS(small_index.add(small_capacity, vec.data()), std::runtime_error);
  }
}

TEST_CASE("HNSWIndex - search quality", "[hnsw]") {
  constexpr size_t dim = 32;
  constexpr size_t num_vectors = 500;
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, num_vectors, 16, 100);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // Store vectors for ground truth computation
  std::vector<std::vector<float>> all_vectors(num_vectors);

  // Add vectors
  for (uint64_t i = 0; i < num_vectors; ++i) {
    all_vectors[i].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      all_vectors[i][j] = dis(gen);
    }
    index.add(i, all_vectors[i].data());
  }

  SECTION("Search finds exact match") {
    // Search for an existing vector
    auto results = index.search(all_vectors[42].data(), 1);
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].id == 42);
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-5f));
  }

  SECTION("Search returns k results") {
    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
      query[i] = dis(gen);
    }

    auto results = index.search(query.data(), 10);
    REQUIRE(results.size() == 10);

    // Results should be sorted by distance
    for (size_t i = 1; i < results.size(); ++i) {
      REQUIRE(results[i].distance >= results[i-1].distance);
    }
  }

  SECTION("Higher ef_search improves recall") {
    // Compute ground truth (brute force)
    std::vector<float> query(dim);
    for (size_t i = 0; i < dim; ++i) {
      query[i] = dis(gen);
    }

    std::vector<std::pair<float, uint64_t>> ground_truth;
    for (uint64_t i = 0; i < num_vectors; ++i) {
      float dist = quiverdb::l2_sq(query.data(), all_vectors[i].data(), dim);
      ground_truth.emplace_back(dist, i);
    }
    std::sort(ground_truth.begin(), ground_truth.end());

    // Search with low ef
    index.set_ef_search(10);
    auto results_low = index.search(query.data(), 10);

    // Search with high ef
    index.set_ef_search(100);
    auto results_high = index.search(query.data(), 10);

    // Count recall
    std::unordered_set<uint64_t> gt_set;
    for (size_t i = 0; i < 10; ++i) {
      gt_set.insert(ground_truth[i].second);
    }

    int recall_low = 0, recall_high = 0;
    for (const auto& r : results_low) {
      if (gt_set.count(r.id)) recall_low++;
    }
    for (const auto& r : results_high) {
      if (gt_set.count(r.id)) recall_high++;
    }

    // Higher ef should give same or better recall
    REQUIRE(recall_high >= recall_low);

    // With ef=100 on 500 vectors, recall should be high
    REQUIRE(recall_high >= 8);  // At least 80% recall
  }
}

TEST_CASE("HNSWIndex - distance metrics", "[hnsw]") {
  constexpr size_t dim = 8;

  SECTION("L2 distance") {
    quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, 100);

    std::vector<float> v1 = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v2 = {0, 1, 0, 0, 0, 0, 0, 0};
    std::vector<float> v3 = {1, 0, 0, 0, 0, 0, 0, 0};  // Same as v1

    index.add(1, v1.data());
    index.add(2, v2.data());
    index.add(3, v3.data());

    // Query with v1 - should find v3 (identical) then v2
    auto results = index.search(v1.data(), 3);
    REQUIRE(results.size() == 3);
    // First result should be v1 or v3 (distance 0)
    REQUIRE((results[0].id == 1 || results[0].id == 3));
    REQUIRE(results[0].distance == Approx(0.0f).margin(1e-6f));
  }

  SECTION("Cosine distance") {
    quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::COSINE, 100);

    std::vector<float> v1 = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v2 = {2, 0, 0, 0, 0, 0, 0, 0};  // Same direction, different magnitude
    std::vector<float> v3 = {0, 1, 0, 0, 0, 0, 0, 0};  // Orthogonal

    index.add(1, v1.data());
    index.add(2, v2.data());
    index.add(3, v3.data());

    // Query with v1 - v2 should be closest (same direction)
    auto results = index.search(v1.data(), 3);
    REQUIRE(results.size() == 3);
    // v1 and v2 have cosine distance ~0
    REQUIRE((results[0].id == 1 || results[0].id == 2));
  }

  SECTION("Dot product (MIPS)") {
    quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::DOT, 100);

    std::vector<float> v1 = {1, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> v2 = {2, 0, 0, 0, 0, 0, 0, 0};  // Higher dot product
    std::vector<float> v3 = {0, 1, 0, 0, 0, 0, 0, 0};  // Orthogonal

    index.add(1, v1.data());
    index.add(2, v2.data());
    index.add(3, v3.data());

    // Query with v1 - v2 should be "closest" (highest dot product = lowest negative)
    auto results = index.search(v1.data(), 3);
    REQUIRE(results.size() == 3);
    // v2 has highest dot product with v1
    REQUIRE(results[0].id == 2);
  }
}

TEST_CASE("HNSWIndex - stress test", "[hnsw][stress]") {
  constexpr size_t dim = 128;
  constexpr size_t num_vectors = 1000;
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, num_vectors);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  SECTION("Insert many vectors") {
    for (uint64_t i = 0; i < num_vectors; ++i) {
      std::vector<float> vec(dim);
      for (size_t j = 0; j < dim; ++j) {
        vec[j] = dis(gen);
      }
      index.add(i, vec.data());
    }

    REQUIRE(index.size() == num_vectors);

    // Should be able to search
    std::vector<float> query(dim);
    for (size_t j = 0; j < dim; ++j) {
      query[j] = dis(gen);
    }

    auto results = index.search(query.data(), 10);
    REQUIRE(results.size() == 10);
  }
}

TEST_CASE("HNSWIndex - concurrent search", "[hnsw][thread]") {
  constexpr size_t dim = 64;
  constexpr size_t num_vectors = 500;
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, num_vectors);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // Pre-populate index
  for (uint64_t i = 0; i < num_vectors; ++i) {
    std::vector<float> vec(dim);
    for (size_t j = 0; j < dim; ++j) {
      vec[j] = dis(gen);
    }
    index.add(i, vec.data());
  }

  SECTION("Multiple concurrent searches") {
    std::atomic<int> completed{0};
    constexpr int num_threads = 4;
    constexpr int searches_per_thread = 100;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&index, &completed, t]() {
        std::mt19937 local_gen(t * 1000);
        std::uniform_real_distribution<float> local_dis(-1.0f, 1.0f);

        for (int i = 0; i < searches_per_thread; ++i) {
          std::vector<float> query(dim);
          for (size_t j = 0; j < dim; ++j) {
            query[j] = local_dis(local_gen);
          }
          auto results = index.search(query.data(), 10);
          REQUIRE(results.size() == 10);
        }
        ++completed;
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    REQUIRE(completed == num_threads);
  }
}

TEST_CASE("HNSWIndex - serialization", "[hnsw][serialization]") {
  const std::string filename = "test_hnsw_index.bin";
  constexpr size_t dim = 16;
  constexpr size_t max_elements = 100;
  constexpr size_t M = 8;
  constexpr size_t ef_construction = 50;
  constexpr quiverdb::HNSWDistanceMetric metric = quiverdb::HNSWDistanceMetric::COSINE;

  // Create and populate an index
  quiverdb::HNSWIndex original_index(dim, metric, max_elements, M, ef_construction);
  original_index.set_ef_search(30);

  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<std::vector<float>> test_vectors(max_elements / 2); // Populate half capacity
  for (uint64_t i = 0; i < test_vectors.size(); ++i) {
    test_vectors[i].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      test_vectors[i][j] = dis(gen);
    }
    original_index.add(i + 1, test_vectors[i].data()); // IDs starting from 1
  }

  SECTION("Save and load index successfully") {
    // Save the original index
    REQUIRE_NOTHROW(original_index.save(filename));

    // Load into a new index
    std::unique_ptr<quiverdb::HNSWIndex> loaded_index_ptr;
    REQUIRE_NOTHROW(loaded_index_ptr = quiverdb::HNSWIndex::load(filename));
    quiverdb::HNSWIndex& loaded_index = *loaded_index_ptr;

    // Verify configuration parameters
    REQUIRE(loaded_index.dimension() == original_index.dimension());
    REQUIRE(loaded_index.capacity() == original_index.capacity());
    REQUIRE(loaded_index.size() == original_index.size());
    REQUIRE(loaded_index.get_ef_search() == original_index.get_ef_search());
    // Metric is private, cannot directly check. Assume it's loaded correctly.

    // Verify search results are identical
    for (const auto& vec : test_vectors) {
      std::vector<quiverdb::HNSWSearchResult> original_results = original_index.search(vec.data(), 5);
      std::vector<quiverdb::HNSWSearchResult> loaded_results = loaded_index.search(vec.data(), 5);

      REQUIRE(original_results.size() == loaded_results.size());
      for (size_t i = 0; i < original_results.size(); ++i) {
        REQUIRE(original_results[i].id == loaded_results[i].id);
        REQUIRE(original_results[i].distance == Approx(loaded_results[i].distance).margin(1e-5f));
      }
    }
  }

  SECTION("Loading from non-existent file throws") {
    std::filesystem::remove(filename); // Ensure file doesn't exist
    REQUIRE_THROWS_AS(quiverdb::HNSWIndex::load(filename + "_nonexistent"), std::runtime_error);
  }

  SECTION("Loading corrupted file - bad magic number") {
    // Create a file with wrong magic number
    std::string corrupt_file = filename + "_corrupt_magic";
    {
      std::ofstream ofs(corrupt_file, std::ios::binary);
      uint32_t bad_magic = 0xDEADBEEF;
      ofs.write(reinterpret_cast<const char*>(&bad_magic), sizeof(bad_magic));
    }
    REQUIRE_THROWS_AS(quiverdb::HNSWIndex::load(corrupt_file), std::runtime_error);
    std::filesystem::remove(corrupt_file);
  }

  SECTION("Loading corrupted file - unsupported version") {
    // Create a file with correct magic but wrong version
    std::string corrupt_file = filename + "_corrupt_version";
    {
      std::ofstream ofs(corrupt_file, std::ios::binary);
      uint32_t magic = 0x51565244; // "QVRD" - correct magic
      uint32_t bad_version = 999;
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      ofs.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
    }
    REQUIRE_THROWS_AS(quiverdb::HNSWIndex::load(corrupt_file), std::runtime_error);
    std::filesystem::remove(corrupt_file);
  }

  SECTION("Loading truncated file") {
    // Save valid index then truncate it
    original_index.save(filename);

    // Truncate the file to 50 bytes (incomplete header)
    {
      std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
      uint32_t magic = 0x51565244; // "QVRD" - correct magic but truncated
      ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      // Write partial data - this should fail on load
    }
    REQUIRE_THROWS(quiverdb::HNSWIndex::load(filename));
  }

  SECTION("Loading empty file") {
    std::string empty_file = filename + "_empty";
    {
      std::ofstream ofs(empty_file, std::ios::binary);
      // Empty file
    }
    REQUIRE_THROWS(quiverdb::HNSWIndex::load(empty_file));
    std::filesystem::remove(empty_file);
  }

  SECTION("Loading corrupted file - invalid entry point") {
    // Save valid index first
    original_index.save(filename);

    // Read file, corrupt ep_ to be >= count, write back
    std::string corrupt_file = filename + "_corrupt_ep";
    {
      std::ifstream ifs(filename, std::ios::binary);
      std::vector<char> data((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
      ifs.close();

      // ep_ is at offset: 4(magic) + 4(version) + 8(dim) + 4(metric) + 8(max_el) + 8(M) +
      //                   8(ef_con) + 8(ef_s) + 8(mult) + 8(count) = 68 bytes
      // Write invalid ep_ value (999999, much larger than count)
      size_t invalid_ep = 999999;
      std::memcpy(data.data() + 68, &invalid_ep, sizeof(invalid_ep));

      std::ofstream ofs(corrupt_file, std::ios::binary);
      ofs.write(data.data(), data.size());
    }
    REQUIRE_THROWS_AS(quiverdb::HNSWIndex::load(corrupt_file), std::runtime_error);
    std::filesystem::remove(corrupt_file);
  }

  SECTION("Loading corrupted file - invalid max_level") {
    original_index.save(filename);

    std::string corrupt_file = filename + "_corrupt_maxlevel";
    {
      std::ifstream ifs(filename, std::ios::binary);
      std::vector<char> data((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
      ifs.close();

      // max_level_ is at offset 68 + 8(ep_) = 76 bytes
      int invalid_max_level = 100;  // > MAX_LEVEL (32)
      std::memcpy(data.data() + 76, &invalid_max_level, sizeof(invalid_max_level));

      std::ofstream ofs(corrupt_file, std::ios::binary);
      ofs.write(data.data(), data.size());
    }
    REQUIRE_THROWS_AS(quiverdb::HNSWIndex::load(corrupt_file), std::runtime_error);
    std::filesystem::remove(corrupt_file);
  }

  SECTION("get_vector returns correct data after save/load") {
    original_index.save(filename);
    auto loaded = quiverdb::HNSWIndex::load(filename);

    // Verify vectors are preserved
    for (size_t i = 0; i < test_vectors.size(); ++i) {
      std::vector<float> retrieved = loaded->get_vector(static_cast<uint64_t>(i + 1));
      REQUIRE(retrieved.size() == dim);
      for (size_t j = 0; j < dim; ++j) {
        REQUIRE(retrieved[j] == Approx(test_vectors[i][j]).margin(1e-6f));
      }
    }
  }

  // Cleanup
  std::filesystem::remove(filename);
}

TEST_CASE("HNSWIndex - recall benchmark", "[hnsw][.benchmark]") {
  // This test measures recall rate - marked as hidden benchmark

  constexpr size_t dim = 128;
  constexpr size_t num_vectors = 5000;
  constexpr size_t num_queries = 100;
  constexpr size_t k = 10;

  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, num_vectors, 16, 200);

  std::mt19937 gen(123);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // Store all vectors
  std::vector<std::vector<float>> all_vectors(num_vectors);
  for (uint64_t i = 0; i < num_vectors; ++i) {
    all_vectors[i].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      all_vectors[i][j] = dis(gen);
    }
    index.add(i, all_vectors[i].data());
  }

  // Generate queries
  std::vector<std::vector<float>> queries(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    queries[q].resize(dim);
    for (size_t j = 0; j < dim; ++j) {
      queries[q][j] = dis(gen);
    }
  }

  // Compute ground truth (brute force)
  std::vector<std::unordered_set<uint64_t>> ground_truth(num_queries);
  for (size_t q = 0; q < num_queries; ++q) {
    std::vector<std::pair<float, uint64_t>> distances;
    for (uint64_t i = 0; i < num_vectors; ++i) {
      float dist = quiverdb::l2_sq(queries[q].data(), all_vectors[i].data(), dim);
      distances.emplace_back(dist, i);
    }
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());
    for (size_t i = 0; i < k; ++i) {
      ground_truth[q].insert(distances[i].second);
    }
  }

  // Test different ef values
  for (size_t ef : {10, 50, 100, 200}) {
    index.set_ef_search(ef);

    double total_recall = 0.0;
    for (size_t q = 0; q < num_queries; ++q) {
      auto results = index.search(queries[q].data(), k);
      int hits = 0;
      for (const auto& r : results) {
        if (ground_truth[q].count(r.id)) hits++;
      }
      total_recall += static_cast<double>(hits) / k;
    }

    double avg_recall = total_recall / num_queries;
    INFO("ef=" << ef << " recall=" << avg_recall);
    REQUIRE(avg_recall > 0.5);  // Should have at least 50% recall
  }
}

TEST_CASE("HNSWIndex - ef_search validation", "[hnsw]") {
  constexpr size_t dim = 16;
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, 100);

  SECTION("ef_search = 0 throws") {
    REQUIRE_THROWS_AS(index.set_ef_search(0), std::invalid_argument);
  }

  SECTION("Valid ef_search values succeed") {
    REQUIRE_NOTHROW(index.set_ef_search(1));
    REQUIRE(index.get_ef_search() == 1);

    REQUIRE_NOTHROW(index.set_ef_search(100));
    REQUIRE(index.get_ef_search() == 100);

    REQUIRE_NOTHROW(index.set_ef_search(10000));
    REQUIRE(index.get_ef_search() == 10000);
  }
}

TEST_CASE("HNSWIndex - get_vector edge cases", "[hnsw]") {
  constexpr size_t dim = 8;
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, 100);

  SECTION("get_vector on non-existent ID throws") {
    REQUIRE_THROWS_AS(index.get_vector(42), std::runtime_error);
  }

  SECTION("get_vector on empty index throws") {
    REQUIRE_THROWS_AS(index.get_vector(0), std::runtime_error);
  }

  SECTION("get_vector returns correct vector") {
    std::vector<float> original = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    index.add(42, original.data());

    std::vector<float> retrieved = index.get_vector(42);
    REQUIRE(retrieved.size() == dim);
    for (size_t i = 0; i < dim; ++i) {
      REQUIRE(retrieved[i] == Approx(original[i]).margin(1e-6f));
    }
  }
}

TEST_CASE("HNSWIndex - corrupted RNG state", "[hnsw][serialization]") {
  const std::string filename = "test_hnsw_rng_corrupt.bin";
  constexpr size_t dim = 8;

  // Create and save a valid index
  quiverdb::HNSWIndex original(dim, quiverdb::HNSWDistanceMetric::L2, 50);
  std::vector<float> vec(dim, 1.0f);
  original.add(1, vec.data());
  original.save(filename);

  SECTION("Loading file with corrupted RNG state throws") {
    // Read the file content
    std::ifstream ifs(filename, std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
    ifs.close();

    // The RNG state is serialized near the end as a string stream
    // We corrupt it by truncating the file before RNG state is complete
    size_t corrupt_size = data.size() - 100; // Truncate last 100 bytes
    if (corrupt_size > 0) {
      std::string corrupt_file = filename + "_rng_corrupt";
      std::ofstream ofs(corrupt_file, std::ios::binary);
      ofs.write(data.data(), corrupt_size);
      ofs.close();

      REQUIRE_THROWS(quiverdb::HNSWIndex::load(corrupt_file));
      std::filesystem::remove(corrupt_file);
    }
  }

  std::filesystem::remove(filename);
}

TEST_CASE("HNSWIndex - contains edge cases", "[hnsw]") {
  constexpr size_t dim = 8;
  quiverdb::HNSWIndex index(dim, quiverdb::HNSWDistanceMetric::L2, 100);

  SECTION("contains returns false for empty index") {
    REQUIRE_FALSE(index.contains(0));
    REQUIRE_FALSE(index.contains(1));
    REQUIRE_FALSE(index.contains(999));
  }

  SECTION("contains returns true after add") {
    std::vector<float> vec(dim, 1.0f);
    index.add(42, vec.data());

    REQUIRE(index.contains(42));
    REQUIRE_FALSE(index.contains(0));
    REQUIRE_FALSE(index.contains(43));
  }
}
