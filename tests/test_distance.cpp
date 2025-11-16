#include "core/distance.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>

using Catch::Approx;

TEST_CASE("L2 distance scalar - basic calculation", "[distance][scalar]") {
  SECTION("Simple 3d vectors") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float result = quiverdb::l2_sq_scalar(a, b, 3);

    // Expected: (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
    REQUIRE(result == Approx(27.0f).epsilon(0.001));
  }

  SECTION("Identical vectors return zero distance") {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float result = quiverdb::l2_sq_scalar(a, a, 4);

    REQUIRE(result == Approx(0.0f).margin(1e-6));
  }

  SECTION("High dimensional vectors (768d)") {
    constexpr size_t dim = 768;
    std::vector<float> a(dim);
    std::vector<float> b(dim);
    for (size_t i = 0; i < dim; ++i) {
      a[i] = static_cast<float>(i) / dim;
      b[i] = static_cast<float>(i) / dim + 0.5f;
    }

    float result = quiverdb::l2_sq_scalar(a.data(), b.data(), dim);

    // Expected: 768 * 0.5^2 = 768 * 0.25 = 192
    REQUIRE(result == Approx(192.0f).epsilon(0.001));
  }

  SECTION("Negative values") {
    float a[] = {-1.0f, -2.0f, -3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    float result = quiverdb::l2_sq_scalar(a, b, 3);
    // Expected: (-1-1)^2 + (-2-2)^2 + (-3-3)^2 = 4+16+36 = 56
    REQUIRE(result == Approx(56.0f).epsilon(0.001));
  }
}

#ifdef QUIVER_ARM_NEON
TEST_CASE("L2 distance ARM NEON - correctness vs scalar", "[distance][neon]") {
  SECTION("Simple 4d vectors (single NEON register)") {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_result = quiverdb::l2_sq_scalar(a, b, 4);
    float neon_result = quiverdb::l2_sq_neon(a, b, 4);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }

  SECTION("128d vectors") {
    constexpr size_t dim = 128;
    std::vector<float> a(dim);
    std::vector<float> b(dim);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < dim; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    float scalar_result = quiverdb::l2_sq_scalar(a.data(), b.data(), dim);
    float neon_result = quiverdb::l2_sq_neon(a.data(), b.data(), dim);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }

  SECTION("768d vectors") {
    constexpr size_t dim = 768;
    std::vector<float> a(dim);
    std::vector<float> b(dim);

    std::mt19937 gen(54321);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < dim; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    float scalar_result = quiverdb::l2_sq_scalar(a.data(), b.data(), dim);
    float neon_result = quiverdb::l2_sq_neon(a.data(), b.data(), dim);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }

  SECTION("1536d vectors (OpenAI large embeddings)") {
    constexpr size_t dim = 1536;
    std::vector<float> a(dim);
    std::vector<float> b(dim);

    std::mt19937 gen(99999);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < dim; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    float scalar_result = quiverdb::l2_sq_scalar(a.data(), b.data(), dim);
    float neon_result = quiverdb::l2_sq_neon(a.data(), b.data(), dim);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }

  SECTION("Dimension not divisible by 4 (tests remainder handling)") {
    constexpr size_t dim = 773; // 768 + 5
    std::vector<float> a(dim);
    std::vector<float> b(dim);

    for (size_t i = 0; i < dim; ++i) {
      a[i] = static_cast<float>(i);
      b[i] = static_cast<float>(i) + 1.0f;
    }

    float scalar_result = quiverdb::l2_sq_scalar(a.data(), b.data(), dim);
    float neon_result = quiverdb::l2_sq_neon(a.data(), b.data(), dim);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }
}

TEST_CASE("L2 distance NEON - small dimensions (remainder path)", "[distance][neon][edge]") {
  SECTION("dim=1 (pure remainder)") {
    float a[] = {3.0f};
    float b[] = {7.0f};
    float scalar = quiverdb::l2_sq_scalar(a, b, 1);
    float neon = quiverdb::l2_sq_neon(a, b, 1);
    REQUIRE(neon == Approx(scalar).epsilon(0.0001));
    REQUIRE(neon == Approx(16.0f));
  }

  SECTION("dim=2 (pure remainder)") {
    float a[] = {1.0f, 2.0f};
    float b[] = {4.0f, 6.0f};
    float scalar = quiverdb::l2_sq_scalar(a, b, 2);
    float neon = quiverdb::l2_sq_neon(a, b, 2);
    REQUIRE(neon == Approx(scalar).epsilon(0.0001));
    REQUIRE(neon == Approx(25.0f));
  }

  SECTION("dim=3 (pure remainder)") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float scalar = quiverdb::l2_sq_scalar(a, b, 3);
    float neon = quiverdb::l2_sq_neon(a, b, 3);
    REQUIRE(neon == Approx(scalar).epsilon(0.0001));
    REQUIRE(neon == Approx(27.0f));
  }
}
#endif

TEST_CASE("L2 distance - edge cases", "[distance][edge]") {
  SECTION("Zero vectors") {
    float a[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 0.0f, 0.0f, 0.0f};

    float scalar_result = quiverdb::l2_sq_scalar(a, b, 4);

    REQUIRE(scalar_result == Approx(0.0f).margin(1e-6));

#ifdef QUIVER_ARM_NEON
    float neon_result = quiverdb::l2_sq_neon(a, b, 4);
    REQUIRE(neon_result == Approx(0.0f).margin(1e-6));
#endif
  }

  SECTION("Identical large vectors") {
    constexpr size_t dim = 1024;
    std::vector<float> a(dim, 42.0f);

    float scalar_result = quiverdb::l2_sq_scalar(a.data(), a.data(), dim);

    REQUIRE(scalar_result == Approx(0.0f).margin(1e-5));

#ifdef QUIVER_ARM_NEON
    float neon_result = quiverdb::l2_sq_neon(a.data(), a.data(), dim);
    REQUIRE(neon_result == Approx(0.0f).margin(1e-5));
#endif
  }

  SECTION("Zero dimension returns zero") {
    float a[] = {1.0f};
    float b[] = {2.0f};

    REQUIRE(quiverdb::l2_sq_scalar(a, b, 0) == 0.0f);
#ifdef QUIVER_ARM_NEON
    REQUIRE(quiverdb::l2_sq_neon(a, b, 0) == 0.0f);
#endif
    REQUIRE(quiverdb::l2_sq(a, b, 0) == 0.0f);
  }
}

TEST_CASE("L2 distance - public API uses best implementation","[distance][api]") {
  SECTION("768d vectors") {
    constexpr size_t dim = 768;
    std::vector<float> a(dim);
    std::vector<float> b(dim);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < dim; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    float api_result = quiverdb::l2_sq(a.data(), b.data(), dim);
    float scalar_result = quiverdb::l2_sq_scalar(a.data(), b.data(), dim);

    // Should match scalar (or be SIMD-optimized version that matches)
    REQUIRE(api_result == Approx(scalar_result).epsilon(0.0001));
  }
}

// ============================================================================
// Dot Product Tests
// ============================================================================

TEST_CASE("Dot product scalar - basic calculation", "[distance][dot][scalar]") {
  SECTION("Simple 3d vectors") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float result = quiverdb::dot_product_scalar(a, b, 3);

    // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    REQUIRE(result == Approx(32.0f).epsilon(0.001));
  }

  SECTION("Orthogonal vectors") {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    float result = quiverdb::dot_product_scalar(a, b, 3);

    REQUIRE(result == Approx(0.0f).margin(1e-6));
  }

  SECTION("Identical vectors") {
    float a[] = {2.0f, 3.0f, 4.0f};
    float result = quiverdb::dot_product_scalar(a, a, 3);

    // Expected: 2*2 + 3*3 + 4*4 = 4 + 9 + 16 = 29
    REQUIRE(result == Approx(29.0f).epsilon(0.001));
  }

  SECTION("Negative values") {
    float a[] = {-1.0f, -2.0f, -3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    float result = quiverdb::dot_product_scalar(a, b, 3);

    // Expected: -1*1 + -2*2 + -3*3 = -1 - 4 - 9 = -14
    REQUIRE(result == Approx(-14.0f).epsilon(0.001));
  }

  SECTION("Zero dimension") {
    float a[] = {1.0f};
    float b[] = {2.0f};
    REQUIRE(quiverdb::dot_product_scalar(a, b, 0) == 0.0f);
  }
}

#ifdef QUIVER_ARM_NEON
TEST_CASE("Dot product NEON - correctness vs scalar", "[distance][dot][neon]") {
  SECTION("Simple 4d vectors") {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_result = quiverdb::dot_product_scalar(a, b, 4);
    float neon_result = quiverdb::dot_product_neon(a, b, 4);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }

  SECTION("768d vectors") {
    constexpr size_t dim = 768;
    std::vector<float> a(dim);
    std::vector<float> b(dim);

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < dim; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    float scalar_result = quiverdb::dot_product_scalar(a.data(), b.data(), dim);
    float neon_result = quiverdb::dot_product_neon(a.data(), b.data(), dim);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }

  SECTION("Non-multiple of 4 dimension") {
    constexpr size_t dim = 773;
    std::vector<float> a(dim, 1.0f);
    std::vector<float> b(dim, 2.0f);

    float scalar_result = quiverdb::dot_product_scalar(a.data(), b.data(), dim);
    float neon_result = quiverdb::dot_product_neon(a.data(), b.data(), dim);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }
}
#endif

TEST_CASE("Dot product - public API", "[distance][dot][api]") {
  SECTION("768d vectors") {
    constexpr size_t dim = 768;
    std::vector<float> a(dim);
    std::vector<float> b(dim);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < dim; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    float api_result = quiverdb::dot_product(a.data(), b.data(), dim);
    float scalar_result = quiverdb::dot_product_scalar(a.data(), b.data(), dim);

    REQUIRE(api_result == Approx(scalar_result).epsilon(0.0001));
  }
}

// ============================================================================
// Cosine Distance Tests
// ============================================================================

TEST_CASE("Cosine distance scalar - basic calculation", "[distance][cosine][scalar]") {
  SECTION("Identical vectors (distance = 0)") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    float result = quiverdb::cosine_distance_scalar(a, b, 3);

    REQUIRE(result == Approx(0.0f).margin(1e-6));
  }

  SECTION("Scaled identical vectors") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {2.0f, 4.0f, 6.0f};
    float result = quiverdb::cosine_distance_scalar(a, b, 3);

    // Same direction, should be ~0
    REQUIRE(result == Approx(0.0f).margin(1e-6));
  }

  SECTION("Orthogonal vectors (distance = 1)") {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    float result = quiverdb::cosine_distance_scalar(a, b, 3);

    // Orthogonal, cosine = 0, distance = 1
    REQUIRE(result == Approx(1.0f).epsilon(0.001));
  }

  SECTION("Opposite vectors (distance = 2)") {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {-1.0f, 0.0f, 0.0f};
    float result = quiverdb::cosine_distance_scalar(a, b, 3);

    // Opposite direction, cosine = -1, distance = 2
    REQUIRE(result == Approx(2.0f).epsilon(0.001));
  }

  SECTION("Zero vectors (edge case)") {
    float a[] = {0.0f, 0.0f, 0.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    float result = quiverdb::cosine_distance_scalar(a, b, 3);

    // Should return 1.0 (max distance) for zero magnitude
    REQUIRE(result == Approx(1.0f).epsilon(0.001));
  }

  SECTION("General case") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float result = quiverdb::cosine_distance_scalar(a, b, 3);

    // dot = 32, ||a|| = sqrt(14), ||b|| = sqrt(77)
    // cosine = 32 / sqrt(14*77) = 32 / sqrt(1078) ≈ 0.9746
    // distance = 1 - 0.9746 ≈ 0.0254
    REQUIRE(result == Approx(0.0254).epsilon(0.01));
  }
}

#ifdef QUIVER_ARM_NEON
TEST_CASE("Cosine distance NEON - correctness vs scalar", "[distance][cosine][neon]") {
  SECTION("Simple 4d vectors") {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float scalar_result = quiverdb::cosine_distance_scalar(a, b, 4);
    float neon_result = quiverdb::cosine_distance_neon(a, b, 4);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }

  SECTION("768d vectors") {
    constexpr size_t dim = 768;
    std::vector<float> a(dim);
    std::vector<float> b(dim);

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < dim; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    float scalar_result = quiverdb::cosine_distance_scalar(a.data(), b.data(), dim);
    float neon_result = quiverdb::cosine_distance_neon(a.data(), b.data(), dim);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }

  SECTION("Non-multiple of 4 dimension") {
    constexpr size_t dim = 773;
    std::vector<float> a(dim);
    std::vector<float> b(dim);

    std::mt19937 gen(99999);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < dim; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    float scalar_result = quiverdb::cosine_distance_scalar(a.data(), b.data(), dim);
    float neon_result = quiverdb::cosine_distance_neon(a.data(), b.data(), dim);

    REQUIRE(neon_result == Approx(scalar_result).epsilon(0.0001));
  }
}
#endif

TEST_CASE("Cosine distance - public API", "[distance][cosine][api]") {
  SECTION("768d vectors") {
    constexpr size_t dim = 768;
    std::vector<float> a(dim);
    std::vector<float> b(dim);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < dim; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }

    float api_result = quiverdb::cosine_distance(a.data(), b.data(), dim);
    float scalar_result = quiverdb::cosine_distance_scalar(a.data(), b.data(), dim);

    REQUIRE(api_result == Approx(scalar_result).epsilon(0.0001));
  }
}
