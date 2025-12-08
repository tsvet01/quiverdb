// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#include "core/distance.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>
#include <vector>

using Catch::Approx;

TEST_CASE("l2_sq", "[distance]") {
  SECTION("3d") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    REQUIRE(quiverdb::l2_sq(a, b, 3) == Approx(27.0f));
  }

  SECTION("identical") {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    REQUIRE(quiverdb::l2_sq(a, a, 4) == Approx(0.0f).margin(1e-6));
  }

  SECTION("768d") {
    std::vector<float> a(768), b(768);
    for (size_t i = 0; i < 768; ++i) {
      a[i] = static_cast<float>(i) / 768;
      b[i] = static_cast<float>(i) / 768 + 0.5f;
    }
    REQUIRE(quiverdb::l2_sq(a.data(), b.data(), 768) == Approx(192.0f));
  }

  SECTION("negative") {
    float a[] = {-1.0f, -2.0f, -3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    REQUIRE(quiverdb::l2_sq(a, b, 3) == Approx(56.0f));
  }

  SECTION("dim=0") {
    float a[] = {1.0f}, b[] = {2.0f};
    REQUIRE(quiverdb::l2_sq(a, b, 0) == 0.0f);
  }

  SECTION("dim=1") {
    float a[] = {3.0f}, b[] = {7.0f};
    REQUIRE(quiverdb::l2_sq(a, b, 1) == Approx(16.0f));
  }

  SECTION("dim=2") {
    float a[] = {1.0f, 2.0f}, b[] = {4.0f, 6.0f};
    REQUIRE(quiverdb::l2_sq(a, b, 2) == Approx(25.0f));
  }

  SECTION("non-aligned dim") {
    std::vector<float> a(773), b(773);
    for (size_t i = 0; i < 773; ++i) {
      a[i] = static_cast<float>(i);
      b[i] = static_cast<float>(i) + 1.0f;
    }
    REQUIRE(quiverdb::l2_sq(a.data(), b.data(), 773) == Approx(773.0f));
  }

  SECTION("1536d random") {
    std::vector<float> a(1536), b(1536);
    std::mt19937 gen(99999);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < 1536; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }
    float r = quiverdb::l2_sq(a.data(), b.data(), 1536);
    REQUIRE(r > 0.0f);
  }
}

TEST_CASE("dot_product", "[distance]") {
  SECTION("3d") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    REQUIRE(quiverdb::dot_product(a, b, 3) == Approx(32.0f));
  }

  SECTION("orthogonal") {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    REQUIRE(quiverdb::dot_product(a, b, 3) == Approx(0.0f).margin(1e-6));
  }

  SECTION("identical") {
    float a[] = {2.0f, 3.0f, 4.0f};
    REQUIRE(quiverdb::dot_product(a, a, 3) == Approx(29.0f));
  }

  SECTION("negative") {
    float a[] = {-1.0f, -2.0f, -3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    REQUIRE(quiverdb::dot_product(a, b, 3) == Approx(-14.0f));
  }

  SECTION("dim=0") {
    float a[] = {1.0f}, b[] = {2.0f};
    REQUIRE(quiverdb::dot_product(a, b, 0) == 0.0f);
  }

  SECTION("768d") {
    std::vector<float> a(768), b(768);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < 768; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }
    float r = quiverdb::dot_product(a.data(), b.data(), 768);
    REQUIRE(std::abs(r) < 768.0f);
  }

  SECTION("non-aligned dim") {
    std::vector<float> a(773, 1.0f), b(773, 2.0f);
    REQUIRE(quiverdb::dot_product(a.data(), b.data(), 773) == Approx(1546.0f));
  }
}

TEST_CASE("cosine_distance", "[distance]") {
  SECTION("identical") {
    float a[] = {1.0f, 2.0f, 3.0f};
    REQUIRE(quiverdb::cosine_distance(a, a, 3) == Approx(0.0f).margin(1e-6));
  }

  SECTION("scaled") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {2.0f, 4.0f, 6.0f};
    REQUIRE(quiverdb::cosine_distance(a, b, 3) == Approx(0.0f).margin(1e-6));
  }

  SECTION("orthogonal") {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    REQUIRE(quiverdb::cosine_distance(a, b, 3) == Approx(1.0f));
  }

  SECTION("opposite") {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {-1.0f, 0.0f, 0.0f};
    REQUIRE(quiverdb::cosine_distance(a, b, 3) == Approx(2.0f));
  }

  SECTION("zero vector") {
    float a[] = {0.0f, 0.0f, 0.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    REQUIRE(quiverdb::cosine_distance(a, b, 3) == Approx(1.0f));
  }

  SECTION("general") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    // cos = 32/sqrt(14*77) ≈ 0.9746, dist = 1 - 0.9746 ≈ 0.0254
    REQUIRE(quiverdb::cosine_distance(a, b, 3) == Approx(0.0254).epsilon(0.01));
  }

  SECTION("768d") {
    std::vector<float> a(768), b(768);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < 768; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }
    float r = quiverdb::cosine_distance(a.data(), b.data(), 768);
    REQUIRE(r >= 0.0f);
    REQUIRE(r <= 2.0f);
  }

  SECTION("non-aligned dim") {
    std::vector<float> a(773), b(773);
    std::mt19937 gen(99999);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < 773; ++i) {
      a[i] = dis(gen);
      b[i] = dis(gen);
    }
    float r = quiverdb::cosine_distance(a.data(), b.data(), 773);
    REQUIRE(r >= 0.0f);
    REQUIRE(r <= 2.0f);
  }
}
