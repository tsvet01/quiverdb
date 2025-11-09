#pragma once
#include <cassert>
#include <cstddef>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define QUIVER_ARM_NEON
#elif defined(__AVX2__)
#include <immintrin.h>
#define QUIVER_AVX2
#endif

// Compiler-specific restrict keyword
#if defined(_MSC_VER)
#define QUIVER_RESTRICT __restrict
#else
#define QUIVER_RESTRICT __restrict__
#endif

namespace quiverdb {

// ============================================================================
// Scalar Implementation
// ============================================================================

[[nodiscard]] inline float l2_sq_scalar(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

// ============================================================================
// ARM NEON Implementation (Apple Silicon M1/M2/M3)
// ============================================================================

#ifdef QUIVER_ARM_NEON
[[nodiscard]] inline float l2_sq_neon(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

  float32x4_t sum_vec = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t diff = vsubq_f32(va, vb);
    float32x4_t sq = vmulq_f32(diff, diff);
    sum_vec = vaddq_f32(sum_vec, sq);
  }

  float sum = vaddvq_f32(sum_vec);
  for (; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}
#endif

// ============================================================================
// AVX2 Implementation (Intel/AMD x86_64)
// ============================================================================

#ifdef QUIVER_AVX2
[[nodiscard]] inline float l2_sq_avx2(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

  __m256 sum_vec = _mm256_setzero_ps();
  size_t i = 0;
  // Processing 8 floats at a time
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    __m256 sq = _mm256_mul_ps(diff, diff);
    sum_vec = _mm256_add_ps(sum_vec, sq);
  }

  __m128 low = _mm256_castps256_ps128(sum_vec);
  __m128 high = _mm256_extractf128_ps(sum_vec, 1);
  __m128 sum128 = _mm_add_ps(low, high);

  __m128 shuf = _mm_movehdup_ps(sum128);
  __m128 sums = _mm_add_ps(sum128, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);

  float sum = _mm_cvtss_f32(sums);
  for (; i < dim; ++i) {
    const float diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}
#endif

// ============================================================================
// Public API: Auto-selects best implementation for current platform
// ============================================================================

/**
 * @brief Computes squared L2 (Euclidean) distance between two vectors.
 *
 * Returns the sum of squared differences: sum((a[i] - b[i])^2)
 *
 * This function automatically selects the best SIMD implementation available
 * for the current platform (ARM NEON, x86 AVX2, or scalar fallback).
 *
 * @param a First vector (must be non-null)
 * @param b Second vector (must be non-null)
 * @param dim Dimension of vectors (number of elements to compare, >= 0)
 * @return Squared L2 distance between vectors a and b
 *
 * @pre a != nullptr
 * @pre b != nullptr
 * @pre Vectors a and b must have at least dim elements accessible
 *
 * @note For actual Euclidean distance, take sqrt() of the result.
 */
[[nodiscard]] inline float l2_sq(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

#ifdef QUIVER_ARM_NEON
  return l2_sq_neon(a, b, dim);
#elif defined(QUIVER_AVX2)
  return l2_sq_avx2(a, b, dim);
#else
  return l2_sq_scalar(a, b, dim);
#endif
}

} // namespace quiverdb
