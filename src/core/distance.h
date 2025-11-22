// QuiverDB - Fast vector database for edge AI
// Copyright (c) 2025 Anton Tsvetkov
// SPDX-License-Identifier: MIT

#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define QUIVER_ARM_NEON
#elif defined(__AVX2__)
#include <immintrin.h>
#define QUIVER_AVX2
#endif

// FMA detection
#if defined(__FMA__) && defined(QUIVER_AVX2)
#define QUIVER_FMA
#endif

// Compiler-specific restrict keyword
#if defined(_MSC_VER)
#define QUIVER_RESTRICT __restrict
#else
#define QUIVER_RESTRICT __restrict__
#endif

namespace quiverdb {

// ============================================================================
// SIMD Helper Functions
// ============================================================================

#ifdef QUIVER_ARM_NEON
// Horizontal sum for NEON - compatible with both ARMv7 and ARMv8
[[nodiscard]] inline float hsum_f32_neon(float32x4_t v) noexcept {
#if defined(__aarch64__)
  // ARMv8: Use vaddvq_f32 (faster single instruction)
  return vaddvq_f32(v);
#else
  // ARMv7: Manual reduction for compatibility
  float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  sum = vpadd_f32(sum, sum);
  return vget_lane_f32(sum, 0);
#endif
}
#endif

#ifdef QUIVER_AVX2
// Horizontal sum for AVX2 - reduces __m256 to single float
[[nodiscard]] inline float hsum_f32_avx2(__m256 v) noexcept {
  __m128 low = _mm256_castps256_ps128(v);
  __m128 high = _mm256_extractf128_ps(v, 1);
  __m128 sum128 = _mm_add_ps(low, high);
  __m128 shuf = _mm_movehdup_ps(sum128);
  __m128 sums = _mm_add_ps(sum128, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}
#endif

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
    sum_vec = vmlaq_f32(sum_vec, diff, diff); // FMA: sum += diff * diff
  }

  float sum = hsum_f32_neon(sum_vec);
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
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
#ifdef QUIVER_FMA
    sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec); // FMA: sum += diff * diff
#else
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(diff, diff));
#endif
  }

  float sum = hsum_f32_avx2(sum_vec);
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

// ============================================================================
// Dot Product - Scalar Implementation
// ============================================================================

[[nodiscard]] inline float dot_product_scalar(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum += a[i] * b[i];
  }

  return sum;
}

// ============================================================================
// Dot Product - ARM NEON Implementation
// ============================================================================

#ifdef QUIVER_ARM_NEON
[[nodiscard]] inline float dot_product_neon(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
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
    sum_vec = vmlaq_f32(sum_vec, va, vb); // FMA: sum += a * b
  }

  float sum = hsum_f32_neon(sum_vec);
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }

  return sum;
}
#endif

// ============================================================================
// Dot Product - AVX2 Implementation
// ============================================================================

#ifdef QUIVER_AVX2
[[nodiscard]] inline float dot_product_avx2(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

  __m256 sum_vec = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
#ifdef QUIVER_FMA
    sum_vec = _mm256_fmadd_ps(va, vb, sum_vec); // FMA: sum += a * b
#else
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
#endif
  }

  float sum = hsum_f32_avx2(sum_vec);
  for (; i < dim; ++i) {
    sum += a[i] * b[i];
  }

  return sum;
}
#endif

// ============================================================================
// Dot Product - Public API
// ============================================================================

/**
 * @brief Computes dot product (inner product) of two vectors.
 *
 * Returns sum(a[i] * b[i])
 *
 * This function automatically selects the best SIMD implementation available
 * for the current platform (ARM NEON, x86 AVX2, or scalar fallback).
 *
 * @param a First vector (must be non-null)
 * @param b Second vector (must be non-null)
 * @param dim Dimension of vectors (number of elements, >= 0)
 * @return Dot product of vectors a and b
 *
 * @pre a != nullptr
 * @pre b != nullptr
 * @pre Vectors a and b must have at least dim elements accessible
 */
[[nodiscard]] inline float dot_product(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

#ifdef QUIVER_ARM_NEON
  return dot_product_neon(a, b, dim);
#elif defined(QUIVER_AVX2)
  return dot_product_avx2(a, b, dim);
#else
  return dot_product_scalar(a, b, dim);
#endif
}

// ============================================================================
// Cosine Distance - Scalar Implementation
// ============================================================================

[[nodiscard]] inline float cosine_distance_scalar(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

  float dot = 0.0f;
  float mag_a = 0.0f;
  float mag_b = 0.0f;

  for (size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
    mag_a += a[i] * a[i];
    mag_b += b[i] * b[i];
  }

  // Avoid division by zero
  const float epsilon = 1e-8f;
  float magnitude_product = mag_a * mag_b;
  if (magnitude_product < epsilon) {
    return 1.0f; // Maximum distance for zero vectors
  }

  float cosine_sim = dot / sqrtf(magnitude_product);
  // Clamp to [-1, 1] to handle numerical errors
  cosine_sim = (cosine_sim > 1.0f) ? 1.0f : (cosine_sim < -1.0f) ? -1.0f : cosine_sim;

  return 1.0f - cosine_sim;
}

// ============================================================================
// Cosine Distance - ARM NEON Implementation
// ============================================================================

#ifdef QUIVER_ARM_NEON
[[nodiscard]] inline float cosine_distance_neon(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

  float32x4_t dot_vec = vdupq_n_f32(0.0f);
  float32x4_t mag_a_vec = vdupq_n_f32(0.0f);
  float32x4_t mag_b_vec = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 4 <= dim; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);

    dot_vec = vmlaq_f32(dot_vec, va, vb);        // FMA: dot += a * b
    mag_a_vec = vmlaq_f32(mag_a_vec, va, va);    // FMA: mag_a += a * a
    mag_b_vec = vmlaq_f32(mag_b_vec, vb, vb);    // FMA: mag_b += b * b
  }

  float dot = hsum_f32_neon(dot_vec);
  float mag_a = hsum_f32_neon(mag_a_vec);
  float mag_b = hsum_f32_neon(mag_b_vec);

  for (; i < dim; ++i) {
    dot += a[i] * b[i];
    mag_a += a[i] * a[i];
    mag_b += b[i] * b[i];
  }

  const float epsilon = 1e-8f;
  float magnitude_product = mag_a * mag_b;
  if (magnitude_product < epsilon) {
    return 1.0f;
  }

  float cosine_sim = dot / sqrtf(magnitude_product);
  cosine_sim = (cosine_sim > 1.0f) ? 1.0f : (cosine_sim < -1.0f) ? -1.0f : cosine_sim;

  return 1.0f - cosine_sim;
}
#endif

// ============================================================================
// Cosine Distance - AVX2 Implementation
// ============================================================================

#ifdef QUIVER_AVX2
[[nodiscard]] inline float cosine_distance_avx2(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

  __m256 dot_vec = _mm256_setzero_ps();
  __m256 mag_a_vec = _mm256_setzero_ps();
  __m256 mag_b_vec = _mm256_setzero_ps();

  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);

#ifdef QUIVER_FMA
    dot_vec = _mm256_fmadd_ps(va, vb, dot_vec);       // FMA: dot += a * b
    mag_a_vec = _mm256_fmadd_ps(va, va, mag_a_vec);   // FMA: mag_a += a * a
    mag_b_vec = _mm256_fmadd_ps(vb, vb, mag_b_vec);   // FMA: mag_b += b * b
#else
    dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(va, vb));
    mag_a_vec = _mm256_add_ps(mag_a_vec, _mm256_mul_ps(va, va));
    mag_b_vec = _mm256_add_ps(mag_b_vec, _mm256_mul_ps(vb, vb));
#endif
  }

  float dot = hsum_f32_avx2(dot_vec);
  float mag_a = hsum_f32_avx2(mag_a_vec);
  float mag_b = hsum_f32_avx2(mag_b_vec);

  for (; i < dim; ++i) {
    dot += a[i] * b[i];
    mag_a += a[i] * a[i];
    mag_b += b[i] * b[i];
  }

  const float epsilon = 1e-8f;
  float magnitude_product = mag_a * mag_b;
  if (magnitude_product < epsilon) {
    return 1.0f;
  }

  float cosine_sim = dot / sqrtf(magnitude_product);
  cosine_sim = (cosine_sim > 1.0f) ? 1.0f : (cosine_sim < -1.0f) ? -1.0f : cosine_sim;

  return 1.0f - cosine_sim;
}
#endif

// ============================================================================
// Cosine Distance - Public API
// ============================================================================

/**
 * @brief Computes cosine distance between two vectors.
 *
 * Cosine distance = 1 - cosine_similarity
 * where cosine_similarity = dot(a, b) / (||a|| * ||b||)
 *
 * Returns a value in range [0, 2]:
 * - 0 means vectors point in the same direction (identical after normalization)
 * - 1 means vectors are orthogonal
 * - 2 means vectors point in opposite directions
 *
 * This function automatically selects the best SIMD implementation available
 * for the current platform (ARM NEON, x86 AVX2, or scalar fallback).
 *
 * @param a First vector (must be non-null)
 * @param b Second vector (must be non-null)
 * @param dim Dimension of vectors (number of elements, >= 0)
 * @return Cosine distance between vectors a and b
 *
 * @pre a != nullptr
 * @pre b != nullptr
 * @pre Vectors a and b must have at least dim elements accessible
 *
 * @note Returns 1.0 (maximum distance) if either vector has zero magnitude.
 */
[[nodiscard]] inline float cosine_distance(const float* QUIVER_RESTRICT a, const float* QUIVER_RESTRICT b, size_t dim) noexcept {
  assert(a != nullptr && "Vector a must not be null");
  assert(b != nullptr && "Vector b must not be null");

  if (dim == 0) {
    return 0.0f;
  }

#ifdef QUIVER_ARM_NEON
  return cosine_distance_neon(a, b, dim);
#elif defined(QUIVER_AVX2)
  return cosine_distance_avx2(a, b, dim);
#else
  return cosine_distance_scalar(a, b, dim);
#endif
}

} // namespace quiverdb
