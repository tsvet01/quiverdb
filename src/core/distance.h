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
    float32x4_t prod = vmulq_f32(va, vb);
    sum_vec = vaddq_f32(sum_vec, prod);
  }

  float sum = vaddvq_f32(sum_vec);
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
    __m256 prod = _mm256_mul_ps(va, vb);
    sum_vec = _mm256_add_ps(sum_vec, prod);
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

    dot_vec = vaddq_f32(dot_vec, vmulq_f32(va, vb));
    mag_a_vec = vaddq_f32(mag_a_vec, vmulq_f32(va, va));
    mag_b_vec = vaddq_f32(mag_b_vec, vmulq_f32(vb, vb));
  }

  float dot = vaddvq_f32(dot_vec);
  float mag_a = vaddvq_f32(mag_a_vec);
  float mag_b = vaddvq_f32(mag_b_vec);

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

    dot_vec = _mm256_add_ps(dot_vec, _mm256_mul_ps(va, vb));
    mag_a_vec = _mm256_add_ps(mag_a_vec, _mm256_mul_ps(va, va));
    mag_b_vec = _mm256_add_ps(mag_b_vec, _mm256_mul_ps(vb, vb));
  }

  // Horizontal sum for dot product
  __m128 dot_low = _mm256_castps256_ps128(dot_vec);
  __m128 dot_high = _mm256_extractf128_ps(dot_vec, 1);
  __m128 dot_sum128 = _mm_add_ps(dot_low, dot_high);
  __m128 dot_shuf = _mm_movehdup_ps(dot_sum128);
  __m128 dot_sums = _mm_add_ps(dot_sum128, dot_shuf);
  dot_shuf = _mm_movehl_ps(dot_shuf, dot_sums);
  dot_sums = _mm_add_ss(dot_sums, dot_shuf);
  float dot = _mm_cvtss_f32(dot_sums);

  // Horizontal sum for mag_a
  __m128 mag_a_low = _mm256_castps256_ps128(mag_a_vec);
  __m128 mag_a_high = _mm256_extractf128_ps(mag_a_vec, 1);
  __m128 mag_a_sum128 = _mm_add_ps(mag_a_low, mag_a_high);
  __m128 mag_a_shuf = _mm_movehdup_ps(mag_a_sum128);
  __m128 mag_a_sums = _mm_add_ps(mag_a_sum128, mag_a_shuf);
  mag_a_shuf = _mm_movehl_ps(mag_a_shuf, mag_a_sums);
  mag_a_sums = _mm_add_ss(mag_a_sums, mag_a_shuf);
  float mag_a = _mm_cvtss_f32(mag_a_sums);

  // Horizontal sum for mag_b
  __m128 mag_b_low = _mm256_castps256_ps128(mag_b_vec);
  __m128 mag_b_high = _mm256_extractf128_ps(mag_b_vec, 1);
  __m128 mag_b_sum128 = _mm_add_ps(mag_b_low, mag_b_high);
  __m128 mag_b_shuf = _mm_movehdup_ps(mag_b_sum128);
  __m128 mag_b_sums = _mm_add_ps(mag_b_sum128, mag_b_shuf);
  mag_b_shuf = _mm_movehl_ps(mag_b_shuf, mag_b_sums);
  mag_b_sums = _mm_add_ss(mag_b_sums, mag_b_shuf);
  float mag_b = _mm_cvtss_f32(mag_b_sums);

  // Remainder
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
