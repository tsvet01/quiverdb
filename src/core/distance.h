// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once
#include <algorithm>
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

#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

namespace quiverdb {

#ifdef QUIVER_ARM_NEON
[[nodiscard]] inline float hsum(float32x4_t v) noexcept {
#if defined(__aarch64__)
  return vaddvq_f32(v);
#else
  float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
  return vget_lane_f32(vpadd_f32(r, r), 0);
#endif
}
#endif

#ifdef QUIVER_AVX2
[[nodiscard]] inline float hsum(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  lo = _mm_add_ps(lo, shuf);
  return _mm_cvtss_f32(_mm_add_ss(lo, _mm_movehl_ps(shuf, lo)));
}
#endif

[[nodiscard]] inline float l2_sq(const float* RESTRICT a, const float* RESTRICT b, size_t n) noexcept {
  assert(a && b);
  float sum = 0.0f;
  size_t i = 0;

#ifdef QUIVER_ARM_NEON
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    acc = vmlaq_f32(acc, d, d);
  }
  sum = hsum(acc);
#elif defined(QUIVER_AVX2)
  __m256 acc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc = _mm256_fmadd_ps(d, d, acc);
  }
  sum = hsum(acc);
#endif

  for (; i < n; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

[[nodiscard]] inline float dot_product(const float* RESTRICT a, const float* RESTRICT b, size_t n) noexcept {
  assert(a && b);
  float sum = 0.0f;
  size_t i = 0;

#ifdef QUIVER_ARM_NEON
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4)
    acc = vmlaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
  sum = hsum(acc);
#elif defined(QUIVER_AVX2)
  __m256 acc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  sum = hsum(acc);
#endif

  for (; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
}

[[nodiscard]] inline float cosine_distance(const float* RESTRICT a, const float* RESTRICT b, size_t n) noexcept {
  assert(a && b);
  float dot = 0.0f, na = 0.0f, nb = 0.0f;
  size_t i = 0;

#ifdef QUIVER_ARM_NEON
  float32x4_t vdot = vdupq_n_f32(0.0f), vna = vdupq_n_f32(0.0f), vnb = vdupq_n_f32(0.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i), vb = vld1q_f32(b + i);
    vdot = vmlaq_f32(vdot, va, vb);
    vna = vmlaq_f32(vna, va, va);
    vnb = vmlaq_f32(vnb, vb, vb);
  }
  dot = hsum(vdot); na = hsum(vna); nb = hsum(vnb);
#elif defined(QUIVER_AVX2)
  __m256 vdot = _mm256_setzero_ps(), vna = _mm256_setzero_ps(), vnb = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
    vdot = _mm256_fmadd_ps(va, vb, vdot);
    vna = _mm256_fmadd_ps(va, va, vna);
    vnb = _mm256_fmadd_ps(vb, vb, vnb);
  }
  dot = hsum(vdot); na = hsum(vna); nb = hsum(vnb);
#endif

  for (; i < n; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }

  float denom = na * nb;
  if (denom < 1e-12f) return 1.0f;
  float sim = dot / sqrtf(denom);
  return 1.0f - std::clamp(sim, -1.0f, 1.0f);
}

} // namespace quiverdb
