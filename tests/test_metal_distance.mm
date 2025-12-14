// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#import <Foundation/Foundation.h>
#include "core/gpu/metal_distance.h"
#include "core/distance.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace quiverdb;
using namespace std::chrono;

bool approx(float a, float b, float eps = 1e-4f) { return std::abs(a - b) < eps; }

void test_correctness() {
  if (!gpu::metal_available()) { std::cout << "Metal: SKIP\n"; return; }
  constexpr size_t D = 128, N = 1000;
  std::mt19937 g(42);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> q(D), v(N * D);
  for (auto& x : q) x = u(g);
  for (auto& x : v) x = u(g);

  auto r = gpu::MetalCompute::get().l2(q.data(), v.data(), D, N);
  for (size_t i = 0; i < N; ++i) {
    float cpu = l2_sq(q.data(), v.data() + i * D, D);
    if (!approx(r[i], cpu)) { std::cout << "L2 FAIL\n"; return; }
  }
  std::cout << "L2: PASS\n";

  r = gpu::MetalCompute::get().cos(q.data(), v.data(), D, N);
  for (size_t i = 0; i < N; ++i) {
    float cpu = cosine_distance(q.data(), v.data() + i * D, D);
    if (!approx(r[i], cpu, 1e-3f)) { std::cout << "Cosine FAIL\n"; return; }
  }
  std::cout << "Cosine: PASS\n";

  r = gpu::MetalCompute::get().dot(q.data(), v.data(), D, N);
  for (size_t i = 0; i < N; ++i) {
    float cpu = -dot_product(q.data(), v.data() + i * D, D);
    if (!approx(r[i], cpu)) { std::cout << "Dot FAIL\n"; return; }
  }
  std::cout << "Dot: PASS\n";
}

void bench_persistent() {
  if (!gpu::metal_available()) return;
  std::cout << "\n=== Persistent GPU Buffer Benchmark ===\n";
  constexpr size_t D = 768;
  std::vector<size_t> sizes = {10000, 100000, 500000};
  std::mt19937 g(42);
  std::uniform_real_distribution<float> u(-1, 1);

  for (size_t N : sizes) {
    std::vector<float> q(D), v(N * D);
    for (auto& x : q) x = u(g);
    for (auto& x : v) x = u(g);

    // CPU baseline
    auto t0 = high_resolution_clock::now();
    std::vector<float> cpu_r(N);
    for (size_t i = 0; i < N; ++i) cpu_r[i] = l2_sq(q.data(), v.data() + i * D, D);
    auto cpu_us = duration_cast<microseconds>(high_resolution_clock::now() - t0).count();

    // GPU with transfer each time
    t0 = high_resolution_clock::now();
    auto gpu_r = gpu::MetalCompute::get().l2(q.data(), v.data(), D, N);
    auto gpu_xfer_us = duration_cast<microseconds>(high_resolution_clock::now() - t0).count();

    // GPU with persistent buffer (upload once)
    t0 = high_resolution_clock::now();
    id<MTLBuffer> vbuf = gpu::MetalCompute::get().upload(v.data(), N, D);
    auto upload_us = duration_cast<microseconds>(high_resolution_clock::now() - t0).count();

    // Multiple queries on persistent buffer
    constexpr int QUERIES = 10;
    t0 = high_resolution_clock::now();
    for (int i = 0; i < QUERIES; ++i) {
      gpu::MetalCompute::get().search(q.data(), vbuf, D, N, gpu::MetalMetric::L2);
    }
    auto gpu_persist_us = duration_cast<microseconds>(high_resolution_clock::now() - t0).count() / QUERIES;

    float speedup = (float)cpu_us / gpu_persist_us;
    std::cout << N << " vecs: CPU=" << cpu_us << "us, GPU(xfer)=" << gpu_xfer_us
              << "us, GPU(persist)=" << gpu_persist_us << "us, upload=" << upload_us
              << "us, speedup=" << speedup << "x\n";
  }
}

int main() {
  std::cout << "QuiverDB Metal GPU Tests\n========================\n";
  std::cout << "Metal: " << (gpu::metal_available() ? "YES" : "NO") << "\n\n";
  test_correctness();
  bench_persistent();
  return 0;
}
