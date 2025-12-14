// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once

#ifdef __CUDACC__
#define QUIVER_CUDA 1

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quiverdb {
namespace gpu {

#define CU_CHK(x) do{cudaError_t e=x;if(e)throw std::runtime_error(cudaGetErrorString(e));}while(0)

__global__ void l2_k(const float4* __restrict__ q, const float4* __restrict__ v,
                     float* __restrict__ r, uint32_t d4, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 s = make_float4(0,0,0,0);
  uint32_t o = i * d4;
  for (uint32_t j = 0; j < d4; ++j) {
    float4 x = make_float4(q[j].x-v[o+j].x, q[j].y-v[o+j].y, q[j].z-v[o+j].z, q[j].w-v[o+j].w);
    s.x += x.x*x.x; s.y += x.y*x.y; s.z += x.z*x.z; s.w += x.w*x.w;
  }
  r[i] = s.x + s.y + s.z + s.w;
}

__global__ void dot_k(const float4* __restrict__ q, const float4* __restrict__ v,
                      float* __restrict__ r, uint32_t d4, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 s = make_float4(0,0,0,0);
  uint32_t o = i * d4;
  for (uint32_t j = 0; j < d4; ++j) {
    s.x += q[j].x*v[o+j].x; s.y += q[j].y*v[o+j].y;
    s.z += q[j].z*v[o+j].z; s.w += q[j].w*v[o+j].w;
  }
  r[i] = -(s.x + s.y + s.z + s.w);
}

__global__ void cos_k(const float4* __restrict__ q, const float4* __restrict__ v,
                      float* __restrict__ r, uint32_t d4, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 d = make_float4(0,0,0,0), nq = d, nv = d;
  uint32_t o = i * d4;
  for (uint32_t j = 0; j < d4; ++j) {
    float4 a = q[j], b = v[o+j];
    d.x += a.x*b.x; d.y += a.y*b.y; d.z += a.z*b.z; d.w += a.w*b.w;
    nq.x += a.x*a.x; nq.y += a.y*a.y; nq.z += a.z*a.z; nq.w += a.w*a.w;
    nv.x += b.x*b.x; nv.y += b.y*b.y; nv.z += b.z*b.z; nv.w += b.w*b.w;
  }
  float dot = d.x+d.y+d.z+d.w, na = nq.x+nq.y+nq.z+nq.w, nb = nv.x+nv.y+nv.z+nv.w;
  float dn = na * nb;
  float sim = (dn < 1e-12f) ? 0.0f : dot * rsqrtf(dn);
  r[i] = 1.0f - fminf(fmaxf(sim, -1.0f), 1.0f);
}

enum class CudaMetric { L2, COSINE, DOT };

class CudaCompute {
public:
  static CudaCompute& get() { static CudaCompute c; return c; }
  bool ok() const { return ok_; }

  // Persistent buffer API
  float* upload(const float* v, size_t n, size_t d) {
    float* dv; CU_CHK(cudaMalloc(&dv, n*d*sizeof(float)));
    CU_CHK(cudaMemcpy(dv, v, n*d*sizeof(float), cudaMemcpyHostToDevice));
    return dv;
  }
  void free(float* dv) { cudaFree(dv); }

  std::vector<float> search(const float* q, float* dv, size_t d, size_t n, CudaMetric m) {
    if (!ok_ || d%4) throw std::runtime_error("CUDA unavailable or dim%4!=0");
    std::vector<float> out(n);
    float *dq, *dr;
    CU_CHK(cudaMalloc(&dq, d*sizeof(float)));
    CU_CHK(cudaMalloc(&dr, n*sizeof(float)));
    CU_CHK(cudaMemcpy(dq, q, d*sizeof(float), cudaMemcpyHostToDevice));
    int t = 256, b = (n + t - 1) / t;
    uint32_t d4 = d / 4;
    switch (m) {
      case CudaMetric::L2: l2_k<<<b,t>>>((float4*)dq,(float4*)dv,dr,d4,n); break;
      case CudaMetric::DOT: dot_k<<<b,t>>>((float4*)dq,(float4*)dv,dr,d4,n); break;
      case CudaMetric::COSINE: cos_k<<<b,t>>>((float4*)dq,(float4*)dv,dr,d4,n); break;
    }
    CU_CHK(cudaGetLastError());
    CU_CHK(cudaMemcpy(out.data(), dr, n*sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dq); cudaFree(dr);
    return out;
  }

private:
  CudaCompute() { int c=0; ok_=(cudaGetDeviceCount(&c)==cudaSuccess && c>0); }
  bool ok_ = false;
};

inline bool cuda_available() { return CudaCompute::get().ok(); }

} // namespace gpu
} // namespace quiverdb

#endif
