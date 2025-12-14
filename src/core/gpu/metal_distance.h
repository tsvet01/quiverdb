// QuiverDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once

#ifdef __APPLE__
#include <TargetConditionals.h>
#if TARGET_OS_MAC || TARGET_OS_IPHONE
#define QUIVER_METAL 1

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quiverdb {
namespace gpu {

enum class MetalMetric { L2, COSINE, DOT };

class MetalCompute {
public:
  static MetalCompute& get() { static MetalCompute c; return c; }
  bool ok() const { return dev_ != nil; }

  std::vector<float> l2(const float* q, const float* v, size_t d, size_t n) {
    return run(q, v, d, n, l2_); }
  std::vector<float> dot(const float* q, const float* v, size_t d, size_t n) {
    return run(q, v, d, n, dot_); }
  std::vector<float> cos(const float* q, const float* v, size_t d, size_t n) {
    return run(q, v, d, n, cos_); }

  // Persistent buffer API - vectors stay on GPU
  id<MTLBuffer> upload(const float* v, size_t n, size_t d) {
    return [dev_ newBufferWithBytes:v length:n*d*sizeof(float) options:MTLResourceStorageModeShared];
  }

  std::vector<float> search(const float* q, id<MTLBuffer> vbuf, size_t d, size_t n, MetalMetric m) {
    if (!ok() || d % 4 != 0) throw std::runtime_error("Metal unavailable or dim%4!=0");
    std::vector<float> out(n);
    @autoreleasepool {
      id<MTLBuffer> qb = [dev_ newBufferWithBytes:q length:d*sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> rb = [dev_ newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
      uint32_t d4 = (uint32_t)(d/4);
      id<MTLBuffer> db = [dev_ newBufferWithBytes:&d4 length:4 options:MTLResourceStorageModeShared];
      id<MTLComputePipelineState> p = (m==MetalMetric::L2)?l2_:(m==MetalMetric::DOT)?dot_:cos_;
      id<MTLCommandBuffer> cmd = [q_ commandBuffer];
      id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
      [e setComputePipelineState:p];
      [e setBuffer:qb offset:0 atIndex:0];
      [e setBuffer:vbuf offset:0 atIndex:1];
      [e setBuffer:rb offset:0 atIndex:2];
      [e setBuffer:db offset:0 atIndex:3];
      MTLSize grid = MTLSizeMake(n,1,1);
      MTLSize group = MTLSizeMake(MIN(p.maxTotalThreadsPerThreadgroup,n),1,1);
      [e dispatchThreads:grid threadsPerThreadgroup:group];
      [e endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      memcpy(out.data(), [rb contents], n*sizeof(float));
    }
    return out;
  }

private:
  MetalCompute() { init(); }
  void init() {
    @autoreleasepool {
      dev_ = MTLCreateSystemDefaultDevice();
      if (!dev_) return;
      q_ = [dev_ newCommandQueue];
      if (!q_) { dev_=nil; return; }
      NSError* err = nil;
      NSString* src = @R"(
#include <metal_stdlib>
using namespace metal;
kernel void l2(device const float4* q[[buffer(0)]], device const float4* v[[buffer(1)]],
  device float* r[[buffer(2)]], constant uint& d4[[buffer(3)]], uint i[[thread_position_in_grid]]) {
  float4 s=0; uint o=i*d4; for(uint j=0;j<d4;++j){float4 x=q[j]-v[o+j];s+=x*x;} r[i]=s.x+s.y+s.z+s.w;
}
kernel void dp(device const float4* q[[buffer(0)]], device const float4* v[[buffer(1)]],
  device float* r[[buffer(2)]], constant uint& d4[[buffer(3)]], uint i[[thread_position_in_grid]]) {
  float4 s=0; uint o=i*d4; for(uint j=0;j<d4;++j)s+=q[j]*v[o+j]; r[i]=-(s.x+s.y+s.z+s.w);
}
kernel void cs(device const float4* q[[buffer(0)]], device const float4* v[[buffer(1)]],
  device float* r[[buffer(2)]], constant uint& d4[[buffer(3)]], uint i[[thread_position_in_grid]]) {
  float4 d=0,nq=0,nv=0; uint o=i*d4;
  for(uint j=0;j<d4;++j){float4 a=q[j],b=v[o+j];d+=a*b;nq+=a*a;nv+=b*b;}
  float dot=d.x+d.y+d.z+d.w,na=nq.x+nq.y+nq.z+nq.w,nb=nv.x+nv.y+nv.z+nv.w;
  float dn=na*nb; r[i]=1.0f-clamp((dn<1e-12f)?0.0f:dot*rsqrt(dn),-1.0f,1.0f);
}
      )";
      id<MTLLibrary> lib = [dev_ newLibraryWithSource:src options:nil error:&err];
      if (!lib) { dev_=nil; return; }
      l2_ = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"l2"] error:&err];
      dot_ = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"dp"] error:&err];
      cos_ = [dev_ newComputePipelineStateWithFunction:[lib newFunctionWithName:@"cs"] error:&err];
      if (!l2_||!dot_||!cos_) dev_=nil;
    }
  }

  std::vector<float> run(const float* q, const float* v, size_t d, size_t n,
                         id<MTLComputePipelineState> p) {
    if (!ok() || d % 4 != 0) throw std::runtime_error("Metal unavailable or dim%4!=0");
    std::vector<float> out(n);
    @autoreleasepool {
      id<MTLBuffer> qb = [dev_ newBufferWithBytes:q length:d*sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> vb = [dev_ newBufferWithBytes:v length:n*d*sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> rb = [dev_ newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
      uint32_t d4 = (uint32_t)(d/4);
      id<MTLBuffer> db = [dev_ newBufferWithBytes:&d4 length:4 options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cmd = [q_ commandBuffer];
      id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
      [e setComputePipelineState:p];
      [e setBuffer:qb offset:0 atIndex:0];
      [e setBuffer:vb offset:0 atIndex:1];
      [e setBuffer:rb offset:0 atIndex:2];
      [e setBuffer:db offset:0 atIndex:3];
      MTLSize grid = MTLSizeMake(n,1,1);
      MTLSize group = MTLSizeMake(MIN(p.maxTotalThreadsPerThreadgroup,n),1,1);
      [e dispatchThreads:grid threadsPerThreadgroup:group];
      [e endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      memcpy(out.data(), [rb contents], n*sizeof(float));
    }
    return out;
  }

  id<MTLDevice> dev_ = nil;
  id<MTLCommandQueue> q_ = nil;
  id<MTLComputePipelineState> l2_ = nil, dot_ = nil, cos_ = nil;
};

inline bool metal_available() { return MetalCompute::get().ok(); }

} // namespace gpu
} // namespace quiverdb

#endif
#endif
