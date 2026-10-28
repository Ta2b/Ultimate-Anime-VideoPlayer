#include <cuda_fp8.h>

#include "fp8_fused.cuh"

namespace ajp::gpu::cuda {
namespace {

__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Deux octets E4M3 -> deux float (conversion matérielle sm_89 via half2).
__device__ __forceinline__ float2 fp8x2_to_float2(uint16_t v) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v), __NV_E4M3);
  return __half22float2(__half2(h));
}

__device__ __forceinline__ uint16_t float2_to_fp8x2(float a, float b) {
  return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3));
}

template <bool kFast>
__device__ __forceinline__ float tanh_sel(float x) {
  if constexpr (kFast) return tanh_approx(x);
  return tanhf(x);
}

// Un thread = 16 éléments : un chargement uint4 pour c3, skip, sortie ; deux pour att (FP16).
template <bool kFast>
__global__ void span_tail_kernel(const uint4* __restrict__ c3, const uint4* __restrict__ att,
                                 const uint4* __restrict__ skip, uint4* __restrict__ out, size_t n_vec,
                                 Fp8SpanTailScales sc) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_vec) return;
  uint4 vc = c3[i], vs = skip[i], va0 = att[2 * i], va1 = att[2 * i + 1];
  const uint16_t* pc = reinterpret_cast<const uint16_t*>(&vc);
  const uint16_t* ps = reinterpret_cast<const uint16_t*>(&vs);
  const __half2* pa0 = reinterpret_cast<const __half2*>(&va0);
  const __half2* pa1 = reinterpret_cast<const __half2*>(&va1);
  uint4 vo;
  uint16_t* po = reinterpret_cast<uint16_t*>(&vo);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    float2 c = fp8x2_to_float2(pc[k]);
    float2 s = fp8x2_to_float2(ps[k]);
    float2 a = __half22float2(k < 4 ? pa0[k] : pa1[k - 4]);
    float r0 = (c.x * sc.c3 + s.x * sc.skip) * tanh_sel<kFast>(a.x) * sc.out_inv;
    float r1 = (c.y * sc.c3 + s.y * sc.skip) * tanh_sel<kFast>(a.y) * sc.out_inv;
    po[k] = float2_to_fp8x2(r0, r1);
  }
  out[i] = vo;
}

__global__ void span_tail_att8_kernel(const uint4* __restrict__ c3, const uint4* __restrict__ att,
                                      const uint4* __restrict__ skip, uint4* __restrict__ out, size_t n_vec,
                                      Fp8SpanTailScales sc, float s_att) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_vec) return;
  uint4 vc = c3[i], vs = skip[i], va = att[i];
  const uint16_t* pc = reinterpret_cast<const uint16_t*>(&vc);
  const uint16_t* ps = reinterpret_cast<const uint16_t*>(&vs);
  const uint16_t* pa = reinterpret_cast<const uint16_t*>(&va);
  uint4 vo;
  uint16_t* po = reinterpret_cast<uint16_t*>(&vo);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    float2 c = fp8x2_to_float2(pc[k]);
    float2 s = fp8x2_to_float2(ps[k]);
    float2 a = fp8x2_to_float2(pa[k]);
    float r0 = (c.x * sc.c3 + s.x * sc.skip) * tanh_approx(a.x * s_att) * sc.out_inv;
    float r1 = (c.y * sc.c3 + s.y * sc.skip) * tanh_approx(a.y * s_att) * sc.out_inv;
    po[k] = float2_to_fp8x2(r0, r1);
  }
  out[i] = vo;
}

constexpr int kThreads = 256;

}  // namespace

cudaError_t launch_fp8_span_tail(const uint8_t* c3, const __half* att, const uint8_t* skip, uint8_t* out,
                                 size_t n, const Fp8SpanTailScales& scales, bool tanh_fast,
                                 cudaStream_t stream) {
  if (n % 16 != 0) return cudaErrorInvalidValue;
  size_t n_vec = n / 16;
  unsigned grid = static_cast<unsigned>((n_vec + kThreads - 1) / kThreads);
  auto args = [&](auto kernel) {
    kernel<<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(c3), reinterpret_cast<const uint4*>(att),
        reinterpret_cast<const uint4*>(skip), reinterpret_cast<uint4*>(out), n_vec, scales);
  };
  if (tanh_fast)
    args(span_tail_kernel<true>);
  else
    args(span_tail_kernel<false>);
  return cudaGetLastError();
}

cudaError_t launch_fp8_span_tail_att8(const uint8_t* c3, const uint8_t* att, const uint8_t* skip,
                                      uint8_t* out, size_t n, const Fp8SpanTailScales& scales,
                                      float att_scale, cudaStream_t stream) {
  if (n % 16 != 0) return cudaErrorInvalidValue;
  size_t n_vec = n / 16;
  unsigned grid = static_cast<unsigned>((n_vec + kThreads - 1) / kThreads);
  span_tail_att8_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const uint4*>(c3), reinterpret_cast<const uint4*>(att),
      reinterpret_cast<const uint4*>(skip), reinterpret_cast<uint4*>(out), n_vec, scales, att_scale);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
