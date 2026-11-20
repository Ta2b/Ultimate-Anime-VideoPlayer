#pragma once

#include "gpu/cuda/backend/rife_native_kernels.cuh"
#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::gmfss_grid_head {
namespace rn = rife_native;
struct Images {
  const __half* a;
  const __half* middle;
  const __half* b;
};
using Launch = void (*)(dim3, int, cudaStream_t, const rn::Params&, Images);

__device__ __forceinline__ std::uint32_t activated_pair(__half lo, __half hi, float slope) {
  __half2 pair = __halves2half2(lo, hi);
  if (slope != 1.F) {
    const float2 values = __half22float2(pair);
    pair = __floats2half2_rn(values.x < 0.F ? values.x * slope : values.x,
                             values.y < 0.F ? values.y * slope : values.y);
  }
  return static_cast<std::uint32_t>(__half_as_ushort(__low2half(pair))) |
         (static_cast<std::uint32_t>(__half_as_ushort(__high2half(pair))) << 16U);
}

// Les neuf plans entrent dans la même tuile de 16 canaux que le chemin pack_head, PReLU inclus.
// Le poids rangé, l'ordre des MMA et l'épilogue sont ceux de la convolution ordinaire.
template <int CT, int Rows, rn::Acc A>
__global__ void __launch_bounds__(rn::kThreads, 2) conv_planes(rn::Params p, Images images) {
  constexpr int kInput = 16, kPad = kInput + 8;
  using G = rn::Geometry<3, 1, Rows>;
  extern __shared__ __align__(16) unsigned char bytes[];
  auto* shared = reinterpret_cast<__half*>(bytes);
  const int top = static_cast<int>(blockIdx.y) * G::kTH;
  const int left = static_cast<int>(blockIdx.x) * rn::kTW;
  const long plane = static_cast<long>(p.iw) * p.ih;
  const __half zero = __ushort_as_half(0);
  for (int i = static_cast<int>(threadIdx.x); i < G::kIR * G::kIC; i += rn::kThreads) {
    const int y = top + i / G::kIC - 1;
    const int x = left + i % G::kIC - 1;
    auto* out = reinterpret_cast<uint4*>(shared + i * kPad);
    if (y < 0 || y >= p.ih || x < 0 || x >= p.iw) {
      out[0] = out[1] = make_uint4(0, 0, 0, 0);
      continue;
    }
    const long at = static_cast<long>(y) * p.iw + x;
    const auto* a = images.a + at;
    const auto* mid = images.middle + at;
    const auto* b = images.b + at;
    const float slope = p.in_slope;
    out[0] =
        make_uint4(activated_pair(a[0], a[plane], slope), activated_pair(a[2 * plane], mid[0], slope),
                   activated_pair(mid[plane], mid[2 * plane], slope), activated_pair(b[0], b[plane], slope));
    out[1] = make_uint4(activated_pair(b[2 * plane], zero, slope), 0, 0, 0);
  }
  __syncthreads();
  const int z = static_cast<int>(blockIdx.z);
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  constexpr int weight_stride = 9 * (CT / 16) * 32;
  rn::Frag<A> sums[Rows * 2][CT / 8];
#pragma unroll
  for (int f = 0; f < Rows * 2; ++f)
#pragma unroll
    for (int n = 0; n < CT / 8; ++n) sums[f][n].clear();
  rn::accumulate<kInput, CT, 3, 1, Rows, A>(sums, shared, p.w + z * weight_stride, 0, 1, 0, 9, warp, lane);
  __syncthreads();
  rn::stage<CT, Rows, A>(sums, shared, p.bias + z * CT, warp, lane);
  __syncthreads();
  rn::store<CT, Rows, rn::Out::kNhwc>(shared, p, top, left, 0, z * CT);
}

template <int CT, int Rows, rn::Acc A>
void launch(dim3 grid, int bytes, cudaStream_t stream, const rn::Params& p, Images images) {
  conv_planes<CT, Rows, A><<<grid, rn::kThreads, bytes, stream>>>(p, images);
}

inline Launch pick(const rn::Config& config, RifeNativeSums sums) {
  if (config.kc != 16 || config.ksplit != 1 || config.ahead || config.bands != 0) return nullptr;
  if (sums == RifeNativeSums::kF16) {
    if (config.ct == 32 && config.rows == 1) return launch<32, 1, rn::Acc::kF16>;
    if (config.ct == 64 && config.rows == 1) return launch<64, 1, rn::Acc::kF16>;
    if (config.ct == 64 && config.rows == 2) return launch<64, 2, rn::Acc::kF16>;
  }
  if (sums == RifeNativeSums::kF32 && config.ct == 32 && config.rows == 2)
    return launch<32, 2, rn::Acc::kF32>;
  return nullptr;
}

}  // namespace kirari::gpu::cuda::gmfss_grid_head
