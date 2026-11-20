#pragma once

#include "gpu/cuda/backend/rife_native_plan.h"

namespace kirari::gpu::cuda::drba_tail_wino {
namespace rn = kirari::gpu::cuda::rife_native;

__device__ __forceinline__ std::uint32_t transform(std::uint32_t a, std::uint32_t b, bool add) {
  const auto x = *reinterpret_cast<const __half2*>(&a);
  const auto y = *reinterpret_cast<const __half2*>(&b);
  const __half2 result = add ? __hadd2(x, y) : __hsub2(x, y);
  return *reinterpret_cast<const std::uint32_t*>(&result);
}

template <int C, int W, int Slice>
struct Tile {
  static constexpr int kHeight = 2, kIn = C + 8, kOut = Slice + 8, kHaloW = W + 2;
  static constexpr int kHalo = (kHeight + 2) * kHaloW;
  static constexpr int kGroup = 32, kThreads = (Slice / kGroup) * 32;
  static constexpr int kBytes = kHalo * kIn * sizeof(__half);
  static_assert(kBytes <= 48 * 1024 && kHalo * kIn >= kHeight * W * kOut);
};

template <class T, int C, int W>
__device__ __forceinline__ void load(__half* s, rn::Params p, int x0, int y0, int tid) {
  const auto raw = rn::smem_u32(s);
  for (int i = tid; i < T::kHalo * (C / 8); i += T::kThreads) {
    const int pixel = i / (C / 8), c = (i % (C / 8)) * 8;
    const int x = x0 + pixel % T::kHaloW - 1, y = y0 + pixel / T::kHaloW - 1;
    const bool valid = x >= 0 && x < p.iw && y >= 0 && y < p.ih;
    const auto* from = valid ? p.in + (static_cast<long>(y) * p.iw + x) * p.in_ld + c : p.in;
    rn::cp_async16(raw + static_cast<std::uint32_t>((pixel * T::kIn + c) * 2), from, valid);
  }
  rn::cp_async_wait_all();
  __syncthreads();
  if (p.in_slope == 1.F) return;
  for (int i = tid; i < T::kHalo * (C / 8); i += T::kThreads) {
    const int pixel = i / (C / 8), c = (i % (C / 8)) * 8;
    auto* v = reinterpret_cast<__half2*>(s + pixel * T::kIn + c);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __half22float2(v[j]);
      v[j] = __floats2half2_rn(f.x < 0.F ? f.x * p.in_slope : f.x, f.y < 0.F ? f.y * p.in_slope : f.y);
    }
  }
  __syncthreads();
}

template <int C, int W, int Slice>
__global__ void __launch_bounds__((Tile<C, W, Slice>::kThreads), 2) conv(rn::Params p, const float* beta) {
  using T = Tile<C, W, Slice>;
  extern __shared__ __align__(16) unsigned char raw[];
  auto* s = reinterpret_cast<__half*>(raw);
  const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
  const int x0 = static_cast<int>(blockIdx.x) * W, y0 = static_cast<int>(blockIdx.y) * T::kHeight;
  const int channel0 = static_cast<int>(blockIdx.z) * Slice;
  load<T, C, W>(s, p, x0, y0, tid);
  constexpr int kCols = W / 16, kInner = C / 16, kTiles = T::kGroup / 8;
  const int local_y = (warp / (Slice / T::kGroup)) * 2, local_c = (warp % (Slice / T::kGroup)) * T::kGroup;
  const int n = channel0 + local_c;
  rn::Frag<rn::Acc::kF16> acc[kCols][4][kTiles];
#pragma unroll
  for (int f = 0; f < kCols; ++f)
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int k = 0; k < kTiles; ++k) acc[f][b][k].clear();
  const auto base =
      rn::smem_u32(s) +
      static_cast<std::uint32_t>(((local_y * T::kHaloW + (lane & 15)) * T::kIn + (lane >> 4) * 8) * 2);
#pragma unroll 1
  for (int dx = 0; dx < 3; ++dx) {
#pragma unroll 1
    for (int step = 0; step < kInner; ++step) {
      std::uint32_t a[kCols][4][4];
#pragma unroll
      for (int f = 0; f < kCols; ++f)
#pragma unroll
        for (int r = 0; r < 4; ++r)
          rn::ldmatrix4(a[f][r], base + static_cast<std::uint32_t>(
                                            ((r * T::kHaloW + dx + 16 * f) * T::kIn + step * 16) * 2));
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        constexpr int kStart[4]{0, 1, 2, 1}, kEnd[4]{2, 2, 1, 3};
        std::uint32_t values[kCols][4];
#pragma unroll
        for (int f = 0; f < kCols; ++f)
#pragma unroll
          for (int k = 0; k < 4; ++k) values[f][k] = transform(a[f][kStart[v]][k], a[f][kEnd[v]][k], v == 1);
        const auto* weights = p.w + ((((4 * dx + v) * kInner + step) * kInner) + n / 16) * 32 + lane;
#pragma unroll
        for (int k = 0; k < kTiles / 2; ++k) {
          const uint4 b = __ldg(weights + k * 32);
#pragma unroll
          for (int f = 0; f < kCols; ++f) {
            acc[f][v][2 * k].mma(values[f], b.x, b.y);
            acc[f][v][2 * k + 1].mma(values[f], b.z, b.w);
          }
        }
      }
    }
  }

  __syncthreads();
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int k = 0; k < kTiles; ++k) {
    const int ch = local_c + k * 8 + 2 * t;
    const float2 bias = __ldg(reinterpret_cast<const float2*>(p.bias + channel0 + ch));
#pragma unroll
    for (int f = 0; f < kCols; ++f)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int x = g + 8 * h + 16 * f, at = (local_y * W + x) * T::kOut + ch;
        const auto a = acc[f][0][k].pair(h), b = acc[f][1][k].pair(h);
        const auto c = acc[f][2][k].pair(h), d = acc[f][3][k].pair(h);
        *reinterpret_cast<__half2*>(s + at) =
            rn::to_half2(a.x + b.x + c.x + bias.x, a.y + b.y + c.y + bias.y);
        *reinterpret_cast<__half2*>(s + at + W * T::kOut) =
            rn::to_half2(b.x - c.x - d.x + bias.x, b.y - c.y - d.y + bias.y);
      }
  }
  __syncthreads();
  for (int i = tid; i < T::kHeight * W * (Slice / 8); i += T::kThreads) {
    const int px = i / (Slice / 8), c = (i % (Slice / 8)) * 8;
    const int x = x0 + px % W, y = y0 + px / W;
    if (x >= p.ow || y >= p.oh) continue;
    const long pixel = static_cast<long>(y) * p.ow + x;
    uint4 v = *reinterpret_cast<const uint4*>(s + px * T::kOut + c);
    if (beta != nullptr) {
      auto values = reinterpret_cast<__half2*>(&v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const auto b = __ldg(reinterpret_cast<const float2*>(beta + channel0 + c + 2 * j));
        const auto r = __half22float2(values[j]);
        values[j] = rn::to_half2(r.x * b.x, r.y * b.y);
      }
    }
    *reinterpret_cast<uint4*>(p.out + pixel * p.out_ld + channel0 + c) =
        rn::finish8(v, p, pixel, channel0 + c);
  }
}

template <int C, int W, int Slice>
void launch(dim3 grid, int smem, cudaStream_t stream, const rn::Params& p, const float* beta) {
  using T = Tile<C, W, Slice>;
  conv<C, W, Slice><<<grid, T::kThreads, smem, stream>>>(p, beta);
}

inline std::vector<std::uint32_t> pack(const rn::Dense& dense) {
  constexpr double matrix[4][3]{{1, 0, 0}, {.5, .5, .5}, {.5, -.5, .5}, {0, 0, 1}};
  rn::Dense transformed{.cout = dense.cout, .taps = 12, .cin = dense.cin, .w = {}, .bias = {}};
  transformed.w.resize(transformed.cout * transformed.taps * transformed.cin);
  for (int o = 0; o < dense.cout; ++o)
    for (int dx = 0; dx < 3; ++dx)
      for (int p = 0; p < 4; ++p)
        for (int c = 0; c < dense.cin; ++c) {
          double value = 0;
          for (int dy = 0; dy < 3; ++dy)
            value += matrix[p][dy] * __half2float(dense.w[(o * 9 + dy * 3 + dx) * dense.cin + c]);
          transformed.w[(o * 12 + 4 * dx + p) * dense.cin + c] = __float2half_rn(static_cast<float>(value));
        }
  return rn::packed_weights(transformed, dense.cout);
}
}  // namespace kirari::gpu::cuda::drba_tail_wino
