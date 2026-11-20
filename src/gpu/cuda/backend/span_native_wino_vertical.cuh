// Winograd F(2,3) sur les lignes des 3×3 du SPAN FP16 C48 et C64. Chaque paire requiert quatre produits de
// matrices par prise horizontale, au lieu de six. Les deux couches ReLU conservent leurs biais et c3
// applique la 1×1, le résidu et tanh dans la même tuile. L'ordre des sommes diffère du chemin direct.
#pragma once

#include "gpu/cuda/backend/span_native_conv.cuh"

namespace kirari::gpu::cuda::span_native::vertical {

inline constexpr int kWidth = 32, kHeight = 4;
inline constexpr int kRawWidth = kWidth + 2, kPoints = 4;
static_assert(kWidth == kTW && kTH % kHeight == 0);

template <int Channels>
inline constexpr int kStride = Channels + 8;

template <int Channels>
inline constexpr int kWorkers = Channels == 48 ? 192 : 128;

template <int Channels>
inline constexpr int kShared = (kHeight + 2) * kRawWidth * kStride<Channels> * sizeof(__half);

__device__ __forceinline__ std::uint32_t transform_word(std::uint32_t a, std::uint32_t b, bool add) {
  __half2 ah, bh;
  *reinterpret_cast<std::uint32_t*>(&ah) = a;
  *reinterpret_cast<std::uint32_t*>(&bh) = b;
  const __half2 result = add ? __hadd2(ah, bh) : __hsub2(ah, bh);
  return *reinterpret_cast<const std::uint32_t*>(&result);
}

__device__ __forceinline__ float fixed(float x) {
  return isfinite(x) ? x : isnan(x) ? 0.F : fmaxf(-65504.F, fminf(65504.F, x));
}

template <int C>
__device__ __forceinline__ void vertical_gate(__half* s, Params p, int x0, int y0, int warp, int lane) {
  constexpr int kKC = C / 16, kChunks = C / 8;
  constexpr auto em = Acc::kF16;
  // La moitié haute de l'ancien halo stocke les poids de la porte après les sorties 3×3.
  const auto weight_s = smem_u32(s + kWidth * kHeight * kStride<C>);
  const int g = lane >> 2, t = lane & 3;
  constexpr int kColumns = kWidth / 16;
  const int first = C == 48 ? warp : warp * kColumns;
  const int end = C == 48 ? kHeight * kColumns : first + kColumns;
  constexpr int kStep = C == 48 ? kWorkers<C> / 32 : 1;
  for (int fragment = first; fragment < end; fragment += kStep) {
    const int row = fragment / kColumns, column = fragment % kColumns * 16;
    const int old_y = y0 + row;
    const int q0 = row * kWidth + column + g;
    std::uint32_t a[kKC][4];
    float2 cached[kChunks][2];
#pragma unroll
    for (int kc = 0; kc < kKC; ++kc)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int ch = kc * 16 + j * 8 + 2 * t;
        const float2 bias = __ldg(reinterpret_cast<const float2*>(p.bias + ch));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const auto v =
              __half22float2(*reinterpret_cast<const __half2*>(s + (q0 + 8 * h) * kStride<C> + ch));
          const float2 vb = make_float2(v.x + bias.x, v.y + bias.y);
          cached[2 * kc + j][h] = vb;
          a[kc][2 * j + h] = pack_half2(vb.x, vb.y);
        }
      }
    __syncwarp();
#pragma unroll
    for (int np = 0; np < kKC; ++np) {
      Frag<em> d[2];
      d[0].clear();
      d[1].clear();
#pragma unroll
      for (int kc = 0; kc < kKC; ++kc) {
        const uint4 b = lds128(weight_s + static_cast<unsigned>(((kc * kKC + np) * 32 + lane) * 16));
        d[0].mma(a[kc], b.x, b.y);
        d[1].mma(a[kc], b.z, b.w);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int channel = np * 16 + j * 8 + 2 * t;
        const float2 b1 = __ldg(reinterpret_cast<const float2*>(p.b1 + channel));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int old_x = x0 + column + g + 8 * h;
          const float2 x = (old_x < p.fw && old_y < p.fh)
                               ? __half22float2(*reinterpret_cast<const __half2*>(
                                     p.skip + (static_cast<long>(old_y) * p.fw + old_x) * C + channel))
                               : make_float2(0.F, 0.F);
          const auto sums = d[j].pair(h), v = cached[2 * np + j][h];
          *reinterpret_cast<__half2*>(s + (q0 + 8 * h) * kStride<C> + channel) =
              to_half2((v.x + x.x) * tanh_approx(fixed(sums.x) + b1.x),
                       (v.y + x.y) * tanh_approx(fixed(sums.y) + b1.y));
        }
      }
    }
  }
}

template <int C>
__device__ __forceinline__ void load_halo(std::uint32_t shared, const Params& p, int x0, int y0, int tid) {
  for (int i = tid; i < (kHeight + 2) * kRawWidth * (C / 8); i += kWorkers<C>) {
    const int pixel = i / (C / 8), c = (i % (C / 8)) * 8;
    const int x = x0 + pixel % kRawWidth - 1, y = y0 + pixel / kRawWidth - 1;
    const bool valid = x >= 0 && x < p.fw && y >= 0 && y < p.fh;
    const auto* from = valid ? p.in[0] + (static_cast<long>(y) * p.fw + x) * C + c : p.in[0];
    cp_async16(shared + static_cast<std::uint32_t>((pixel * kStride<C> + c) * 2), from, valid);
  }
  cp_async_wait_all();
}

template <int C>
__device__ __forceinline__ void store_output(const __half* s, const Params& p, int x0, int y0, int tid) {
  const int end = row_end(p);
  for (int i = tid; i < kHeight * kWidth * (C / 8); i += kWorkers<C>) {
    const int px = i / (C / 8), c = (i % (C / 8)) * 8;
    const int x = x0 + px % kWidth, y = y0 + px / kWidth;
    if (x < p.fw && y >= p.y0 && y < end)
      *reinterpret_cast<uint4*>(p.out + (static_cast<long>(y) * p.fw + x) * C + c) =
          *reinterpret_cast<const uint4*>(s + px * kStride<C> + c);
  }
}

// Une transformée F(2,3) par paire de lignes. L'axe horizontal garde les trois prises directes.
// Deux warps en C64 et trois en C48 partagent les canaux de sortie d'une paire. V reste dans les registres.
// Le masque reste celui des tuiles 32 × 8 de SPAN, deux rangées de blocs ici.
template <int C, bool Gate>
__global__ void __launch_bounds__(kWorkers<C>, 3) convolution(Params p) {
  static_assert(C == 48 || C == 64);
  static_assert((kWidth * kHeight * kStride<C> + C * C) * sizeof(__half) <= kShared<C>);
  extern __shared__ __align__(16) unsigned char raw[];
  auto* s = reinterpret_cast<__half*>(raw);
  const auto shared = smem_u32(s);
  const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int x0 = static_cast<int>(blockIdx.x) * kWidth;
  // Aligner la tuile pour qu'elle ne franchisse jamais une frontière de masque ni une paire.
  const int y0 = (p.y0 & ~(kHeight - 1)) + static_cast<int>(blockIdx.y) * kHeight;
  if (p.mask != nullptr) {
    const unsigned row = static_cast<unsigned>(y0 / kTH);
    const unsigned tile = row * gridDim.x + blockIdx.x;
    if (((__ldg(p.mask + (tile >> 5U)) >> (tile & 31U)) & 1U) == 0U) return;
  }
  load_halo<C>(shared, p, x0, y0, tid);
  __syncthreads();
  constexpr int kGroups = C == 48 ? 3 : 2;
  constexpr int kInner = C / 16, kGroup = C / kGroups;
  const int local_y = (warp / kGroups) * 2, channel = (warp % kGroups) * kGroup;
  Frag<Acc::kF16> acc[2][kPoints][kGroup / 8];
#pragma unroll
  for (int f = 0; f < 2; ++f)
#pragma unroll
    for (int a = 0; a < kPoints; ++a)
#pragma unroll
      for (int n = 0; n < kGroup / 8; ++n) acc[f][a][n].clear();
  const unsigned base =
      shared +
      static_cast<unsigned>(((local_y * kRawWidth + (lane & 15)) * kStride<C> + (lane >> 4) * 8) * 2);
#pragma unroll 1
  for (int dx = 0; dx < 3; ++dx) {
#pragma unroll 1
    for (int step = 0; step < kInner; ++step) {
      std::uint32_t a[2][4][4];
#pragma unroll
      for (int f = 0; f < 2; ++f)
#pragma unroll
        for (int r = 0; r < 4; ++r)
          ldmatrix4(a[f][r], base + static_cast<std::uint32_t>(
                                        ((r * kRawWidth + dx + 16 * f) * kStride<C> + step * 16) * 2));
#pragma unroll
      for (int point = 0; point < kPoints; ++point) {
        constexpr int kStart[4]{0, 1, 2, 1}, kEnd[4]{2, 2, 1, 3};
        std::uint32_t v[2][4];
#pragma unroll
        for (int f = 0; f < 2; ++f)
#pragma unroll
          for (int n = 0; n < 4; ++n)
            v[f][n] = transform_word(a[f][kStart[point]][n], a[f][kEnd[point]][n], point == 1);
        const auto* weights =
            p.w + ((((dx * kPoints + point) * kInner + step) * kInner) + channel / 16) * 32 + lane;
#pragma unroll
        for (int tile = 0; tile < kGroup / 16; ++tile) {
          const uint4 w = __ldg(weights + tile * 32);
#pragma unroll
          for (int f = 0; f < 2; ++f) {
            acc[f][point][2 * tile].mma(v[f], w.x, w.y);
            acc[f][point][2 * tile + 1].mma(v[f], w.z, w.w);
          }
        }
      }
    }
  }
  __syncthreads();
  if constexpr (Gate) {
    const auto base_w = smem_u32(s + kWidth * kHeight * kStride<C>);
    for (int k = tid; k < C * C / 8; k += kWorkers<C>)
      cp_async16(base_w + static_cast<unsigned>(16 * k), p.w1 + k, true);
    cp_async_wait_all();
  }
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int n = 0; n < kGroup / 8; ++n) {
    const int c = channel + n * 8 + 2 * t;
    const float2 bias = __ldg(reinterpret_cast<const float2*>(p.bias + c));
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int x = g + 8 * h + 16 * f, at = (local_y * kWidth + x) * kStride<C> + c;
        const float2 a = acc[f][0][n].pair(h), b = acc[f][1][n].pair(h);
        const float2 d = acc[f][2][n].pair(h), e = acc[f][3][n].pair(h);
        const auto put = [&](int index, float lo, float hi) {
          const auto narrowed = to_half2(lo, hi);
          if constexpr (Gate)
            *reinterpret_cast<__half2*>(s + index) = narrowed;
          else {
            const auto v = __half22float2(narrowed);
            *reinterpret_cast<__half2*>(s + index) =
                to_half2(fmaxf(0.F, v.x + bias.x), fmaxf(0.F, v.y + bias.y));
          }
        };
        put(at, fixed(a.x) + fixed(b.x) + fixed(d.x), fixed(a.y) + fixed(b.y) + fixed(d.y));
        put(at + kWidth * kStride<C>, fixed(b.x) - fixed(d.x) - fixed(e.x),
            fixed(b.y) - fixed(d.y) - fixed(e.y));
      }
  }
  __syncthreads();
  if constexpr (Gate) {
    vertical_gate<C>(s, p, x0, y0, warp, lane);
    __syncthreads();
  }
  store_output<C>(s, p, x0, y0, tid);
}

template <int C>
inline std::vector<std::uint32_t> pack(const std::uint16_t* w) {
  // G transforme les trois lignes de coefficients en quatre rangées, par prise horizontale.
  constexpr double g[4][3]{{1, 0, 0}, {.5, .5, .5}, {.5, -.5, .5}, {0, 0, 1}};
  std::vector<std::uint16_t> data(C * 3 * kPoints * C);
  for (int o = 0; o < C; ++o)
    for (int dx = 0; dx < 3; ++dx)
      for (int p = 0; p < kPoints; ++p)
        for (int c = 0; c < C; ++c) {
          double sum = 0.;
          for (int ry = 0; ry < 3; ++ry)
            sum += g[p][ry] * __half2float(__ushort_as_half(w[(o * 9 + ry * 3 + dx) * C + c]));
          data[((o * 3 + dx) * kPoints + p) * C + c] = __half_as_ushort(__float2half_rn(float(sum)));
        }
  return span_native::pack(data.data(), C, 3 * kPoints, C);
}
}  // namespace kirari::gpu::cuda::span_native::vertical
