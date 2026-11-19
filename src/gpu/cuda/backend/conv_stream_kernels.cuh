#pragma once
// La forme en flux des convolutions natives (Config::bands, rife_native_plan.h) : 3 × 3 de pas 1 dont toute
// l'entrée tient en une tranche de Kc canaux, sortie NHWC (résiduel et activation). Un bloc descend une
// colonne de bandes, l'entrée dans un anneau de lignes (calendrier : conv_stream.h) ; la bande suivante se
// charge par cp.async pendant les mma de la courante, et les deux lignes du bord ne se relisent pas.
//
// Les sommes sont celles de conv_kernel<Kc, CT, 3, 1, Rows, …> (rife_native_kernels.cuh) à une seule tranche,
// dans le même ordre (prise, puis pas de 16 canaux, poids du pas suivant chargés pendant les mma du courant),
// avec le même PReLU à la lecture, le même biais arrondi en FP16 et la même fin (finish8) : la sortie est
// celle de cette forme, au bit près. Seules changent les adresses en mémoire partagée (les lignes de
// l'anneau) et le moment des chargements. Sur sm_75, cp_async16 lit puis écrit la mémoire partagée, et
// l'anneau n'y laisse qu'un bloc par multiprocesseur (64 Ko) : la forme y reste juste, sans recouvrement ni
// second bloc, et le banc la juge comme ailleurs.
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/conv_stream.h"
#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::conv_stream {

namespace rn = rife_native;

static_assert(kWarps == rn::kWarps && kTW == rn::kTW && kCols == rn::Geometry<3, 1, 1>::kIC,
              "le calendrier suit la géométrie des noyaux du RIFE natif");

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}

// L'adresse (en éléments) du pixel col de la ligne row de la fenêtre de la bande band, au pas ld.
template <int TH, int Ld>
__device__ __forceinline__ int ring_at(int band, int row, int col) {
  return (window_slot(band, row, TH) * kCols + col) * Ld;
}

// Les lignes [first, first + count) de la fenêtre de la bande band, depuis l'entrée NHWC (zéros hors de
// l'image) ; la ligne 0 de la bande 0 est la ligne d'entrée y_top - 1, x0 la colonne du pixel 0.
template <int Kc, int TH>
__device__ __forceinline__ void load_rows(__half* ring, const rn::Params& p, int band, int first, int count,
                                          int y_top, int x0) {
  constexpr int kChunks = Kc / 8, kLds = Kc + 8, kPer = kCols * kChunks;
  const std::uint32_t base = rn::smem_u32(ring);
  for (int i = static_cast<int>(threadIdx.x); i < count * kPer; i += rn::kThreads) {
    const int r = i / kPer, rest = i - r * kPer;
    const int col = rest / kChunks, c = rest - col * kChunks;
    const int y = y_top + band * TH - 1 + first + r, x = x0 + col;
    const bool ok = y >= 0 && y < p.ih && x >= 0 && x < p.iw;
    const __half* from = ok ? p.in + (static_cast<long>(y) * p.iw + x) * p.in_ld + c * 8 : p.in;
    rn::cp_async16(base + static_cast<std::uint32_t>((ring_at<TH, kLds>(band, first + r, col) + c * 8) * 2),
                   from, ok);
  }
}

// Le PReLU de l'entrée (prelu_tile) sur les lignes [first, first + count) de la fenêtre.
template <int Kc, int TH>
__device__ __forceinline__ void prelu_rows(__half* ring, int band, int first, int count, float a) {
  constexpr int kChunks = Kc / 8, kLds = Kc + 8, kPer = kCols * kChunks;
  for (int i = static_cast<int>(threadIdx.x); i < count * kPer; i += rn::kThreads) {
    const int r = i / kPer, rest = i - r * kPer;
    const int col = rest / kChunks, c = rest - col * kChunks;
    auto* v = reinterpret_cast<__half2*>(ring + ring_at<TH, kLds>(band, first + r, col) + c * 8);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __half22float2(v[j]);
      v[j] = __floats2half2_rn(f.x < 0.F ? f.x * a : f.x, f.y < 0.F ? f.y * a : f.y);
    }
  }
}

// Les sommes de la bande band (rn::accumulate sans Ahead, une tranche, prises 0 à 8) : les fragments A de la
// ligne (warp · Rows + f / 2) + dy de la fenêtre.
template <int Kc, int CT, int Rows, rn::Acc A>
__device__ __forceinline__ void accumulate(rn::Frag<A> (&acc)[Rows * 2][CT / 8], const __half* ring,
                                           const uint4* w, int band, int warp, int lane) {
  constexpr int kTH = kWarps * Rows, kLds = Kc + 8, kSub = Kc / 16, kNP = CT / 16, kMF = Rows * 2;
  const std::uint32_t base =
      rn::smem_u32(ring) + static_cast<std::uint32_t>(((lane & 15) * kLds + (lane >> 4) * 8) * 2);
  const uint4* wl = w + lane;
  const auto fetch = [&](uint4(&b)[kNP], int tap, int sub) {
    const long ks = static_cast<long>(tap) * kSub + sub;
#pragma unroll
    for (int np = 0; np < kNP; ++np) b[np] = __ldg(wl + (ks * kNP + np) * 32);
  };
  const auto step = [&](const uint4(&b)[kNP], const int (&at)[kMF], int sub) {
    std::uint32_t a[kMF][4];
#pragma unroll
    for (int f = 0; f < kMF; ++f)
      rn::ldmatrix4(a[f], base + static_cast<std::uint32_t>((at[f] + sub * 16) * 2));
#pragma unroll
    for (int f = 0; f < kMF; ++f)
#pragma unroll
      for (int np = 0; np < kNP; ++np) {
        acc[f][2 * np].mma(a[f], b[np].x, b[np].y);
        acc[f][2 * np + 1].mma(a[f], b[np].z, b[np].w);
      }
  };
  uint4 b[kNP];
  fetch(b, 0, 0);
#pragma unroll 1
  for (int tap = 0; tap < 9; ++tap) {
    const int dy = tap / 3, dx = tap - dy * 3;
    int at[kMF];
#pragma unroll
    for (int f = 0; f < kMF; ++f)
      at[f] = ring_at<kTH, kLds>(band, warp * Rows + (f >> 1) + dy, 16 * (f & 1) + dx);
#pragma unroll
    for (int sub = 0; sub < kSub; ++sub) {
      uint4 next[kNP];
#pragma unroll
      for (int np = 0; np < kNP; ++np) next[np] = b[np];
      if (sub + 1 < kSub)
        fetch(next, tap, sub + 1);
      else if (tap + 1 < 9)
        fetch(next, tap + 1, 0);
      step(b, at, sub);
#pragma unroll
      for (int np = 0; np < kNP; ++np) b[np] = next[np];
    }
#pragma unroll
    for (int f = 0; f < kMF; ++f)
#pragma unroll
      for (int nt = 0; nt < CT / 8; ++nt) acc[f][nt].promote();
  }
}

// rn::stage, la ligne r de la sortie rangée dans la ligne r de la fenêtre, au pas CT + 8.
template <int Kc, int CT, int Rows, rn::Acc A>
__device__ __forceinline__ void stage(const rn::Frag<A> (&acc)[Rows * 2][CT / 8], __half* ring, int band,
                                      const float* bias, int warp, int lane) {
  constexpr int kTH = kWarps * Rows, kLdo = CT + 8;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int f = 0; f < Rows * 2; ++f) {
    __half* row = ring + ring_at<kTH, Kc + 8>(band, warp * Rows + (f >> 1), 0);
    const int x0 = 16 * (f & 1) + g;
#pragma unroll
    for (int nt = 0; nt < CT / 8; ++nt) {
      const int ch = nt * 8 + 2 * t;
      const float2 b = __ldg(reinterpret_cast<const float2*>(bias + ch));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 v = acc[f][nt].pair(h);
        *reinterpret_cast<__half2*>(row + (x0 + 8 * h) * kLdo + ch) = rn::to_half2(v.x + b.x, v.y + b.y);
      }
    }
  }
}

// rn::store en NHWC (résiduel puis activation, finish8), depuis les lignes de la fenêtre.
template <int Kc, int CT, int Rows>
__device__ __forceinline__ void store(const __half* ring, int band, const rn::Params& p, int ty0, int tx0,
                                      int n0) {
  constexpr int kTH = kWarps * Rows, kLdo = CT + 8, kChunks = CT / 8;
  for (int i = static_cast<int>(threadIdx.x); i < kTH * kTW * kChunks; i += rn::kThreads) {
    const int q = i / kChunks, c = i - q * kChunks;
    const int r = q / kTW, col = q - r * kTW;
    const int y = ty0 + r, x = tx0 + col;
    const int n = n0 + c * 8;
    if (y >= p.oh || x >= p.ow || n >= p.out_ch) continue;
    const __half* row = ring + ring_at<kTH, Kc + 8>(band, r, 0);
    const long pixel = static_cast<long>(y) * p.ow + x;
    const uint4 v = rn::finish8(*reinterpret_cast<const uint4*>(row + col * kLdo + c * 8), p, pixel, n);
    *reinterpret_cast<uint4*>(p.out + pixel * p.out_ld + n) = v;
  }
}

// Une convolution 3 × 3 de pas 1, p.cin == Kc : blockIdx.x la colonne de 32 pixels, blockIdx.y la colonne de
// bandes (strip_of sur gridDim.y), blockIdx.z la tuile de CT sorties. Ni K découpé ni phases.
template <int Kc, int CT, int Rows, rn::Acc A>
__global__ void __launch_bounds__(rn::kThreads, 2) stream_kernel(rn::Params p) {
  static_assert(Kc % 16 == 0 && CT % 16 == 0, "canaux par 16");
  static_assert(stages_in_slot(Kc, CT), "la sortie d'une ligne tient dans un emplacement de l'anneau");
  constexpr int kTH = kWarps * Rows;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* ring = reinterpret_cast<__half*>(smem_raw);
  const int tiles = (p.oh + kTH - 1) / kTH;
  const Strip strip = strip_of(tiles, static_cast<int>(gridDim.y), static_cast<int>(blockIdx.y));
  if (strip.count <= 0 || p.cin != Kc) return;  // une colonne de trop ; une entrée d'une autre largeur
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int y_top = strip.first * kTH, tx0 = static_cast<int>(blockIdx.x) * kTW;
  const int n0 = static_cast<int>(blockIdx.z) * CT;
  const uint4* w = p.w + static_cast<long>(blockIdx.z) * 9 * (Kc / 16) * (CT / 16) * 32;

  load_rows<Kc, kTH>(ring, p, 0, 0, kTH + 2, y_top, tx0 - 1);
  cp_async_commit();
#pragma unroll 1
  for (int band = 0; band < strip.count; ++band) {
    // La fenêtre de la bande est arrivée ; la sortie de la bande d'avant est partie : ses lignes reçoivent la
    // bande suivante, en vol pendant les mma.
    cp_async_wait();
    __syncthreads();
    if (band + 1 < strip.count) load_rows<Kc, kTH>(ring, p, band + 1, 2, kTH, y_top, tx0 - 1);
    cp_async_commit();
    if (p.in_slope != 1.F) {
      prelu_rows<Kc, kTH>(ring, band, band == 0 ? 0 : 2, band == 0 ? kTH + 2 : kTH, p.in_slope);
      __syncthreads();
    }
    rn::Frag<A> acc[Rows * 2][CT / 8];
#pragma unroll
    for (int f = 0; f < Rows * 2; ++f)
#pragma unroll
      for (int nt = 0; nt < CT / 8; ++nt) acc[f][nt].clear();
    accumulate<Kc, CT, Rows, A>(acc, ring, w, band, warp, lane);
    __syncthreads();  // la fenêtre n'est plus lue : ses kTH premières lignes reçoivent la sortie
    stage<Kc, CT, Rows, A>(acc, ring, band, p.bias + n0, warp, lane);
    __syncthreads();
    store<Kc, CT, Rows>(ring, band, p, y_top + band * kTH, tx0, n0);
  }
}

}  // namespace kirari::gpu::cuda::conv_stream
