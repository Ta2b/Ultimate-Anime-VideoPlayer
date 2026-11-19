#pragma once
// La convolution des noyaux natifs du SPAN (docs/span-native16.md) : conv_kernel, sur les aides de
// span_native16_kernels.cuh, et sa boucle à poids partagés (accumulate_shared, « Poids partagés »).
#include "gpu/cuda/backend/span_native16_kernels.cuh"

namespace kirari::gpu::cuda::span_native {

// accumulate, poids partagés : les poids d'un pas k (weight_stage_bytes) arrivent une fois par bloc dans la
// mémoire partagée (wbuf : deux pas, celui qu'on lit et celui qui arrive) au lieu d'une fois par warp depuis
// le L2, où quatre warps relisaient 74 Ko par tuile à 64 canaux ; une barrière par pas. La tuile est rangée
// selon Tile<Cin, true>. Mêmes sommes dans le même ordre : sortie identique au bit près.
template <int Cin, int Cout, int Taps, Acc A>
__device__ __forceinline__ void accumulate_shared(Frag<A> (&acc)[kMF][Cout / 8], const __half* s,
                                                  const uint4* w, std::uint32_t wbuf, int ks0, int warp,
                                                  int lane) {
  using T = Tile<Cin, true>;
  constexpr int kHalo = Taps == 9 ? 1 : 0, kSW = kTW + 2 * kHalo;
  constexpr int kKC = Cin / 16, kNP = Cout / 16, kKS = Taps * kKC, kStage = kNP * 32;  // uint4 par pas
  static_assert(kStage <= kThreads, "un uint4 des poids d'un pas par fil au plus");
  const std::uint32_t tile = smem_u32(s);
  // Le pas ks des poids vers sa place (ks % 2) : un uint4 par fil.
  const bool stager = static_cast<int>(threadIdx.x) < kStage;
  const std::uint32_t to = wbuf + static_cast<std::uint32_t>(threadIdx.x) * 16;
  const uint4* from = w + static_cast<long>(ks0) * kStage + threadIdx.x;
  if (stager) cp_async16(to, from, true);
  // ldmatrix.x4 : la lane l donne la ligne l % 16 du fragment, demi-K l / 16.
  const int row = warp * kRows * kSW + (lane & 15), half = lane >> 4;
#pragma unroll 1
  for (int tap = 0; tap < Taps; ++tap) {
    const int dy = tap / 3, dx = tap - dy * 3;
    int pixel[kMF];
#pragma unroll
    for (int f = 0; f < kMF; ++f) pixel[f] = row + ((f >> 1) + dy) * kSW + 16 * (f & 1) + dx;
#pragma unroll
    for (int kc = 0; kc < kKC; ++kc) {
      const int ks = tap * kKC + kc;
      cp_async_wait_all();
      __syncthreads();  // le pas ks est là ; plus personne ne lit le pas ks - 1, sa place est libre
      if (stager && ks + 1 < kKS)
        cp_async16(to + static_cast<std::uint32_t>(((ks + 1) & 1) * kStage * 16), from + (ks + 1) * kStage,
                   true);
      uint4 b[kNP];
#pragma unroll
      for (int np = 0; np < kNP; ++np)
        b[np] = lds128(wbuf + static_cast<std::uint32_t>(((ks & 1) * kStage + np * 32 + lane) * 16));
      std::uint32_t a[kMF][4];
#pragma unroll
      for (int f = 0; f < kMF; ++f)
        ldmatrix4(a[f], tile + static_cast<std::uint32_t>(T::at(pixel[f], 2 * kc + half)));
#pragma unroll
      for (int f = 0; f < kMF; ++f)
#pragma unroll
        for (int np = 0; np < kNP; ++np) {
          acc[f][2 * np].mma(a[f], b[np].x, b[np].y);
          acc[f][2 * np + 1].mma(a[f], b[np].z, b[np].w);
        }
    }
  }
}

// Une convolution entière. Sources : 1 (kImage, ou une NHWC), ou 4 (conv_cat, K = 4 · Cin). Shared : les
// poids partagés (accumulate_shared), leurs deux pas juste après la tuile.
template <int Cin, int Cout, int Taps, int Sources, Src From, Mode M, bool Relu, Acc A, bool Shared = false>
__global__ void __launch_bounds__(kThreads) conv_kernel(Params p) {
  static_assert(Cin % 16 == 0 && Cout % 16 == 0, "canaux par 16");
  static_assert(M != Mode::kGate || Cin == Cout, "la porte suit une conv carrée");
  constexpr int kHalo = Taps == 9 ? 1 : 0;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int ty0 = p.y0 + static_cast<int>(blockIdx.y) * kTH, tx0 = static_cast<int>(blockIdx.x) * kTW;
  if (p.mask != nullptr) {  // tout le bloc sort ensemble, avant la première barrière
    const unsigned tile = blockIdx.y * gridDim.x + blockIdx.x;
    if (((__ldg(p.mask + (tile >> 5U)) >> (tile & 31U)) & 1U) == 0U) return;
  }

  Frag<A> acc[kMF][Cout / 8];
#pragma unroll
  for (int f = 0; f < kMF; ++f)
#pragma unroll
    for (int nt = 0; nt < Cout / 8; ++nt) acc[f][nt].clear();

#pragma unroll 1
  for (int src = 0; src < Sources; ++src) {
    if (src > 0) __syncthreads();  // la source d'avant n'est plus lue
    if constexpr (From == Src::kImage) {
      load_image(s, p.in[0], p.fh, p.fw, ty0, tx0);
    } else {
      // Pas p.in[src] : un indice inconnu à la compilation copie le tableau des paramètres sur la pile.
      const __half* from = src == 0 ? p.in[0] : src == 1 ? p.in[1] : src == 2 ? p.in[2] : p.in[3];
      load_nhwc<Cin, kHalo, Shared>(s, from, p.in_ld != 0 ? p.in_ld : Cin, p.fh, p.fw, ty0, tx0);
      cp_async_wait_all();
    }
    __syncthreads();
    if constexpr (Shared) {
      constexpr int kTile = (kTH + 2 * kHalo) * (kTW + 2 * kHalo) * Tile<Cin, true>::kPitch * 16;
      accumulate_shared<Cin, Cout, Taps, A>(acc, s, p.w, smem_u32(s) + kTile, src * Taps * (Cin / 16), warp,
                                            lane);
    } else {
      accumulate<Cin, Cout, Taps, A>(acc, s, p.w, src * Taps * (Cin / 16), warp, lane);
    }
  }
  __syncthreads();   // la tuile d'entrée n'est plus lue : la mémoire partagée reçoit la sortie
  bool hit = false;  // une somme f16 débordée (sums)

  if constexpr (M == Mode::kGate) {
    // x de la tuile au pas Cout + 8, puis la 1×1 rangée (Cout² demi-flottants) juste après.
    const std::uint32_t w1s = smem_u32(s) + static_cast<std::uint32_t>(kTH * kTW * (Cout + 8) * 2);
    load_nhwc<Cout, 0>(s, p.skip, Cout, p.fh, p.fw, ty0, tx0);
    for (int i = static_cast<int>(threadIdx.x); i < Cout * Cout / 8; i += kThreads)
      cp_async16(w1s + static_cast<std::uint32_t>(i * 16), p.w1 + i, true);
    cp_async_wait_all();
    __syncthreads();
    gate<Cout, A>(acc, s, w1s, p, warp, lane, hit);
  } else {
    plain<Cout, Relu, A>(acc, s, p.bias, warp, lane, hit);
  }
  // Rare : plusieurs fils écrivent la même valeur, sans atomique (note_overflow du FP8).
  if constexpr (A == Acc::kF16)
    if (hit && p.overflow != nullptr) *reinterpret_cast<volatile int*>(p.overflow) = 1;
  __syncthreads();
  if constexpr (M == Mode::kPixel)
    store_pixel(s, p, ty0, tx0);
  else
    store_nhwc<Cout>(s, p, ty0, tx0);
}

}  // namespace kirari::gpu::cuda::span_native
