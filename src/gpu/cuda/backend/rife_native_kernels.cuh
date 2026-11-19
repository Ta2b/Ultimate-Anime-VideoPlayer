#pragma once
// Noyaux du RIFE natif (--rife-native, docs/rife-natif.md) : une famille de convolutions mma.sync FP16, de la
// même facture que celle du SPAN natif (span_native16_kernels.cuh) mais aux formes de RIFE : canaux d'entrée
// pris par tranches de Kc (la tuile d'entrée et son halo tiennent sous 48 Ko), canaux de sortie par tuiles de
// CT (blockIdx.z), pas 1 ou 2, et la transposée 4 × 4 de pas 2 vue comme quatre convolutions 2 × 2, une par
// phase de sortie (blockIdx.z). Sommes en f16 (Acc16, le débit plein des tensor cores des GeForce) ou en f32.
// sm_80 et après : mma m16n8k16 ; sm_75 (Turing) : deux m16n8k8, mêmes fragments.
//
// Un bloc de 4 warps calcule Rows lignes par warp sur 32 colonnes. La fin passe par la mémoire partagée :
// biais ajouté, arrondi FP16, puis à l'écriture en lignes entières l'addition résiduelle et l'activation
// (pente négative par canal : LeakyRelu, PRelu, Relu).
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace kirari::gpu::cuda::rife_native {

inline constexpr int kWarps = 4;
inline constexpr int kThreads = kWarps * 32;
inline constexpr int kTW = 32;

// Entrée : NHWC FP16 au pas in_ld, ou (kImage) trois plans FP16 d'une image, complétés à 16 canaux nuls.
enum class In : int { kNhwc, kImage };
// Sortie : NHWC au pas out_ld (kNhwc) ; la phase (py, px) d'une transposée, NHWC deux fois plus grande
// (kPhase) ; ou la transposée de l'encodeur en 16 canaux phase · 4 + c, rangés en quatre plans deux fois plus
// grands (kPlanes4).
enum class Out : int { kNhwc, kPhase, kPlanes4 };

struct Params {
  const __half* in = nullptr;  // NHWC, ou plans (kImage)
  int in_ld = 0;               // canaux par pixel de in
  long in_plane = 0;           // kImage : éléments d'un plan
  int ih = 0, iw = 0;          // taille de l'entrée
  __half* out = nullptr;
  int out_ld = 0;                // canaux par pixel (kNhwc, kPhase)
  int out_ch = 0;                // canaux écrits : les canaux de rembourrage au-delà ne le sont pas
  long out_plane = 0;            // kPlanes4 : éléments d'un plan de sortie
  int oh = 0, ow = 0;            // grille de la convolution (avant le doublement d'une transposée)
  const uint4* w = nullptr;      // poids rangés (pack), un paquet par blockIdx.z
  const float* bias = nullptr;   // tous les canaux de la grille z (tuile · CT + n)
  const float* slope = nullptr;  // nul : pas d'activation
  const __half* skip = nullptr;  // résiduel : NHWC au pas skip_ld, même grille que la sortie
  int skip_ld = 0;
  int cin = 0;    // multiple de Kc
  int tiles = 1;  // tuiles de sortie par phase
};

// Hauteur et largeur de la tuile d'entrée (halo compris), et octets de mémoire partagée.
template <int KH, int S, int Rows>
struct Geometry {
  static constexpr int kTH = kWarps * Rows;
  static constexpr int kIR = (kTH - 1) * S + KH;
  static constexpr int kIC = (kTW - 1) * S + KH;
};

constexpr int smem_bytes(int kc, int ct, int kh, int s, int rows) {
  const int th = kWarps * rows;
  const int in = ((th - 1) * s + kh) * ((kTW - 1) * s + kh) * (kc + 8) * 2;
  const int out = th * kTW * (ct + 8) * 2;
  return in > out ? in : out;
}

__device__ __forceinline__ std::uint32_t smem_u32(const void* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

// Copie de 16 octets vers la mémoire partagée (zéros si !ok) : cp.async dès sm_80, chargement puis écriture
// sur sm_75.
__device__ __forceinline__ void cp_async16(std::uint32_t dst, const void* src, bool ok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0));
#else
  const uint4 v = ok ? __ldg(static_cast<const uint4*>(src)) : make_uint4(0U, 0U, 0U, 0U);
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
#endif
}
__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void ldmatrix4(std::uint32_t (&a)[4], std::uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}

// Accumulateur d'une tuile m16n8 : 2 mots (4 demi-flottants) en f16, 4 flottants en f32. mma de m16n8k16 ;
// sur sm_75, deux m16n8k8 (a0 a1 / b0 : k 0-7 ; a2 a3 / b1 : k 8-15).
template <bool Acc16>
struct Frag;
template <>
struct Frag<true> {
  std::uint32_t r[2];
  __device__ __forceinline__ void clear() { r[0] = r[1] = 0U; }
  __device__ __forceinline__ void mma(const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(r[0]), "+r"(r[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(r[0]), "+r"(r[1])
        : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(r[0]), "+r"(r[1])
        : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
  }
  // h = 0 : ligne g, colonnes 2t et 2t + 1 ; h = 1 : ligne g + 8.
  __device__ __forceinline__ float2 pair(int h) const {
    __half2 v;
    *reinterpret_cast<std::uint32_t*>(&v) = r[h];
    return __half22float2(v);
  }
};
template <>
struct Frag<false> {
  float r[4];
  __device__ __forceinline__ void clear() { r[0] = r[1] = r[2] = r[3] = 0.F; }
  __device__ __forceinline__ void mma(const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(r[0]), "+f"(r[1]), "+f"(r[2]), "+f"(r[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(r[0]), "+f"(r[1]), "+f"(r[2]), "+f"(r[3])
        : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(r[0]), "+f"(r[1]), "+f"(r[2]), "+f"(r[3])
        : "r"(a[2]), "r"(a[3]), "r"(b1));
#endif
  }
  __device__ __forceinline__ float2 pair(int h) const { return make_float2(r[2 * h], r[2 * h + 1]); }
};

// Deux valeurs en FP16, bornées au plus grand fini : une somme f16 débordée ne sort pas en infini.
__device__ __forceinline__ __half2 to_half2(float lo, float hi) {
  constexpr float kMax = 65504.F;
  return __floats2half2_rn(fmaxf(fminf(lo, kMax), -kMax), fmaxf(fminf(hi, kMax), -kMax));
}

// La tranche [c0, c0 + Kc) de la tuile d'entrée en mémoire partagée, au pas Kc + 8 : kIR × kIC pixels dont
// l'origine est (y0, x0) ; zéros hors de l'image.
template <int Kc, int IR, int IC>
__device__ __forceinline__ void load_nhwc(__half* s, const Params& p, int c0, int y0, int x0) {
  constexpr int kChunks = Kc / 8, kLds = Kc + 8;
  const std::uint32_t base = smem_u32(s);
  for (int i = static_cast<int>(threadIdx.x); i < IR * IC * kChunks; i += kThreads) {
    const int q = i / kChunks, c = i - q * kChunks;
    const int y = y0 + q / IC, x = x0 + q % IC;
    const bool ok = y >= 0 && y < p.ih && x >= 0 && x < p.iw;
    const __half* from = ok ? p.in + (static_cast<long>(y) * p.iw + x) * p.in_ld + c0 + c * 8 : p.in;
    cp_async16(base + static_cast<std::uint32_t>((q * kLds + c * 8) * 2), from, ok);
  }
}

// Les trois plans d'une image en 16 canaux (3 lus, 13 nuls), au pas 24.
template <int IR, int IC>
__device__ __forceinline__ void load_image(__half* s, const Params& p, int y0, int x0) {
  constexpr int kLds = 16 + 8;
  const __half zero = __float2half(0.F);
  for (int q = static_cast<int>(threadIdx.x); q < IR * IC; q += kThreads) {
    const int y = y0 + q / IC, x = x0 + q % IC;
    __half v[3] = {zero, zero, zero};
    if (y >= 0 && y < p.ih && x >= 0 && x < p.iw) {
      const long at = static_cast<long>(y) * p.iw + x;
#pragma unroll
      for (int c = 0; c < 3; ++c) v[c] = p.in[c * p.in_plane + at];
    }
    __half2 lo = __halves2half2(v[0], v[1]);
    __half2 hi = __halves2half2(v[2], zero);
    uint4* to = reinterpret_cast<uint4*>(s + q * kLds);
    to[0] =
        make_uint4(*reinterpret_cast<std::uint32_t*>(&lo), *reinterpret_cast<std::uint32_t*>(&hi), 0U, 0U);
    to[1] = make_uint4(0U, 0U, 0U, 0U);
  }
}

// Les sommes d'une tranche : taps × (Kc / 16) pas k, à partir de la tranche q des poids rangés.
// acc[f][nt] : fragment f (ligne f / 2 du warp, colonnes 16·(f % 2) à + 15), tuile n8 nt.
template <int Kc, int CT, int KH, int S, int Rows, bool Acc16>
__device__ __forceinline__ void accumulate(Frag<Acc16> (&acc)[Rows * 2][CT / 8], const __half* s,
                                           const uint4* w, int q, int kcs, int warp, int lane) {
  using G = Geometry<KH, S, Rows>;
  constexpr int kLds = Kc + 8, kSub = Kc / 16, kNP = CT / 16, kMF = Rows * 2, kTaps = KH * KH;
  // ldmatrix.x4 : la lane l donne la ligne l % 16 du fragment (la colonne de sortie), demi-K l / 16.
  const std::uint32_t base =
      smem_u32(s) + static_cast<std::uint32_t>(((lane & 15) * S * kLds + (lane >> 4) * 8) * 2);
  const uint4* wl = w + lane;
#pragma unroll 1
  for (int tap = 0; tap < kTaps; ++tap) {
    const int dy = tap / KH, dx = tap - dy * KH;
#pragma unroll
    for (int sub = 0; sub < kSub; ++sub) {
      const long ks = static_cast<long>(tap) * kcs + q * kSub + sub;
      uint4 b[kNP];
#pragma unroll
      for (int np = 0; np < kNP; ++np) b[np] = __ldg(wl + (ks * kNP + np) * 32);
      std::uint32_t a[kMF][4];
#pragma unroll
      for (int f = 0; f < kMF; ++f) {
        const int row = (warp * Rows + (f >> 1)) * S + dy;
        const int col = 16 * (f & 1) * S + dx;
        ldmatrix4(a[f], base + static_cast<std::uint32_t>(((row * G::kIC + col) * kLds + sub * 16) * 2));
      }
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

// Biais de acc vers o (pas CT + 8), arrondi FP16.
template <int CT, int Rows, bool Acc16>
__device__ __forceinline__ void stage(const Frag<Acc16> (&acc)[Rows * 2][CT / 8], __half* o,
                                      const float* bias, int warp, int lane) {
  constexpr int kLdo = CT + 8;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int f = 0; f < Rows * 2; ++f) {
    const int q0 = (warp * Rows + (f >> 1)) * kTW + 16 * (f & 1) + g;
#pragma unroll
    for (int nt = 0; nt < CT / 8; ++nt) {
      const int ch = nt * 8 + 2 * t;
      const float2 b = __ldg(reinterpret_cast<const float2*>(bias + ch));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 v = acc[f][nt].pair(h);
        *reinterpret_cast<__half2*>(o + (q0 + 8 * h) * kLdo + ch) = to_half2(v.x + b.x, v.y + b.y);
      }
    }
  }
}

// Huit canaux d'un pixel : + résiduel, puis activation ; n : premier canal dans la grille de la couche.
__device__ __forceinline__ uint4 finish8(uint4 v, const Params& p, long pixel, int n) {
  std::uint32_t w[4] = {v.x, v.y, v.z, v.w};
  if (p.skip == nullptr && p.slope == nullptr) return v;
  const uint4 sk = p.skip != nullptr ? __ldg(reinterpret_cast<const uint4*>(p.skip + pixel * p.skip_ld + n))
                                     : make_uint4(0U, 0U, 0U, 0U);
  const std::uint32_t s[4] = {sk.x, sk.y, sk.z, sk.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __half2 h, k;
    *reinterpret_cast<std::uint32_t*>(&h) = w[j];
    *reinterpret_cast<std::uint32_t*>(&k) = s[j];
    float2 f = __half22float2(h);
    const float2 r = __half22float2(k);
    f.x += r.x;
    f.y += r.y;
    if (p.slope != nullptr) {
      const float2 sl = __ldg(reinterpret_cast<const float2*>(p.slope + n + 2 * j));
      f.x = f.x < 0.F ? f.x * sl.x : f.x;
      f.y = f.y < 0.F ? f.y * sl.y : f.y;
    }
    h = to_half2(f.x, f.y);
    w[j] = *reinterpret_cast<std::uint32_t*>(&h);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

// La tuile de sortie, depuis la mémoire partagée (pas CT + 8), en lignes entières.
template <int CT, int Rows, Out O>
__device__ __forceinline__ void store(const __half* o, const Params& p, int ty0, int tx0, int phase, int n0) {
  constexpr int kTH = kWarps * Rows, kLdo = CT + 8;
  if constexpr (O == Out::kPlanes4) {
    // 16 canaux phase · 4 + c : pour chaque plan c et chaque ligne py, les deux colonnes px d'un mot.
    const int w2 = 2 * p.ow;
    for (int q = static_cast<int>(threadIdx.x); q < kTH * kTW; q += kThreads) {
      const int y = ty0 + q / kTW, x = tx0 + q % kTW;
      if (y >= p.oh || x >= p.ow) continue;
      const __half* v = o + q * kLdo;
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int py = 0; py < 2; ++py) {
          const __half2 pair = __halves2half2(v[(py * 2) * 4 + c], v[(py * 2 + 1) * 4 + c]);
          *reinterpret_cast<__half2*>(p.out + c * p.out_plane + static_cast<long>(2 * y + py) * w2 + 2 * x) =
              pair;
        }
    }
  } else {
    constexpr int kChunks = CT / 8;
    const int py = phase >> 1, px = phase & 1;
    for (int i = static_cast<int>(threadIdx.x); i < kTH * kTW * kChunks; i += kThreads) {
      const int q = i / kChunks, c = i - q * kChunks;
      const int y = ty0 + q / kTW, x = tx0 + q % kTW;
      const int n = n0 + c * 8;
      if (y >= p.oh || x >= p.ow || n >= p.out_ch) continue;
      uint4 v = *reinterpret_cast<const uint4*>(o + q * kLdo + c * 8);
      long pixel;
      if constexpr (O == Out::kPhase) {
        pixel = static_cast<long>(2 * y + py) * (2 * p.ow) + 2 * x + px;
      } else {
        pixel = static_cast<long>(y) * p.ow + x;
        v = finish8(v, p, pixel, n);
      }
      *reinterpret_cast<uint4*>(p.out + pixel * p.out_ld + n) = v;
    }
  }
}

// Une convolution. KH = 3 : 3 × 3 de bord 1, pas S. KH = 2 (S = 1) : la phase blockIdx.z / tiles d'une
// transposée 4 × 4 de pas 2, dont les deux prises par axe sont aux décalages py - 1 et py.
template <int Kc, int CT, int KH, int S, int Rows, In I, Out O, bool Acc16>
__global__ void __launch_bounds__(kThreads, 2) conv_kernel(Params p) {
  static_assert(Kc % 16 == 0 && CT % 16 == 0, "canaux par 16");
  static_assert(I != In::kImage || Kc == 16, "l'image tient en une tranche de 16 canaux");
  static_assert(O != Out::kPlanes4 || CT == 16, "la transposée de l'encodeur : 16 canaux");
  using G = Geometry<KH, S, Rows>;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  const int ty0 = static_cast<int>(blockIdx.y) * G::kTH, tx0 = static_cast<int>(blockIdx.x) * kTW;
  const int z = static_cast<int>(blockIdx.z);
  const int phase = z / p.tiles, tile = z - phase * p.tiles;
  const int origin_y = KH == 2 ? (phase >> 1) - 1 : -1;
  const int origin_x = KH == 2 ? (phase & 1) - 1 : -1;
  const int kcs = p.cin / 16;
  const uint4* w = p.w + static_cast<long>(z) * KH * KH * kcs * (CT / 16) * 32;

  Frag<Acc16> acc[Rows * 2][CT / 8];
#pragma unroll
  for (int f = 0; f < Rows * 2; ++f)
#pragma unroll
    for (int nt = 0; nt < CT / 8; ++nt) acc[f][nt].clear();

  const int chunks = p.cin / Kc;
#pragma unroll 1
  for (int q = 0; q < chunks; ++q) {
    if (q > 0) __syncthreads();  // la tranche d'avant n'est plus lue
    if constexpr (I == In::kImage) {
      load_image<G::kIR, G::kIC>(s, p, ty0 * S + origin_y, tx0 * S + origin_x);
    } else {
      load_nhwc<Kc, G::kIR, G::kIC>(s, p, q * Kc, ty0 * S + origin_y, tx0 * S + origin_x);
      cp_async_wait_all();
    }
    __syncthreads();
    accumulate<Kc, CT, KH, S, Rows, Acc16>(acc, s, w, q, kcs, warp, lane);
  }
  __syncthreads();  // la tuile d'entrée n'est plus lue : la mémoire partagée reçoit la sortie
  const int n0 = tile * CT;
  stage<CT, Rows, Acc16>(acc, s, p.bias + static_cast<long>(phase) * p.tiles * CT + n0, warp, lane);
  __syncthreads();
  store<CT, Rows, O>(s, p, ty0, tx0, phase, n0);
}

}  // namespace kirari::gpu::cuda::rife_native
