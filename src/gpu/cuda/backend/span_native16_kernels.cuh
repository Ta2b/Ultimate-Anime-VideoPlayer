#pragma once
// Noyaux du SPAN FP16 natif (lot C du chantier AnimeJaNai, docs/span-native16.md), écrits sans CUTLASS : une
// seule famille de convolutions, mma.sync m16n8k16 FP16 avec sommes en f16 ou en f32 (Acc).
//
// Un bloc de 4 warps calcule une tuile de kTH × kTW pixels de sortie et toutes les colonnes. La tuile
// d'entrée, halo compris, est copiée une fois en mémoire partagée par cp.async (zéros hors de l'image), au
// pas Cin + 8 demi-flottants : les huit lignes d'une matrice de ldmatrix tombent sur des banques distinctes.
// Les poids sont rangés une fois, à l'envoi, dans l'ordre des fragments B du mma (pack_weights) : un __ldg
// de 16 octets par lane donne deux tuiles n8, coalescé sur le warp. La sortie passe par la mémoire partagée
// pour s'écrire en lignes entières.
//
// Modes de l'épilogue :
//   kPlain : biais (+ Relu), sortie NHWC FP16 ;
//   kGate  : fin de bloc SPAN, (v + x) · tanh(W1·v + b1), la 1×1 calculée depuis les accumulateurs ;
//   kPixel : to_pixel, DepthToSpace(4) fondu, sortie NCHW FP16 et/ou emplacement de l'anneau.
// Entrées : kNhwc (une ou plusieurs sources NHWC sommées sur K : conv_cat lit les quatre tenseurs sans
// concaténation) ou kImage (conv_1 : l'image NCHW FP16 dégroupée au chargement, 12 canaux + 4 nuls).
// Le noyau (conv_kernel) et sa boucle à poids partagés sont dans span_native_conv.cuh.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <vector>

#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "gpu/cuda/backend/ring_surface.h"

namespace kirari::gpu::cuda::span_native {

inline constexpr int kWarps = 4;
inline constexpr int kThreads = kWarps * 32;
inline constexpr int kTW = 32;              // largeur de tuile : deux fragments m16 par ligne
inline constexpr int kRows = 2;             // lignes de sortie par warp
inline constexpr int kTH = kWarps * kRows;  // hauteur de tuile
inline constexpr int kMF = kRows * 2;       // fragments m16 par warp
inline constexpr int kPixel = 48;           // sorties de to_pixel
inline constexpr int kImageCh = 16;         // canaux de conv_1 (12 + 4 nuls)

enum class Mode : int { kPlain, kGate, kPixel };
enum class Src : int { kNhwc, kImage };

struct Params {
  const __half* in[4] = {};      // sources NHWC (kNhwc), ou in[0] l'image NCHW 3×2fh×2fw (kImage)
  __half* out = nullptr;         // NHWC Cout, ou (kPixel) NCHW 3×4fh×4fw, ou nul
  const uint4* w = nullptr;      // poids rangés (pack_weights)
  const float* bias = nullptr;   // Cout flottants
  const uint4* w1 = nullptr;     // kGate : la 1×1 rangée de même
  const float* b1 = nullptr;     // kGate : son biais
  const __half* skip = nullptr;  // kGate : x, NHWC Cin
  int fh = 0, fw = 0;            // taille de la convolution (demi-résolution)
  // Lignes [y0, y1) de la sortie que le lancement calcule (bandes, docs/span-native16.md) ; y1 = 0 : toutes.
  // La grille couvre ces lignes seulement ; les lectures prennent toute l'image (halo hors de la bande).
  int y0 = 0, y1 = 0;
  // Pas d'un pixel des sources et de out en demi-flottants ; 0 : Cin et Cout. Le FP8 lit et écrit les
  // tranches de sa concaténation (pas 4·C).
  int in_ld = 0, out_ld = 0;
  // Sortie E4M3 de plus (NHWC, pas Cout octets), à l'échelle inv_s8 : l'entrée du premier bloc FP8.
  std::uint8_t* out8 = nullptr;
  float inv_s8 = 1.F;
  // Sommes f16 seulement : le mot de débordement du FP8 (KP::overflow, mémoire de l'hôte projetée), mis à 1
  // quand une somme a débordé ; nul : rien n'est noté (la somme est bornée de même, sums).
  int* overflow = nullptr;
  RingSurface surface{};  // kPixel : emplacement de l'anneau, object 0 : aucun
  int surf_w = 0, surf_h = 0;
  // Tuiles qui bougent (kernels/spec/span_tiles.h) : un bit par tuile kTH × kTW de la grille du lancement,
  // ligne par ligne ; un bloc dont le bit est nul sort sans rien lire ni écrire. Nul : toutes les tuiles.
  // Seulement sur l'image entière (y0 = 0, y1 = 0) : la grille d'une bande n'est pas celle des tuiles.
  const std::uint32_t* mask = nullptr;
};

// La fin des lignes écrites : y1, ou toute l'image.
__host__ __device__ __forceinline__ int row_end(const Params& p) {
  return p.y1 > 0 && p.y1 < p.fh ? p.y1 : p.fh;
}

// La tuile d'entrée en mémoire partagée : Cin / 8 morceaux de 16 octets par pixel, au pas kPitch morceaux.
// Les huit lignes d'une matrice de ldmatrix (huit pixels qui se suivent) doivent tomber sur des banques
// distinctes : un pas impair le donne (un morceau de rembourrage par pixel, le rangement d'origine). Shared
// (poids partagés, conv_kernel) : à 64 canaux, la tuile n'a pas la place du rembourrage à côté des poids ;
// elle est rangée sans, le morceau c du pixel p à la place c ^ (p % 8).
template <int Cin, bool Shared>
struct Tile {
  static constexpr int kChunks = Cin / 8;
  static constexpr bool kSwizzle = Shared && kChunks % 8 == 0;
  static constexpr int kPitch = kSwizzle ? kChunks : kChunks + 1;
  // L'octet du morceau c du pixel p.
  __host__ __device__ static constexpr int at(int p, int c) {
    return (p * kPitch + (kSwizzle ? c ^ (p & 7) : c)) * 16;
  }
};

// Les poids d'un pas k (Cout sorties × 16 entrées) des poids partagés, deux pas en mémoire partagée.
constexpr int weight_stage_bytes(int cout) {
  return cout * 16 * 2;
}

// Octets de mémoire partagée d'un noyau : la tuile d'entrée (halo compris), plus deux pas de poids s'ils sont
// partagés, ou la sortie, la plus grande.
constexpr int smem_bytes(int cin, int cout, int taps, bool shared = false) {
  const int halo = taps == 9 ? 2 : 0;
  const int chunks = cin / 8, pitch = shared && chunks % 8 == 0 ? chunks : chunks + 1;  // Tile::kPitch
  const int in = (kTH + halo) * (kTW + halo) * pitch * 16 + (shared ? 2 * weight_stage_bytes(cout) : 0);
  const int out = kTH * kTW * (cout + 8) * 2 + (cin == cout && taps == 9 ? cout * cout * 2 : 0);  // + 1×1
  return in > out ? in : out;
}

__device__ __forceinline__ std::uint32_t smem_u32(const void* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(std::uint32_t dst, const void* src, bool ok) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ uint4 lds128(std::uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "r"(addr));
  return v;
}

__device__ __forceinline__ void ldmatrix4(std::uint32_t (&a)[4], std::uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}

// Les sommes d'une convolution :
//   kF16 : en f16, le débit plein des tensor cores des GeForce ;
//   kF32 : en f32, la moitié de ce débit : le f32 du cran Qualité y bute sur le calcul.
// Promues (chaque mma en f16 depuis zéro, ajoutée en f32), elles ont perdu au banc : 14,39 ms et 71,5 dB
// contre 14,74 ms et 77,9 dB en f32 (docs/span-native16.md, « Sommes promues »).
enum class Acc : int { kF16, kF32 };

__device__ __forceinline__ float2 unpack_half2(std::uint32_t bits) {
  __half2 v;
  *reinterpret_cast<std::uint32_t*>(&v) = bits;
  return __half22float2(v);
}

// Accumulateur d'une tuile m16n8 : 2 mots (4 demi-flottants) en f16, 4 flottants en f32.
template <Acc A>
struct Frag;
template <>
struct Frag<Acc::kF16> {
  std::uint32_t r[2];
  __device__ __forceinline__ void clear() { r[0] = r[1] = 0U; }
  __device__ __forceinline__ void mma(const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(r[0]), "+r"(r[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  // h = 0 : ligne g, colonnes 2t et 2t + 1 ; h = 1 : ligne g + 8.
  __device__ __forceinline__ float2 pair(int h) const { return unpack_half2(r[h]); }
};
template <>
struct Frag<Acc::kF32> {
  float r[4];
  __device__ __forceinline__ void clear() { r[0] = r[1] = r[2] = r[3] = 0.F; }
  __device__ __forceinline__ void mma(const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(r[0]), "+f"(r[1]), "+f"(r[2]), "+f"(r[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
  __device__ __forceinline__ float2 pair(int h) const { return make_float2(r[2 * h], r[2 * h + 1]); }
};

// Deux valeurs en FP16, bornées au plus grand fini : une somme bornée (sums) plus son biais, ou la porte, ne
// sort pas en infini.
__device__ __forceinline__ __half2 to_half2(float lo, float hi) {
  constexpr float kMax = 65504.F;
  return __floats2half2_rn(fmaxf(fminf(lo, kMax), -kMax), fmaxf(fminf(hi, kMax), -kMax));
}

// Deux sommes de l'accumulateur (ligne g ou g + 8, pair). En f16, une somme débordée, infinie ou NaN (un
// débordement des deux signes), est bornée comme acc_value du FP8 : ±65504, NaN à 0 ; hit le note.
template <Acc A>
__device__ __forceinline__ float2 sums(const Frag<A>& frag, int h, bool& hit) {
  float2 s = frag.pair(h);
  if constexpr (A == Acc::kF16) {
    if (!isfinite(s.x) || !isfinite(s.y)) {
      hit = true;
      s.x = isnan(s.x) ? 0.F : fmaxf(fminf(s.x, 65504.F), -65504.F);
      s.y = isnan(s.y) ? 0.F : fmaxf(fminf(s.y, 65504.F), -65504.F);
    }
  } else {
    static_cast<void>(hit);
  }
  return s;
}

__device__ __forceinline__ std::uint32_t pack_half2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<std::uint32_t*>(&h);
}

__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// La tuile NHWC de src en mémoire partagée, rangée selon Tile : (kTH + 2h) × (kTW + 2h) pixels dont
// l'origine est (ty0 - h, tx0 - h) ; zéros hors de l'image.
template <int Cin, int Halo, bool Shared = false>
__device__ __forceinline__ void load_nhwc(__half* s, const __half* src, int ld, int fh, int fw, int ty0,
                                          int tx0) {
  constexpr int kChunks = Cin / 8, kSW = kTW + 2 * Halo, kPix = (kTH + 2 * Halo) * kSW;
  const std::uint32_t base = smem_u32(s);
  for (int i = static_cast<int>(threadIdx.x); i < kPix * kChunks; i += kThreads) {
    const int p = i / kChunks, c = i - p * kChunks;
    const int y = ty0 - Halo + p / kSW, x = tx0 - Halo + p % kSW;
    const bool ok = y >= 0 && y < fh && x >= 0 && x < fw;
    const __half* from = ok ? src + (static_cast<long>(y) * fw + x) * ld + c * 8 : src;
    cp_async16(base + static_cast<std::uint32_t>(Tile<Cin, Shared>::at(p, c)), from, ok);
  }
}

// conv_1 : l'image NCHW 3 × 2fh × 2fw dégroupée au chargement, canal c·4 + a·2 + b = plan c, ligne 2y + a,
// colonne 2x + b (le dégroupage du SPAN), canaux 12 à 15 nuls. Halo de 1.
__device__ __forceinline__ void load_image(__half* s, const __half* img, int fh, int fw, int ty0, int tx0) {
  constexpr int kSW = kTW + 2, kPix = (kTH + 2) * kSW, kLds = kImageCh + 8;
  const long width = 2L * fw, plane = 2L * fh * width;
  for (int p = static_cast<int>(threadIdx.x); p < kPix; p += kThreads) {
    const int y = ty0 - 1 + p / kSW, x = tx0 - 1 + p % kSW;
    std::uint32_t v[8] = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
    if (y >= 0 && y < fh && x >= 0 && x < fw) {
#pragma unroll
      for (int c = 0; c < 3; ++c)
#pragma unroll
        for (int a = 0; a < 2; ++a)
          v[c * 2 + a] =
              __ldg(reinterpret_cast<const unsigned int*>(img + c * plane + (2L * y + a) * width + 2L * x));
    }
    uint4* to = reinterpret_cast<uint4*>(s + p * kLds);
    to[0] = make_uint4(v[0], v[1], v[2], v[3]);
    to[1] = make_uint4(v[4], v[5], v[6], v[7]);
  }
}

// Les sommes d'une source sur la tuile : K = taps × Cin, à partir du pas ks0 des poids rangés.
// acc[f][nt] : fragment f (ligne f / 2 du warp, colonnes 16·(f % 2) à + 15), tuile n8 nt.
template <int Cin, int Cout, int Taps, Acc A>
__device__ __forceinline__ void accumulate(Frag<A> (&acc)[kMF][Cout / 8], const __half* s, const uint4* w,
                                           int ks0, int warp, int lane) {
  constexpr int kHalo = Taps == 9 ? 1 : 0, kSW = kTW + 2 * kHalo, kLds = Cin + 8;
  constexpr int kKC = Cin / 16, kNP = Cout / 16, kKS = Taps * kKC;
  // ldmatrix.x4 : la lane l donne la ligne l % 16 du fragment, demi-K l / 16.
  const std::uint32_t base =
      smem_u32(s) +
      static_cast<std::uint32_t>(((warp * kRows * kSW + (lane & 15)) * kLds + (lane >> 4) * 8) * 2);
  const uint4* wl = w + static_cast<long>(ks0) * kNP * 32 + lane;
  uint4 bc[kNP];
#pragma unroll
  for (int np = 0; np < kNP; ++np) bc[np] = __ldg(wl + np * 32);
#pragma unroll 1
  for (int tap = 0; tap < Taps; ++tap) {
    const int dy = tap / 3, dx = tap - dy * 3;
#pragma unroll
    for (int kc = 0; kc < kKC; ++kc) {
      const int ks = tap * kKC + kc;
      const int next = ks + 1 < kKS ? ks + 1 : ks;
      uint4 bn[kNP];
#pragma unroll
      for (int np = 0; np < kNP; ++np) bn[np] = __ldg(wl + (static_cast<long>(next) * kNP + np) * 32);
      std::uint32_t a[kMF][4];
#pragma unroll
      for (int f = 0; f < kMF; ++f)
        ldmatrix4(a[f], base + static_cast<std::uint32_t>(
                                   (((f >> 1) + dy) * kSW + 16 * (f & 1) + dx) * kLds * 2 + kc * 32));
#pragma unroll
      for (int f = 0; f < kMF; ++f)
#pragma unroll
        for (int np = 0; np < kNP; ++np) {
          acc[f][2 * np].mma(a[f], bc[np].x, bc[np].y);
          acc[f][2 * np + 1].mma(a[f], bc[np].z, bc[np].w);
        }
#pragma unroll
      for (int np = 0; np < kNP; ++np) bc[np] = bn[np];
    }
  }
}

// La sortie NHWC (Cout canaux) de la tuile, depuis la mémoire partagée au pas Cout + 8, en lignes entières.
// Et, si p.out8, la même en E4M3 à l'échelle p.inv_s8.
template <int Cout>
__device__ __forceinline__ void store_nhwc(const __half* o, const Params& p, int ty0, int tx0) {
  constexpr int kChunks = Cout / 8, kLdo = Cout + 8;
  const int ld = p.out_ld != 0 ? p.out_ld : Cout, ye = row_end(p);
  for (int i = static_cast<int>(threadIdx.x); i < kTH * kTW * kChunks; i += kThreads) {
    const int q = i / kChunks, c = i - q * kChunks;
    const int y = ty0 + q / kTW, x = tx0 + q % kTW;
    if (y >= ye || x >= p.fw) continue;
    const long pixel = static_cast<long>(y) * p.fw + x;
    const uint4 v = *reinterpret_cast<const uint4*>(o + q * kLdo + c * 8);
    if (p.out != nullptr) *reinterpret_cast<uint4*>(p.out + pixel * ld + c * 8) = v;
    if (p.out8 != nullptr) {
      const std::uint32_t w[4] = {v.x, v.y, v.z, v.w};
      std::uint16_t q8[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 h;
        *reinterpret_cast<std::uint32_t*>(&h) = w[j];
        const float2 f = __half22float2(h);
        q8[j] =
            __nv_cvt_float2_to_fp8x2(make_float2(f.x * p.inv_s8, f.y * p.inv_s8), __NV_SATFINITE, __NV_E4M3);
      }
      *reinterpret_cast<uint2*>(p.out8 + pixel * Cout + c * 8) =
          make_uint2(q8[0] | (std::uint32_t{q8[1]} << 16U), q8[2] | (std::uint32_t{q8[3]} << 16U));
    }
  }
}

// to_pixel : la tuile de 48 canaux (pas 56) en DepthToSpace(4, CRD), canal 16c + 4bh + bw -> plan c, pixel
// (4y + bh, 4x + bw). Sortie NCHW FP16 par 8 pixels (16 octets), et l'anneau pixel par pixel.
__device__ __forceinline__ void store_pixel(const __half* o, const Params& p, int ty0, int tx0) {
  constexpr int kLdo = kPixel + 8, kRowsOut = 4 * kTH, kChunks = 4 * kTW / 8;
  const int ow = 4 * p.fw, oh = 4 * p.fh, ye = row_end(p);
  const long plane = static_cast<long>(oh) * ow;
  if (p.out != nullptr) {
    for (int i = static_cast<int>(threadIdx.x); i < 3 * kRowsOut * kChunks; i += kThreads) {
      const int c = i / (kRowsOut * kChunks), rest = i - c * kRowsOut * kChunks;
      const int ly = rest / kChunks, k = rest - ly * kChunks;
      const int y = ty0 + (ly >> 2), x = tx0 + 2 * k;
      if (y >= ye || x >= p.fw) continue;
      const int bh = ly & 3;
      const __half* row = o + ((ly >> 2) * kTW + 2 * k) * kLdo + 16 * c + 4 * bh;
      uint4 v;
      v.x = *reinterpret_cast<const std::uint32_t*>(row);
      v.z = *reinterpret_cast<const std::uint32_t*>(row + 2);
      v.y = *reinterpret_cast<const std::uint32_t*>(row + kLdo);
      v.w = *reinterpret_cast<const std::uint32_t*>(row + kLdo + 2);
      // Deux écritures de 8 octets : une largeur impaire ne garde pas l'alignement de 16 d'une ligne à
      // l'autre, et le second pixel peut sortir de l'image.
      __half* at = p.out + c * plane + static_cast<long>(4 * y + bh) * ow + 4 * x;
      *reinterpret_cast<uint2*>(at) = make_uint2(v.x, v.z);
      if (x + 1 < p.fw) *reinterpret_cast<uint2*>(at + 4) = make_uint2(v.y, v.w);
    }
  }
  if (p.surface.object == 0) return;
  for (int i = static_cast<int>(threadIdx.x); i < kRowsOut * 4 * kTW; i += kThreads) {
    const int ly = i / (4 * kTW), lx = i - ly * 4 * kTW;
    const int ox = 4 * tx0 + lx, oy = 4 * ty0 + ly;
    if (ox >= p.surf_w || oy >= p.surf_h || ty0 + (ly >> 2) >= ye || tx0 + (lx >> 2) >= p.fw) continue;
    const __half* px = o + ((ly >> 2) * kTW + (lx >> 2)) * kLdo + 4 * (ly & 3) + (lx & 3);
    store_ring(p.surface, ox, oy, px[0], px[16], px[32]);
  }
}

// Fin de bloc : v (biais compris) est dans acc ; o tient x de la tuile (pas Cout + 8). La 1×1 lit v en
// opérande A (deux tuiles n8 voisines de l'accumulateur font un fragment m16k16) ; o reçoit la sortie.
// w1s : la 1×1 rangée, en mémoire partagée (lue par lds128 : en __ldg, ptxas sortait ses 64 registres de la
// boucle des fragments, et la variante f32 débordait).
template <int C, Acc A>
__device__ __forceinline__ void gate(const Frag<A> (&acc)[kMF][C / 8], __half* o, std::uint32_t w1s,
                                     const Params& p, int warp, int lane, bool& hit) {
  constexpr int kKC = C / 16, kNP = C / 16, kLdo = C + 8;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int f = 0; f < kMF; ++f) {
    // v = somme + biais, recalculé à l'écriture plutôt que gardé : moins de registres (pas de débordement
    // en f32).
    const auto value = [&](int nt, int h) {
      const float2 b = __ldg(reinterpret_cast<const float2*>(p.bias + nt * 8 + 2 * t));
      const float2 s = sums(acc[f][nt], h, hit);
      return make_float2(s.x + b.x, s.y + b.y);
    };
    std::uint32_t a[kKC][4];
#pragma unroll
    for (int kc = 0; kc < kKC; ++kc) {
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float2 v = value(2 * kc + j, h);
          a[kc][2 * j + h] = pack_half2(v.x, v.y);
        }
    }
    const int q0 = (warp * kRows + (f >> 1)) * kTW + 16 * (f & 1) + g;
#pragma unroll
    for (int np = 0; np < kNP; ++np) {
      Frag<A> d[2];
      d[0].clear();
      d[1].clear();
#pragma unroll
      for (int kc = 0; kc < kKC; ++kc) {
        const uint4 b = lds128(w1s + static_cast<std::uint32_t>(((kc * kNP + np) * 32 + lane) * 16));
        d[0].mma(a[kc], b.x, b.y);
        d[1].mma(a[kc], b.z, b.w);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int nt = 2 * np + j, ch = nt * 8 + 2 * t;
        const float2 b1 = __ldg(reinterpret_cast<const float2*>(p.b1 + ch));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          __half2* at = reinterpret_cast<__half2*>(o + (q0 + 8 * h) * kLdo + ch);
          const float2 x = __half22float2(*at);
          const float2 dd = sums(d[j], h, hit), v = value(nt, h);
          *at = to_half2((v.x + x.x) * tanh_approx(dd.x + b1.x), (v.y + x.y) * tanh_approx(dd.y + b1.y));
        }
      }
    }
  }
}

// Biais (+ Relu) de acc vers o (pas Cout + 8).
template <int Cout, bool Relu, Acc A>
__device__ __forceinline__ void plain(const Frag<A> (&acc)[kMF][Cout / 8], __half* o, const float* bias,
                                      int warp, int lane, bool& hit) {
  constexpr int kNT = Cout / 8, kLdo = Cout + 8;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int f = 0; f < kMF; ++f) {
    const int q0 = (warp * kRows + (f >> 1)) * kTW + 16 * (f & 1) + g;
#pragma unroll
    for (int nt = 0; nt < kNT; ++nt) {
      const int ch = nt * 8 + 2 * t;
      const float2 b = __ldg(reinterpret_cast<const float2*>(bias + ch));
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 s = sums(acc[f][nt], h, hit);
        float x0 = s.x + b.x, x1 = s.y + b.y;
        if constexpr (Relu) {
          x0 = fmaxf(x0, 0.F);
          x1 = fmaxf(x1, 0.F);
        }
        *reinterpret_cast<__half2*>(o + (q0 + 8 * h) * kLdo + ch) = to_half2(x0, x1);
      }
    }
  }
}

// Poids FP16 [sortie][position][entrée] (le paquet) rangés dans l'ordre des fragments B de mma m16n8k16 :
// pas k (position, tranche de 16 entrées), paire de tuiles n8, lane (g = lane / 4, t = lane % 4) ; 4 mots
// par lane : b0 et b1 de la tuile 2·np, puis de la tuile 2·np + 1. b0 = entrées 2t et 2t + 1 de la sortie
// 8·tuile + g, b1 = entrées 2t + 8 et 2t + 9 ; la plus petite entrée dans la moitié basse du mot.
inline std::vector<std::uint32_t> pack(const std::uint16_t* w, int cout, int taps, int cin) {
  const int kcs = cin / 16, nps = cout / 16, kss = taps * kcs;
  std::vector<std::uint32_t> out(static_cast<std::size_t>(kss) * nps * 32 * 4);
  const auto at = [&](int n, int tap, int ci) -> std::uint32_t {
    return w[(static_cast<std::size_t>(n) * taps + tap) * cin + ci];
  };
  for (int ks = 0; ks < kss; ++ks) {
    const int tap = ks / kcs, kc = ks % kcs;
    for (int np = 0; np < nps; ++np)
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2, t = lane & 3, k0 = kc * 16 + 2 * t;
        const std::size_t base = ((static_cast<std::size_t>(ks) * nps + np) * 32 + lane) * 4;
        for (int j = 0; j < 2; ++j) {
          const int n = (2 * np + j) * 8 + g;
          out[base + 2 * j] = at(n, tap, k0) | (at(n, tap, k0 + 1) << 16U);
          out[base + 2 * j + 1] = at(n, tap, k0 + 8) | (at(n, tap, k0 + 9) << 16U);
        }
      }
  }
  return out;
}

}  // namespace kirari::gpu::cuda::span_native
