// Noyaux de RIFE fusionné (voir rife_fused_kernels.h), écrits sur la bibliothèque de warp (warp_lib.cuh).
//
// Disposition des 28 canaux d'entrée des blocs 1 et suivants (celle du graphe ONNX,
// kernels/spec/rife_fused.h) :
//   0-2 image A warpée, 3-5 image B warpée, 6-9 encodage de A warpé, 10-13 encodage de B warpé, 14 instant,
//   15 masque, 16-23 attributs, 24-27 flot accumulé × (1/d).
// Bloc 0 : 15 canaux (images A, B, encodages de A et de B, instant), tous moyennés sans warp.
//
// La colle (lot E) : chaque bloc de fils relève une fois en mémoire partagée la région des sorties Y des
// blocs précédents qu'il lit (16 octets à la fois), puis chaque fil agrandit les flots depuis là. Réduite (d
// ≥ 2), un fil par point d'échantillonnage, les quatre points d'une sortie dans quatre fils voisins du même
// warp, sommés par échange de registres : quatre fois plus de fils que de sorties, ce qui occupe la carte aux
// petites échelles. Spécialisée par forme : pleine résolution ou réduite, et nombre de flots (1 à 4).
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/rife_fused_kernels.h"
#include "gpu/cuda/backend/warp_lib.cuh"

namespace kirari::gpu::cuda {
namespace {

using warp::rounded;

// Les C valeurs d'un pixel de X, arrondies en FP16, en plans ou par pixel (pitch 0 ou P). Par pixel, les
// canaux C à P - 1 sont écrits à zéro (les poids que TensorRT y applique sont nuls, mais un NaN resté en
// mémoire passerait) et le pixel part en P / 8 écritures de 16 octets.
template <int C, int P>
__device__ __forceinline__ void store_x(__half* x, int pitch, std::size_t plane, std::size_t pixel,
                                        const float (&v)[C]) {
  if (pitch == 0) {
#pragma unroll
    for (int c = 0; c < C; ++c) x[static_cast<std::size_t>(c) * plane + pixel] = __float2half_rn(v[c]);
    return;
  }
  static_assert(P % 8 == 0 && P >= C, "P : multiple de 8, au moins C");
  std::uint32_t word[P / 2];
#pragma unroll
  for (int i = 0; i < P / 2; ++i) {
    const __half2 pair = __floats2half2_rn(2 * i < C ? v[2 * i] : 0.0F, 2 * i + 1 < C ? v[2 * i + 1] : 0.0F);
    word[i] = *reinterpret_cast<const std::uint32_t*>(&pair);
  }
  auto* out = reinterpret_cast<uint4*>(x + pixel * P);
#pragma unroll
  for (int j = 0; j < P / 8; ++j)
    out[j] = make_uint4(word[4 * j], word[4 * j + 1], word[4 * j + 2], word[4 * j + 3]);
}

// Les canaux 8 q à 8 q + 7 d'un pixel de X (28 canaux, pas 32), complétés de zéros : une écriture de 16
// octets. q est connu de chaque fil ; la boucle déroulée garde v en registres.
__device__ __forceinline__ void store_chunk(__half* x, std::size_t pixel, int q, const float (&v)[28]) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (j != q) continue;
    std::uint32_t word[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int c = 8 * j + 2 * i;
      const __half2 pair = __floats2half2_rn(c < 28 ? v[c] : 0.0F, c + 1 < 28 ? v[c + 1] : 0.0F);
      word[i] = *reinterpret_cast<const std::uint32_t*>(&pair);
    }
    *reinterpret_cast<uint4*>(x + pixel * 32 + 8 * static_cast<std::size_t>(j)) =
        make_uint4(word[0], word[1], word[2], word[3]);
  }
}

// Bloc 0 : 15 canaux = moyenne 2 × 2 (ou d × d centrés) des images, de l'encodage et de l'instant.
__global__ void __launch_bounds__(256) fused_entry(FusedEntryParams p) {
  const int ox = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int oy = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ox >= p.out_width || oy >= p.out_height) return;
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const std::size_t out_plane =
      static_cast<std::size_t>(p.out_height) * static_cast<std::size_t>(p.out_width);
  const int base = p.reduce / 2 - 1;
  const int px = p.reduce * ox + base;
  const int py = p.reduce * oy + base;
  const std::size_t i00 =
      static_cast<std::size_t>(py) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(px);
  const auto* input = static_cast<const __half*>(p.input);
  const auto* encode = static_cast<const __half*>(p.encode);
  const auto* encode_b = static_cast<const __half*>(p.encode_b);
  const std::size_t w = static_cast<std::size_t>(p.width);
  float v[15];
#pragma unroll
  for (int c = 0; c < 15; ++c) {
    const __half* src = c < 6    ? input + static_cast<std::size_t>(c) * plane
                        : c < 10 ? encode + static_cast<std::size_t>(c - 6) * plane
                        : c < 14 ? encode_b + static_cast<std::size_t>(c - 10) * plane
                                 : input + 6 * plane;
    if (c == 14) {  // l'instant : l'élément 0 du plan 6 seul est tenu (étape 5.2, rife_time)
      v[c] = __half2float(src[0]);
      continue;
    }
    v[c] = (__half2float(src[i00]) + __half2float(src[i00 + 1]) + __half2float(src[i00 + w]) +
            __half2float(src[i00 + w + 1])) *
           0.25F;
  }
  store_x<15, 16>(
      static_cast<__half*>(p.x), p.x_pitch, out_plane,
      static_cast<std::size_t>(oy) * static_cast<std::size_t>(p.out_width) + static_cast<std::size_t>(ox), v);
}

// Z (DepthToSpace CRD de Y, facteur 2) lu dans un Y relevé : Z[c][zy][zx] = Y[4c + 2 (zy & 1) + (zx & 1)].
__device__ __forceinline__ float z_at(const warp::Staged& s, int c, int zy, int zx) {
  return __half2float(s.at(zy >> 1, zx >> 1)[4 * c + 2 * (zy & 1) + (zx & 1)]);
}

// Resize bilinéaire demi-pixel du canal c de Z au point pleine résolution, arrondi FP16 comme la sortie du
// Resize du graphe.
__device__ __forceinline__ float z_enlarge(const warp::Staged& s, int c, const warp::Axis& ax,
                                           const warp::Axis& ay) {
  return rounded(warp::lerp2(z_at(s, c, ay.i0, ax.i0), z_at(s, c, ay.i0, ax.i1), z_at(s, c, ay.i1, ax.i0),
                             z_at(s, c, ay.i1, ax.i1), ax, ay));
}

// Canaux relevés par pixel de Y : 16 pour un flot seul (les 4 canaux de flot de Z à ses quatre parités), 56
// pour le dernier bloc (flot, masque, attributs : 52 canaux, complétés au pas HWC8).
constexpr int kFlowStride = 16;
constexpr int kLastStride = 56;

// La tuile de sorties d'un bloc de 256 fils et la place réservée par source (en pixels de Y). Une source
// d'échelle s ≥ d couvre au plus (largeur + 2) / 2 + 1 colonnes de Y et autant de lignes : 17 × 5 en pleine
// résolution, 10 × 4 réduite (vérifié pour d de 1 à 32 et s ≥ d) ; la marge couvre les arrondis.
template <bool Coarse>
struct GlueTile {
  static constexpr int kWidth = Coarse ? 16 : 32;
  static constexpr int kHeight = Coarse ? 4 : 8;
  static constexpr int kCapacity = Coarse ? 48 : 112;
};

// Relève dans stage la région de chaque Y que lisent les points pleine résolution px × py d'un bloc de fils ;
// src[j] la décrit. L'appelant synchronise.
template <int K, int Capacity>
__device__ __forceinline__ void stage_sources(const FusedGlueParams& p, warp::Span px, warp::Span py,
                                              __half* stage, warp::Staged (&src)[K]) {  // NOLINT
  int offset = 0;
#pragma unroll
  for (int j = 0; j < K; ++j) {
    const FusedFlowSource& f = p.source[j];
    const float inv = 1.0F / static_cast<float>(f.scale);
    const warp::Span zr = warp::coarse_span(py.lo, py.hi, inv, 2 * f.height);
    const warp::Span zc = warp::coarse_span(px.lo, px.hi, inv, 2 * f.width);
    const warp::Span rows{zr.lo >> 1, zr.hi >> 1};
    const warp::Span cols{zc.lo >> 1, zc.hi >> 1};
    const int stride = j == K - 1 ? kLastStride : kFlowStride;
    warp::stage_region(stage + offset, stride, static_cast<const __half*>(f.y), f.height, f.width, f.pitch, 0,
                       j == K - 1 ? 52 : 16, rows, cols);
    src[j] = {stage + offset, rows.lo, cols.lo, cols.hi - cols.lo + 1, stride};
    offset += stride * Capacity;
  }
}

// __maxnreg__ plutôt que __launch_bounds__ : nvcc 13 débordait de 4 octets à 48 registres (pleine résolution,
// un flot).
template <bool Coarse, int K>
__global__ void __maxnreg__(128) fused_glue(FusedGlueParams p) {
  using Tile = GlueTile<Coarse>;
  __shared__ __align__(16) __half stage[((K - 1) * kFlowStride + kLastStride) * Tile::kCapacity];
  const int d = Coarse ? p.reduce : 1;
  const int base = Coarse ? d / 2 - 1 : 0;
  const int last_tap = Coarse ? 1 : 0;
  const int bx = static_cast<int>(blockIdx.x) * Tile::kWidth;
  const int by = static_cast<int>(blockIdx.y) * Tile::kHeight;
  const int ex = min(bx + Tile::kWidth, p.out_width) - 1;
  const int ey = min(by + Tile::kHeight, p.out_height) - 1;

  // Le relèvement : la région de chaque Y que lisent les points pleine résolution du bloc.
  warp::Staged src[K];  // NOLINT(modernize-avoid-c-arrays)
  stage_sources<K, Tile::kCapacity>(p, {d * bx + base, d * ex + base + last_tap},
                                    {d * by + base, d * ey + base + last_tap}, stage, src);
  __syncthreads();

  const int t = static_cast<int>(threadIdx.x);
  const int lane = t & 31;
  int ox = 0;
  int oy = 0;
  int tx = 0;
  int ty = 0;
  if constexpr (Coarse) {
    tx = lane & 1;
    ty = (lane >> 1) & 1;
    ox = bx + (lane >> 2) + 8 * ((t >> 5) & 1);
    oy = by + (t >> 6);
  } else {
    ox = bx + lane;
    oy = by + (t >> 5);
  }
  const bool active = ox <= ex && oy <= ey;
  if (!Coarse && !active) return;  // pleine résolution : aucun échange entre fils
  ox = min(ox, ex);                // réduite : un fil hors image calcule un point valide, sans l'écrire
  oy = min(oy, ey);
  const int px = d * ox + base + tx;
  const int py = d * oy + base + ty;

  // Flot accumulé (Add : sommes FP16 de chaque bloc, multiplié par l'échelle de son Resize).
  float fl[4];
  warp::Axis ax{};
  warp::Axis ay{};
#pragma unroll
  for (int j = 0; j < K; ++j) {
    const FusedFlowSource& f = p.source[j];
    const float inv = 1.0F / static_cast<float>(f.scale);
    ax = warp::coarse_axis(px, inv, 2 * f.width);
    ay = warp::coarse_axis(py, inv, 2 * f.height);
    const float scale = static_cast<float>(f.scale);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float r = z_enlarge(src[j], k, ax, ay) * scale;
      fl[k] = j == 0 ? r : rounded(fl[k] + r);
    }
  }

  float v[28];
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const auto* input = static_cast<const __half*>(p.input);
  const warp::Bounds whole{0, 0, p.width, p.height};
  // Warp des deux images (7 canaux chacune : 3 de l'image, 4 de l'encodage).
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const warp::Tap tap = warp::tap_at(px, py, fl[2 * n], fl[2 * n + 1], whole);
    const auto* encode = static_cast<const __half*>(n == 0 ? p.encode : p.encode_b);
#pragma unroll
    for (int ch = 0; ch < 7; ++ch) {
      const __half* from = ch < 3 ? input + static_cast<std::size_t>(3 * n + ch) * plane
                                  : encode + static_cast<std::size_t>(ch - 3) * plane;
      v[ch < 3 ? 3 * n + ch : 6 + 4 * n + ch - 3] = rounded(warp::bilinear(from, tap, p.width));
    }
  }
  v[14] = __half2float(input[6 * plane]);  // l'instant, à l'élément 0 (étape 5.2)
  // Masque et attributs du dernier bloc calculé (ax, ay : ses axes, laissés par la dernière itération).
#pragma unroll
  for (int k = 0; k < 9; ++k) v[15 + k] = z_enlarge(src[K - 1], 4 + k, ax, ay);
  const float flow_scale = 1.0F / static_cast<float>(d);
#pragma unroll
  for (int k = 0; k < 4; ++k) v[24 + k] = rounded(fl[k] * flow_scale);

  const std::size_t out_plane =
      static_cast<std::size_t>(p.out_height) * static_cast<std::size_t>(p.out_width);
  const std::size_t o =
      static_cast<std::size_t>(oy) * static_cast<std::size_t>(p.out_width) + static_cast<std::size_t>(ox);
  auto* x = static_cast<__half*>(p.x);
  if constexpr (Coarse) {
#pragma unroll
    for (int c = 0; c < 28; ++c) v[c] = warp::sum_quad(v[c]) * 0.25F;
    if (!active) return;
    const int q = lane & 3;  // chacun des quatre fils de la sortie en écrit un quart
    if (p.x_pitch > 0) {
      store_chunk(x, o, q, v);
    } else {
#pragma unroll
      for (int c = 0; c < 28; ++c)
        if ((c & 3) == q) x[static_cast<std::size_t>(c) * out_plane + o] = __float2half_rn(v[c]);
    }
  } else {
    store_x<28, 32>(x, p.x_pitch, out_plane, o, v);
  }
}

// Fin : flot du dernier bloc ajouté ; image (warp bilinéaire ou Catmull-Rom, masque sigmoïde, mélange) ou
// flot et masque. Lectures groupées : les 4 canaux de flot de x_last en 8 octets, chaque canal de Z en 8
// octets (ses quatre parités), quand les tenseurs sont rangés par pixel.
__global__ void __maxnreg__(128) fused_finish(FusedFinishParams p) {  // même raison que fused_glue (sm_75)
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const std::size_t i =
      static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x);
  const auto* input = static_cast<const __half*>(p.input);
  const auto* last_x = static_cast<const __half*>(p.x_last);
  const auto* last_y = static_cast<const __half*>(p.y_last);

  float acc[4];
  if (p.x_pitch > 0) {
    const uint2 word = *reinterpret_cast<const uint2*>(last_x + i * static_cast<std::size_t>(p.x_pitch) + 24);
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&word.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&word.y));
    acc[0] = a.x;
    acc[1] = a.y;
    acc[2] = b.x;
    acc[3] = b.y;
  } else {
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] = __half2float(last_x[static_cast<std::size_t>(24 + k) * plane + i]);
  }
  const int parity = 2 * (y & 1) + (x & 1);
  const std::size_t y_pixel = static_cast<std::size_t>(y >> 1) * static_cast<std::size_t>(p.y_width) +
                              static_cast<std::size_t>(x >> 1);
  const std::size_t y_plane = static_cast<std::size_t>(p.y_height) * static_cast<std::size_t>(p.y_width);
  float z[5];
#pragma unroll
  for (int c = 0; c < 5; ++c) {
    if (p.y_pitch > 0) {
      const uint2 word =
          *reinterpret_cast<const uint2*>(last_y + y_pixel * static_cast<std::size_t>(p.y_pitch) + 4 * c);
      const std::uint32_t pair = (parity & 2) != 0 ? word.y : word.x;
      const __half2 h = *reinterpret_cast<const __half2*>(&pair);
      z[c] = __half2float((parity & 1) != 0 ? __high2half(h) : __low2half(h));
    } else {
      z[c] = __half2float(last_y[static_cast<std::size_t>(4 * c + parity) * y_plane + y_pixel]);
    }
  }
  float fl[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) fl[k] = rounded(acc[k] + z[k]);
  const float sig = rounded(1.0F / (1.0F + __expf(-z[4])));
  if (p.image == nullptr) {  // variante mouvement : le flot final et le masque
    auto* flow = static_cast<__half*>(p.flow);
    auto* out_mask = static_cast<__half*>(p.mask);
#pragma unroll
    for (int k = 0; k < 4; ++k) flow[static_cast<std::size_t>(k) * plane + i] = __float2half_rn(fl[k]);
    out_mask[i] = __float2half_rn(sig);
    return;
  }
  float warped[2][3];
  const warp::Bounds window{p.window_left, p.window_top, p.window_left + p.window_width,
                            p.window_top + p.window_height};
#pragma unroll
  for (int n = 0; n < 2; ++n) {
    const warp::Tap tap = warp::tap_at(x, y, fl[2 * n], fl[2 * n + 1], window);
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
      const __half* source = input + static_cast<std::size_t>(3 * n + ch) * plane;
      warped[n][ch] = rounded(p.cubic != 0 ? warp::catmull_rom(source, tap, p.width, window)
                                           : warp::bilinear(source, tap, p.width));
    }
  }
  const float rest = rounded(1.0F - sig);
  auto* image = static_cast<__half*>(p.image);
#pragma unroll
  for (int ch = 0; ch < 3; ++ch)
    image[static_cast<std::size_t>(ch) * plane + i] =
        __float2half_rn(rounded(warped[0][ch] * sig) + rounded(warped[1][ch] * rest));
}

dim3 grid_for(int width, int height, dim3 block) {
  return dim3((static_cast<unsigned>(width) + block.x - 1U) / block.x,
              (static_cast<unsigned>(height) + block.y - 1U) / block.y);
}

template <bool Coarse, int K>
void glue_launch(const FusedGlueParams& p, cudaStream_t stream) {
  using Tile = GlueTile<Coarse>;
  const dim3 grid((static_cast<unsigned>(p.out_width) + Tile::kWidth - 1U) / Tile::kWidth,
                  (static_cast<unsigned>(p.out_height) + Tile::kHeight - 1U) / Tile::kHeight);
  fused_glue<Coarse, K><<<grid, 256, 0, stream>>>(p);
}

template <bool Coarse>
void glue_launch(const FusedGlueParams& p, cudaStream_t stream) {
  switch (p.flows) {
    case 1:
      glue_launch<Coarse, 1>(p, stream);
      break;
    case 2:
      glue_launch<Coarse, 2>(p, stream);
      break;
    case 3:
      glue_launch<Coarse, 3>(p, stream);
      break;
    default:
      glue_launch<Coarse, 4>(p, stream);
      break;
  }
}

}  // namespace

cudaError_t launch_fused_entry(const FusedEntryParams& params, cudaStream_t stream) {
  const dim3 block(32, 8);
  fused_entry<<<grid_for(params.out_width, params.out_height, block), block, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_fused_glue(const FusedGlueParams& params, cudaStream_t stream) {
  // Réduction : une puissance de deux (base d / 2 − 1 et quatre points par sortie au-delà de 1).
  const int d = params.reduce;
  if (params.flows < 1 || params.flows > 4 || d < 1 || (d & (d - 1)) != 0) return cudaErrorInvalidValue;
  // La place du relèvement suppose chaque source au moins aussi grossière que la sortie (s ≥ d), ce que
  // suivent tous les RIFE découpés (réductions décroissantes) ; sinon refus, et le RIFE entier reprend.
  for (int j = 0; j < params.flows; ++j) {
    const FusedFlowSource& f = params.source[j];
    if (f.scale < params.reduce || (f.pitch != 0 && f.pitch != 56)) return cudaErrorInvalidValue;
  }
  if (params.x_pitch != 0 && params.x_pitch != 32) return cudaErrorInvalidValue;
  if (params.reduce > 1) {
    glue_launch<true>(params, stream);
  } else {
    glue_launch<false>(params, stream);
  }
  return cudaGetLastError();
}

cudaError_t launch_fused_finish(const FusedFinishParams& params, cudaStream_t stream) {
  // Les lectures groupées (8 octets) demandent les pas de HWC8 : 32 pour x_last, 56 pour y_last.
  if ((params.x_pitch != 0 && params.x_pitch != 32) || (params.y_pitch != 0 && params.y_pitch != 56))
    return cudaErrorInvalidValue;
  const dim3 block(32, 8);
  // Fenêtre vide (l'appelant ne la connaît pas) : le tenseur entier, comme le promet FusedFinishParams. Sans
  // ce repli, la prise bornait à [0, -1] et lisait avant le plan (image fausse des tests des noyaux).
  FusedFinishParams p = params;
  if (p.window_width <= 0 || p.window_height <= 0) {
    p.window_left = 0;
    p.window_top = 0;
    p.window_width = p.width;
    p.window_height = p.height;
  }
  fused_finish<<<grid_for(p.width, p.height, block), block, 0, stream>>>(p);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
