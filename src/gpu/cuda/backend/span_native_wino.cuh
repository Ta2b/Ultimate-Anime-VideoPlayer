#pragma once
// Les 3×3 carrées des blocs du SPAN natif par Winograd F(2×2, 3×3) (Lavin et Gray ; docs/span-native16.md,
// « Winograd ») : Y = Aᵀ [U ⊙ V] A, U = G g Gᵀ (les poids, rangés une fois à l'envoi, pack_wino), V = Bᵀ d B
// (chaque tuile 4×4 d'entrée). Seize produits par sortie 2×2 au lieu de trente-six : 2,25 fois moins de
// mma. À 64 canaux, sommes f16 seulement : le cran Qualité (f32) garde la convolution directe.
//
// Un bloc garde la tuile kTH × kTW de conv_kernel (mêmes bandes, même masque, même épilogue : finish), soit
// 4 × 16 tuiles Winograd ; ses 64 tuiles passent en deux moitiés de 32 (une ligne de fragments m16 chacune),
// faute de place pour V entier à côté de la tuile d'entrée sous 48 Ko. Pour chaque moitié et chacun des 16
// points ξ = (p, q) : tous les fils calculent V_ξ en mémoire partagée (sommes f32, un arrondi f16 : celui de
// tools/bench/span_native/winograd_db.py), puis le warp w multiplie V_ξ par U_ξ pour les sorties 16w à
// 16w + 15 (mma f16, 4 pas k) et ajoute M_ξ aux sorties 2×2 en f32 selon Aᵀ. Au bout, Y passe en f16 par la
// mémoire partagée et revient dans la disposition des accumulateurs de conv_kernel.
#include <array>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/span_native_conv.cuh"

namespace kirari::gpu::cuda::span_native {

inline constexpr int kWinoC = 64;                      // canaux du chemin Winograd
inline constexpr int kWinoPoints = 16;                 // points ξ = 4p + q de F(2×2, 3×3)
inline constexpr int kWinoCols = kTW / 2;              // tuiles 2×2 par ligne de la tuile de bloc
inline constexpr int kWinoHalf = kTH / 4 * kWinoCols;  // tuiles 2×2 par moitié
inline constexpr int kWinoTile = (kTH + 2) * (kTW + 2) * kWinoC * 2;  // la tuile d'entrée, Tile<64, true>

// Octets de mémoire partagée de wino_kernel : la tuile d'entrée et V d'une moitié (pas C + 8), ou la sortie
// et la 1×1 de la porte.
constexpr int wino_smem_bytes() {
  const int in = kWinoTile + kWinoHalf * (kWinoC + 8) * 2;
  const int out = kTH * kTW * (kWinoC + 8) * 2 + kWinoC * kWinoC * 2;
  return in > out ? in : out;
}
static_assert(wino_smem_bytes() <= 48 * 1024, "deux blocs par SM sans réglage de la mémoire partagée");

// La première ligne de la grille de wino_kernel pour une bande qui commence à y0.
__host__ __device__ constexpr int wino_first_row(int y0) {
  return y0 & ~1;
}

// Bᵀ : la ligne p vaut d[first] + sign · d[second] (Bᵀ = [1 0 −1 0 ; 0 1 1 0 ; 0 −1 1 0 ; 0 1 0 −1]).
__device__ __forceinline__ int bt_first(int p) {
  return p == 2 ? 2 : p == 0 ? 0 : 1;
}
__device__ __forceinline__ int bt_second(int p) {
  return p == 3 ? 3 : p == 2 ? 1 : 2;
}
__device__ __forceinline__ float bt_sign(int p) {
  return p == 1 ? 1.F : -1.F;
}
// Aᵀ = [1 1 1 0 ; 0 1 −1 −1] : le poids du point p dans la sortie i.
__device__ __forceinline__ float at_coef(int i, int p) {
  if (i == 0) return p < 3 ? 1.F : 0.F;
  return p == 0 ? 0.F : p == 1 ? 1.F : -1.F;
}

// V_ξ d'une moitié vers v (32 tuiles au pas C + 8) : 32 tuiles × 8 morceaux de 8 canaux, deux par fil. Le
// morceau d'une tuile se lit en quatre lds128 (les quatre termes non nuls de Bᵀ d B) ; huit fils de suite
// lisent les huit morceaux d'un pixel et écrivent ceux d'une tuile : pas de conflit de banques.
__device__ __forceinline__ void wino_input(std::uint32_t tile, __half* v, int half, int point) {
  using T = Tile<kWinoC, true>;
  constexpr int kSW = kTW + 2, kChunks = kWinoC / 8, kLdv = kWinoC + 8;
  const int pp = point >> 2, qq = point & 3;
  const int r0 = bt_first(pp), r1 = bt_second(pp), c0 = bt_first(qq), c1 = bt_second(qq);
  const float sr = bt_sign(pp), sc = bt_sign(qq);
#pragma unroll
  for (int k = 0; k < kWinoHalf * kChunks / kThreads; ++k) {
    const int item = static_cast<int>(threadIdx.x) + k * kThreads;
    const int tl = item / kChunks, c = item - tl * kChunks;
    const int y = 2 * (2 * half + tl / kWinoCols), x = 2 * (tl % kWinoCols);
    const auto load = [&](int r, int col) {
      const uint4 d = lds128(tile + static_cast<std::uint32_t>(T::at((y + r) * kSW + x + col, c)));
      return std::array<std::uint32_t, 4>{d.x, d.y, d.z, d.w};
    };
    const std::array<std::uint32_t, 4> d00 = load(r0, c0), d01 = load(r0, c1), d10 = load(r1, c0),
                                       d11 = load(r1, c1);
    std::uint32_t out[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 e00 = unpack_half2(d00[j]), e01 = unpack_half2(d01[j]);
      const float2 e10 = unpack_half2(d10[j]), e11 = unpack_half2(d11[j]);
      out[j] = pack_half2((e00.x + sc * e01.x) + sr * (e10.x + sc * e11.x),
                          (e00.y + sc * e01.y) + sr * (e10.y + sc * e11.y));
    }
    *reinterpret_cast<uint4*>(v + tl * kLdv + c * 8) = make_uint4(out[0], out[1], out[2], out[3]);
  }
}

// Y d'une moitié (ya[ligne de fragments][sortie i·2 + j][tuile n8][4 valeurs], disposition d'un Frag<kF32> :
// tuiles g et g + 8, sorties 2t et 2t + 1) en f16 vers o, au pas C + 8, aux pixels de la tuile de bloc.
__device__ __forceinline__ void wino_output(const float (&ya)[2][4][2][4], __half* o, int half, int warp,
                                            int lane) {
  constexpr int kLdo = kWinoC + 8;
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int ij = 0; ij < 4; ++ij)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int ty = 2 * half + m, tx = g + 8 * h;
          const int pixel = (2 * ty + (ij >> 1)) * kTW + 2 * tx + (ij & 1);
          const int ch = warp * 16 + nt * 8 + 2 * t;
          *reinterpret_cast<__half2*>(o + pixel * kLdo + ch) =
              to_half2(ya[m][ij][nt][2 * h], ya[m][ij][nt][2 * h + 1]);
        }
}

// Une 3×3 carrée à 64 canaux par Winograd, sommes f16 ; p.w : les poids de pack_wino. Mêmes paramètres,
// même grille et même sortie que conv_kernel<64, 64, 9, 1, kNhwc, M, Relu, kF16>.
template <Mode M, bool Relu>
__global__ void __launch_bounds__(kThreads) wino_kernel(Params p) {
  static_assert(M != Mode::kPixel, "to_pixel n'est pas carrée");
  constexpr int C = kWinoC, kKC = C / 16, kNP = C / 16, kLdv = C + 8;
  static_assert(kNP == kWarps, "un warp par tranche de 16 sorties");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto* s = reinterpret_cast<__half*>(smem_raw);
  auto* v = reinterpret_cast<__half*>(smem_raw + kWinoTile);
  const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
  // Les tuiles 2×2 partent des lignes paires de l'image, quelle que soit la bande : la sortie ne dépend pas
  // des bandes, au bit près. Une bande qui commence à une ligne impaire calcule la ligne d'avant sans
  // l'écrire (store_nhwc).
  const int ty0 = wino_first_row(p.y0) + static_cast<int>(blockIdx.y) * kTH;
  const int tx0 = static_cast<int>(blockIdx.x) * kTW;
  if (p.mask != nullptr) {  // tout le bloc sort ensemble, avant la première barrière
    const unsigned tile = blockIdx.y * gridDim.x + blockIdx.x;
    if (((__ldg(p.mask + (tile >> 5U)) >> (tile & 31U)) & 1U) == 0U) return;
  }
  load_nhwc<C, 1, true>(s, p.in[0], p.in_ld != 0 ? p.in_ld : C, p.fh, p.fw, ty0, tx0);
  cp_async_wait_all();
  const std::uint32_t tile = smem_u32(s);
  // ldmatrix.x4 sur V : la lane l donne la tuile l % 16 du fragment, demi-K l / 16.
  const std::uint32_t abase =
      smem_u32(v) + static_cast<std::uint32_t>(((lane & 15) * kLdv + (lane >> 4) * 8) * 2);
  const uint4* wl = p.w + warp * 32 + lane;  // la paire de tuiles n8 du warp
  const int t = lane & 3;
  float ya[2][4][2][4];            // la moitié en cours
  std::uint32_t kept[2][4][2][2];  // la première moitié, en f16
#pragma unroll 1
  for (int half = 0; half < 2; ++half) {
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int ij = 0; ij < 4; ++ij)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int r = 0; r < 4; ++r) ya[m][ij][nt][r] = 0.F;
#pragma unroll 1
    for (int point = 0; point < kWinoPoints; ++point) {
      __syncthreads();  // la tuile est là (premier tour) ; V du point d'avant n'est plus lu
      wino_input(tile, v, half, point);
      __syncthreads();
      Frag<Acc::kF16> mm[2][2];
#pragma unroll
      for (int m = 0; m < 2; ++m) mm[m][0].clear(), mm[m][1].clear();
#pragma unroll
      for (int ks = 0; ks < kKC; ++ks) {
        const uint4 b = __ldg(wl + (point * kKC + ks) * kNP * 32);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          std::uint32_t a[4];
          ldmatrix4(a, abase + static_cast<std::uint32_t>((m * 16 * kLdv + ks * 16) * 2));
          mm[m][0].mma(a, b.x, b.y);
          mm[m][1].mma(a, b.z, b.w);
        }
      }
      // M_ξ dans les quatre sorties selon Aᵀ ; une somme débordée est bornée (0 · inf ferait un NaN).
      const int pp = point >> 2, qq = point & 3;
      float coef[4];
#pragma unroll
      for (int ij = 0; ij < 4; ++ij) coef[ij] = at_coef(ij >> 1, pp) * at_coef(ij & 1, qq);
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float2 f = mm[m][nt].pair(h);
            const float lo = fmaxf(fminf(f.x, 65504.F), -65504.F), hi = fmaxf(fminf(f.y, 65504.F), -65504.F);
#pragma unroll
            for (int ij = 0; ij < 4; ++ij) {
              ya[m][ij][nt][2 * h] += coef[ij] * lo;
              ya[m][ij][nt][2 * h + 1] += coef[ij] * hi;
            }
          }
    }
    if (half == 0) {
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int ij = 0; ij < 4; ++ij)
#pragma unroll
          for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const __half2 y = to_half2(ya[m][ij][nt][2 * h], ya[m][ij][nt][2 * h + 1]);
              kept[m][ij][nt][h] = *reinterpret_cast<const std::uint32_t*>(&y);
            }
    }
  }
  __syncthreads();  // la tuile et V ne sont plus lus : la mémoire partagée reçoit Y
  {
    float y0[2][4][2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int ij = 0; ij < 4; ++ij)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const float2 f = unpack_half2(kept[m][ij][nt][h]);
            y0[m][ij][nt][2 * h] = f.x;
            y0[m][ij][nt][2 * h + 1] = f.y;
          }
    wino_output(y0, s, 0, warp, lane);
  }
  wino_output(ya, s, 1, warp, lane);
  __syncthreads();
  // Y revient dans la disposition des accumulateurs de conv_kernel (accumulate) pour l'épilogue commun.
  Frag<Acc::kF16> acc[kMF][C / 8];
  {
    constexpr int kLdo = C + 8;
    const int g = lane >> 2;
#pragma unroll
    for (int f = 0; f < kMF; ++f) {
      const int q0 = (warp * kRows + (f >> 1)) * kTW + 16 * (f & 1) + g;
#pragma unroll
      for (int nt = 0; nt < C / 8; ++nt)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          acc[f][nt].r[h] = *reinterpret_cast<const std::uint32_t*>(s + (q0 + 8 * h) * kLdo + nt * 8 + 2 * t);
    }
  }
  __syncthreads();  // Y est lu : la porte peut charger x à sa place
  finish<C, M, Relu, Acc::kF16>(acc, s, p, ty0, tx0, warp, lane);
}

// Les poids d'une 3×3 carrée ([sortie][position][entrée] en FP16) pour wino_kernel : U_ξ = G g Gᵀ, calculé
// en double et arrondi en f16 (comme winograd_db.py), rangé par pack avec les 16 points à la place des 9
// positions (G = [1 0 0 ; ½ ½ ½ ; ½ −½ ½ ; 0 0 1]).
inline std::vector<std::uint32_t> pack_wino(const std::uint16_t* w, int cout, int cin) {
  static constexpr double kG[4][3] = {{1.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0.0, 0.0, 1.0}};
  std::vector<std::uint16_t> u(static_cast<std::size_t>(cout) * kWinoPoints * cin);
  for (int o = 0; o < cout; ++o)
    for (int c = 0; c < cin; ++c) {
      double g[3][3];
      for (int tap = 0; tap < 9; ++tap) {
        __half_raw raw;
        raw.x = w[(static_cast<std::size_t>(o) * 9 + tap) * cin + c];
        g[tap / 3][tap % 3] = static_cast<double>(__half2float(__half(raw)));
      }
      for (int pp = 0; pp < 4; ++pp)
        for (int qq = 0; qq < 4; ++qq) {
          double sum = 0.0;
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) sum += kG[pp][a] * kG[qq][b] * g[a][b];
          const __half_raw out = __float2half_rn(static_cast<float>(sum));
          u[(static_cast<std::size_t>(o) * kWinoPoints + pp * 4 + qq) * cin + c] = out.x;
        }
    }
  return pack(u.data(), cout, kWinoPoints, cin);
}

}  // namespace kirari::gpu::cuda::span_native
