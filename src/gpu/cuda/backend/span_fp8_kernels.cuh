#pragma once
// Noyaux du SPAN entier en FP8 (docs/span-fp8-cutlass.md), prototypés dans tools/bench/fp8/cutlass : une
// seule boucle principale CUTLASS (convolution implicit GEMM, entrée FP8 E4M3 ou FP16) et un épilogue écrit à
// la main qui sait :
//   Mode 0 : biais (+ Relu) puis écriture E4M3 (sortie FP8, échelle statique) et/ou FP16 (tranche d'un tampon
//            de concaténation : pas de ligne ld16 en éléments)
//   Mode 1 : fin de bloc SPAN (saut x + Tanh(conv 1x1 de c3) + produit), mêmes sorties
//   Mode 2 : to_pixel (étape 3.4a) : biais puis DepthToSpace(4, CRD) fondu, écriture NCHW FP16 de la sortie
//            et/ou sur la surface d'un emplacement de l'anneau (store_ring)
// Le warp porte les colonnes de sortie complètes (64, ou 48 pour un modèle à 48 canaux natifs) :
// l'accumulateur m16n8 de deux tuiles voisines est l'opérande A d'un mma m16n8k16 f16 pour la conv 1x1.
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/numeric_types.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "gpu/cuda/backend/ring_surface.h"

namespace ajp::gpu::cuda::span_fp8_kernels {

struct KP {
  const uint8_t* skip = nullptr;  // saut x, NHWC E4M3 de la largeur du bloc (mode 1)
  float skip_scale = 0.f;
  float alpha = 1.f;  // accumulateur -> unités réelles
  // Étape 10.3 : non nul, une échelle par colonne de sortie (activation × canal des poids FP8), à la place
  // de alpha ; lu par paires (float2), comme le biais.
  const float* alpha_col = nullptr;
  const float* bias = nullptr;  // biais de la conv 3x3 (unités réelles)
  const __half* w1 = nullptr;   // poids 1x1 [sortie][entrée], FP16, N×N (mode 1)
  const float* b1 = nullptr;    // biais de la 1x1 (mode 1)
  uint8_t* out8 = nullptr;      // sortie E4M3 NHWC (N canaux, la largeur du bloc) ou nullptr
  float inv_s8 = 1.f;           // 1 / échelle de la sortie E4M3
  __half* out16 = nullptr;      // sortie FP16 (déjà décalée à la tranche de canaux) ou nullptr
  int ld16 = 0;                 // pas de ligne de out16, en éléments
  int ncols = 64;               // colonnes de sortie utiles (64 ou 48)
  long rows = 0;                // lignes de la convolution (M = H·W) : la dernière tuile garde les siennes
  int fh = 0, fw = 0;           // hauteur et largeur de la convolution (posées par Launcher::init)
  // Mode 2 : out16 est la sortie NCHW 3×4fh×4fw (ou nullptr) ; surface un emplacement RGBA16 UNORM de
  // surf_w×surf_h pixels au plus (0 : pas d'anneau).
  RingSurface surface{};  // emplacement de l'anneau (ring_surface.h) ; object 0 : aucun
  int surf_w = 0, surf_h = 0;
  // Accumulateur f16 seulement : un mot de la mémoire de l'hôte projetée (cudaHostAllocMapped), mis à 1
  // quand une somme a débordé (infinie ou NaN). La somme est alors bornée à ±65504 (NaN à 0) avant
  // l'épilogue ; SpanFp8Net::poll_accumulator compte l'image et applique span_fp8::Acc16Policy.
  int* overflow = nullptr;
};

// Une valeur de l'accumulateur en unités du mma. Guard (accumulateur f16) : une somme débordée est bornée au
// plus grand f16 de son signe (NaN, un débordement des deux signes, à 0) et hit le note.
template <bool Guard>
__device__ __forceinline__ float acc_value(float x, bool& hit) {
  if constexpr (Guard) {
    if (!isfinite(x)) {
      hit = true;
      return isnan(x) ? 0.f : copysignf(65504.f, x);
    }
  } else {
    static_cast<void>(hit);
  }
  return x;
}

// Les sorties de la tuile m du fil, en unités réelles : accumulateur × alpha (par colonne ou global) + biais.
// Tuile d'accumulateur (m, n) = a[(m + n*kMI)*4 + e] : e=0,1 -> ligne g, colonnes 2t,2t+1 ; e=2,3 -> g+8.
// Vrai si une somme avait débordé (Guard, acc_value).
template <int kNT, int kMI, bool Guard, class Element>
__device__ __forceinline__ bool scaled_acc(float (&v)[kNT][4], const Element* a, const KP& tp, int m, int t,
                                           int nt_used) {
  bool hit = false;
#pragma unroll
  for (int n = 0; n < kNT; ++n) {
    float2 bb =
        n < nt_used ? __ldg(reinterpret_cast<const float2*>(tp.bias + n * 8 + 2 * t)) : make_float2(0.f, 0.f);
    const float2 aa = tp.alpha_col != nullptr && n < nt_used
                          ? __ldg(reinterpret_cast<const float2*>(tp.alpha_col + n * 8 + 2 * t))
                          : make_float2(tp.alpha, tp.alpha);
    v[n][0] = acc_value<Guard>(float(a[(m + n * kMI) * 4 + 0]), hit) * aa.x + bb.x;
    v[n][1] = acc_value<Guard>(float(a[(m + n * kMI) * 4 + 1]), hit) * aa.y + bb.y;
    v[n][2] = acc_value<Guard>(float(a[(m + n * kMI) * 4 + 2]), hit) * aa.x + bb.x;
    v[n][3] = acc_value<Guard>(float(a[(m + n * kMI) * 4 + 3]), hit) * aa.y + bb.y;
  }
  return hit;
}

// Rare : plusieurs fils écrivent la même valeur, sans atomique.
__device__ __forceinline__ void note_overflow(const KP& tp, bool hit) {
  if (hit && tp.overflow != nullptr) *reinterpret_cast<volatile int*>(tp.overflow) = 1;
}

__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Écrit une paire de canaux (2t, 2t+1 de la tuile n) d'une ligne, dans les sorties demandées. ld8 : pas de
// ligne de out8, la largeur du bloc (64 ou 48). Act16 : out8 porte des activations FP16 (SPAN FP16 en noyaux
// CUTLASS, sans quantification ni inv_s8) au lieu d'E4M3.
template <bool Act16>
__device__ __forceinline__ void store_pair(const KP& p, int ld8, long row, int col, float x0, float x1) {
  if constexpr (Act16) {
    if (p.out8)
      *reinterpret_cast<__half2*>(reinterpret_cast<__half*>(p.out8) + row * ld8 + col) =
          __floats2half2_rn(x0, x1);
  } else if (p.out8) {
    *reinterpret_cast<uint16_t*>(p.out8 + row * ld8 + col) =
        __nv_cvt_float2_to_fp8x2(make_float2(x0 * p.inv_s8, x1 * p.inv_s8), __NV_SATFINITE, __NV_E4M3);
  }
  if (p.out16) *reinterpret_cast<__half2*>(p.out16 + row * p.ld16 + col) = __floats2half2_rn(x0, x1);
}

// La paire de canaux (col, col + 1) du saut x à la ligne row (pas ld), en bits : E4M3 ×2 ou FP16 ×2.
template <bool Act16>
__device__ __forceinline__ uint32_t load_skip(const uint8_t* skip, long row, int ld, int col) {
  if constexpr (Act16)
    return *reinterpret_cast<const uint32_t*>(reinterpret_cast<const __half*>(skip) + row * ld + col);
  return *reinterpret_cast<const uint16_t*>(skip + row * ld + col);
}

template <bool Act16>
__device__ __forceinline__ float2 skip_value(uint32_t bits, float scale) {
  if constexpr (Act16) {
    __half2 h;
    *reinterpret_cast<uint32_t*>(&h) = bits;
    return __half22float2(h);
  }
  __nv_fp8x2_e4m3 q;
  q.__x = static_cast<__nv_fp8x2_storage_t>(bits);
  const float2 f = q.operator float2();  // pas float2(q) : en C++20 c'est une initialisation d'agrégat
  return make_float2(f.x * scale, f.y * scale);
}

// Copie W1 (kN × kN demi-flottants) en smem, au pas kLd.
template <int kN, int kLd, int kThreads>
__device__ __forceinline__ void load_w1(__half* w1s, const __half* w1) {
  constexpr int kNT = kN / 8;
  for (int i = threadIdx.x; i < kN * kNT; i += kThreads)
    *reinterpret_cast<uint4*>(w1s + (i / kNT) * kLd + (i % kNT) * 8) =
        *reinterpret_cast<const uint4*>(w1 + (i / kNT) * kN + (i % kNT) * 8);
}

// Un bloc par SM au moins : sans ce second argument, ptxas plafonnait deux noyaux à 48 colonnes à 128
// registres et débordait de 16 octets (nvcc 13.4, sm 89).
template <class Base, int Mode, bool Relu, bool Act16>
__global__ void __launch_bounds__(Base::kThreadCount, 1) net_kernel(typename Base::Params params, KP tp) {
  using Mma = typename Base::Mma;
  constexpr int kWarpM = Mma::Policy::Operator::Shape::kM;
  constexpr int kMI = kWarpM / 16;
  static_assert(Mma::WarpCount::kN == 1 && Mma::WarpCount::kK == 1,
                "un warp doit porter toutes les colonnes");
  constexpr int kN = Mma::Shape::kN;  // 64, ou 48 pour un modèle à 48 canaux natifs (kKernelsVersion 4)
  static_assert(kN == 64 || kN == 48, "le bloc doit couvrir 64 ou 48 colonnes");
  constexpr int kNT = kN / 8;   // tuiles n8 de l'accumulateur
  constexpr int kKS = kN / 16;  // pas k16 de la conv 1x1 du mode 1
  constexpr int kLd = kN + 8;   // pas de W1 en smem (36 ou 28 mots) : sans conflit de banques

  extern __shared__ __align__(16) unsigned char smem[];
  auto& ss = *reinterpret_cast<typename Mma::SharedStorage*>(smem);
  __half* w1s = reinterpret_cast<__half*>(smem + ((sizeof(typename Mma::SharedStorage) + 15) / 16) * 16);

  typename Base::ThreadblockSwizzle swizzle;
  cutlass::gemm::GemmCoord tile = swizzle.get_tile_offset(params.swizzle_log_tile);
  if (params.grid_tiled_shape.m() <= tile.m() || params.grid_tiled_shape.n() <= tile.n()) return;

  int thread_idx = threadIdx.x;
  if constexpr (Mode == 1) load_w1<kN, kLd, Base::kThreadCount>(w1s, tp.w1);
  __syncthreads();

  typename Mma::IteratorA iterator_A(
      params.iterator_A, params.problem_size, params.ptr_A, thread_idx,
      cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
  typename Mma::IteratorB iterator_B(
      params.iterator_B, params.problem_size, params.ptr_B, thread_idx,
      cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane = threadIdx.x % 32;
  Mma mma(ss, thread_idx, warp_idx, lane);
  typename Mma::FragmentC acc;
  acc.clear();
  mma(params.gemm_k_iterations, acc, iterator_A, iterator_B, acc, params.gemm_k_iterations_per_channel);

  // Tuile d'accumulateur (m, n) = acc[(m + n*kMI)*4 + e] : e=0,1 -> ligne g, colonnes 2t,2t+1 ; e=2,3 -> g+8.
  const auto* a = acc.data();
  const int g = lane >> 2, t = lane & 3;
  const int warp_m = warp_idx % Mma::WarpCount::kM;
  const long row_base = long(tile.m()) * Mma::Shape::kM + warp_m * kWarpM;
  const int nt_used = tp.ncols >> 3;
  constexpr bool kGuard = std::is_same_v<typename Mma::FragmentC::Element, cutlass::half_t>;
  bool hit = false;  // une somme de l'accumulateur f16 débordée (acc_value)

#pragma unroll
  for (int m = 0; m < kMI; ++m) {
    const long r0 = row_base + m * 16 + g, r1 = r0 + 8;
    // Étape 3.1 : une ligne au-delà de M n'est ni lue (saut) ni écrite. Les mma_f16 du mode 1 tournent quand
    // même : ils sont collectifs au warp.
    const bool in0 = r0 < tp.rows, in1 = r1 < tp.rows;
    float v[kNT][4];
    hit |= scaled_acc<kNT, kMI, kGuard>(v, a, tp, m, t, nt_used);
    if constexpr (Mode == 0) {
#pragma unroll
      for (int n = 0; n < kNT; ++n) {
        if (n >= nt_used) continue;
        if constexpr (Relu) {
          v[n][0] = fmaxf(v[n][0], 0.f);
          v[n][1] = fmaxf(v[n][1], 0.f);
          v[n][2] = fmaxf(v[n][2], 0.f);
          v[n][3] = fmaxf(v[n][3], 0.f);
        }
        if (in0) store_pair<Act16>(tp, kN, r0, n * 8 + 2 * t, v[n][0], v[n][1]);
        if (in1) store_pair<Act16>(tp, kN, r1, n * 8 + 2 * t, v[n][2], v[n][3]);
      }
    } else if constexpr (Mode == 2) {
      static_assert(kNT >= 6, "to_pixel a 48 colonnes");
      // Canal = 16c + 4bh + bw. Le fil tient les colonnes 8n + 2t et 8n + 2t + 1 (n < 6) : pour j = n % 2, la
      // ligne de sortie bh = 2j + t/2 et les pixels bw = 2(t%2) et 2(t%2) + 1 des trois plans c = n / 2, soit
      // R, G et B complets de 4 pixels de sortie par ligne de convolution. Arrondi à half avant unorm16 : le
      // code que write_rgba16 écrirait de la sortie FP16, au bit près.
      const int ow = 4 * tp.fw;
      const long plane = long(4 * tp.fh) * ow;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if (!(h ? in1 : in0)) continue;
        const long row = h ? r1 : r0;
        const int y = int(row / tp.fw), x = int(row % tp.fw);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int oy = 4 * y + 2 * j + (t >> 1), ox = 4 * x + 2 * (t & 1);
          __half2 rgb[3];
#pragma unroll
          for (int c = 0; c < 3; ++c)
            rgb[c] = __floats2half2_rn(v[2 * c + j][2 * h], v[2 * c + j][2 * h + 1]);
          if (tp.out16) {
#pragma unroll
            for (int c = 0; c < 3; ++c)
              *reinterpret_cast<__half2*>(tp.out16 + c * plane + long(oy) * ow + ox) = rgb[c];
          }
          if (tp.surface.object != 0 && oy < tp.surf_h) {
            if (ox < tp.surf_w)
              store_ring(tp.surface, ox, oy, __low2half(rgb[0]), __low2half(rgb[1]), __low2half(rgb[2]));
            if (ox + 1 < tp.surf_w)
              store_ring(tp.surface, ox + 1, oy, __high2half(rgb[0]), __high2half(rgb[1]),
                         __high2half(rgb[2]));
          }
        }
      }
    } else {
      uint32_t sk[kNT][2];
#pragma unroll
      for (int n = 0; n < kNT; ++n) {
        sk[n][0] = in0 ? load_skip<Act16>(tp.skip, r0, kN, n * 8 + 2 * t) : 0U;
        sk[n][1] = in1 ? load_skip<Act16>(tp.skip, r1, kN, n * 8 + 2 * t) : 0U;
      }
      uint32_t A[kKS][4];
#pragma unroll
      for (int ks = 0; ks < kKS; ++ks) {
        A[ks][0] = pack_half2(v[2 * ks][0], v[2 * ks][1]);
        A[ks][1] = pack_half2(v[2 * ks][2], v[2 * ks][3]);
        A[ks][2] = pack_half2(v[2 * ks + 1][0], v[2 * ks + 1][1]);
        A[ks][3] = pack_half2(v[2 * ks + 1][2], v[2 * ks + 1][3]);
      }
#pragma unroll
      for (int nt = 0; nt < kNT; ++nt) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < kKS; ++ks) {
          const __half* wp = w1s + (nt * 8 + g) * kLd + ks * 16 + 2 * t;
          mma_f16(d, A[ks], *reinterpret_cast<const uint32_t*>(wp),
                  *reinterpret_cast<const uint32_t*>(wp + 8));
        }
        float2 bb1 = __ldg(reinterpret_cast<const float2*>(tp.b1 + nt * 8 + 2 * t));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float2 skf = skip_value<Act16>(sk[nt][h], tp.skip_scale);
          float s0 = v[nt][2 * h] + skf.x, s1 = v[nt][2 * h + 1] + skf.y;
          float o0 = s0 * tanh_approx(d[2 * h] + bb1.x);
          float o1 = s1 * tanh_approx(d[2 * h + 1] + bb1.y);
          if (h ? in1 : in0) store_pair<Act16>(tp, kN, h ? r1 : r0, nt * 8 + 2 * t, o0, o1);
        }
      }
    }
  }
  if constexpr (kGuard) note_overflow(tp, hit);
}

// EIn : float_e4m3_t ou half_t. Acc : float ou half_t (accumulateur du mma). TN : colonnes du bloc, 64 ou 48
// (canaux de sortie natifs d'un modèle à 48 canaux). Alg kFewChannels : la dimension K est (ligne, colonne,
// canal) mise à plat, découpée en tuiles de TK sans retour à chaque position du filtre ; c'est ce qui évite,
// pour 48 canaux d'entrée et des tuiles K de 32 ou 64, la queue masquée de chaque position (chantier SPAN 1).
template <class EIn, int TM, int TK, int WM, int Stages, class Acc = float,
          cutlass::conv::IteratorAlgorithm Alg = cutlass::conv::IteratorAlgorithm::kOptimized, int TN = 64>
struct Cfg {
  using E = EIn;
  static constexpr int kN = TN;
  // À 48 colonnes, trois warps au lieu de quatre (96 lignes de tuile pour 128) : la carte des fils de CUTLASS
  // (PitchLinearWarpRakedThreadMap) range les 48 lignes de la tuile des poids par 8 lignes et par warp, et
  // divise 6 passes entre 4 warps en entier, soit une passe : les lignes 32 à 47 n'étaient jamais chargées,
  // et la mémoire partagée gardait ce qu'un autre noyau y avait laissé (bleu faux de to_pixel, NaN des
  // blocs à 48 canaux, résultats qui changent avec l'ordre des lancements). 6 passes pour 3 warps tombent
  // juste ; covers ci-dessous le vérifie à la compilation.
  static constexpr int kTM = TN == 48 ? TM / 4 * 3 : TM;
  static constexpr bool kFp8 = cutlass::sizeof_bits<EIn>::value == 8;
  using Inst = typename std::conditional<kFp8, cutlass::gemm::GemmShape<16, 8, 32>,
                                         cutlass::gemm::GemmShape<16, 8, 16>>::type;
  using Dummy = cutlass::epilogue::thread::LinearCombination<EIn, 8, Acc, float>;  // jamais exécuté
  using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
      EIn, cutlass::layout::TensorNHWC, EIn, cutlass::layout::TensorNHWC, EIn, cutlass::layout::TensorNHWC,
      Acc, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89, cutlass::gemm::GemmShape<kTM, TN, TK>,
      cutlass::gemm::GemmShape<WM, TN, TK>, Inst, Dummy,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, cutlass::arch::OpMultiplyAdd,
      Alg>::Kernel;

  // Une carte des fils couvre sa tuile entière : passes × fils × éléments par accès = éléments de la tuile.
  template <class Map>
  static constexpr bool covers =
      Map::Iterations::kContiguous * Map::Iterations::kStrided * Map::kThreads * Map::kElementsPerAccess ==
      Map::Shape::kContiguous * Map::Shape::kStrided;
  static_assert(covers<typename Base::Mma::IteratorA::ThreadMap>, "tuile des activations chargée en partie");
  static_assert(covers<typename Base::Mma::IteratorB::ThreadMap>, "tuile des poids chargée en partie");
};

// Un lancement : paramètres CUTLASS construits une fois, lancement compatible avec la capture de graphe.
// Act16 : les activations out8 et skip sont en FP16 (SPAN FP16), sinon en E4M3.
template <class C, int Mode, bool Relu, bool Act16 = false>
struct Launcher {
  using Base = typename C::Base;
  using E = typename C::E;
  typename Base::Params params;
  KP kp;
  dim3 grid, block;
  size_t smem = 0;
  // Refus de cudaFuncSetAttribute (mémoire partagée au-delà de 48 Ko). Posé à init, puis de nouveau avant le
  // premier lancement : net_kernel est instancié dans plusieurs .cu (référence et réseau FP8), et en Release
  // le pliage des souches de l'hôte faisait viser à l'attribut le module d'un autre fichier que celui du
  // lancement (« invalid argument », 58 368 o, RTX 4070 Ti). Posé dans launch, il vise le même.
  mutable cudaError_t attribute = cudaSuccess;
  mutable bool armed = false;

  // Convolution RxR (remplissage pad), Cin canaux d'entrée, Kout canaux de sortie, image HxW.
  void init(int H, int W, int R, int pad, int Cin, int Kout, const E* in, const E* w, const KP& k) {
    cutlass::conv::Conv2dProblemSize ps({1, H, W, Cin}, {Kout, R, R, Cin}, {pad, pad, pad, pad}, {1, 1},
                                        {1, 1}, cutlass::conv::Mode::kCrossCorrelation, 1);
    E* dummy = reinterpret_cast<E*>(0x1000);  // jamais déréférencé : l'épilogue standard n'est pas exécuté
    typename Base::Arguments args(
        ps, {const_cast<E*>(in), {Cin, Cin * W, Cin * W * H}},
        {const_cast<E*>(w), {Cin, Cin * R, Cin * R * R}}, {dummy, {Kout, Kout * W, Kout * W * H}},
        {dummy, {Kout, Kout * W, Kout * W * H}}, typename C::Dummy::Params(1.f, 0.f));
    params = typename Base::Params(args, nullptr);
    kp = k;
    kp.ncols = Kout;
    kp.rows = long(H) * W;
    kp.fh = H;
    kp.fw = W;
    grid = typename Base::ThreadblockSwizzle().get_grid_shape(params.grid_tiled_shape);
    block = dim3(Base::kThreadCount);
    smem = ((sizeof(typename Base::Mma::SharedStorage) + 15) / 16) * 16 + C::kN * (C::kN + 8) * 2;
    if (smem > 48 * 1024)
      attribute = cudaFuncSetAttribute(net_kernel<Base, Mode, Relu, Act16>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  }
  void launch(cudaStream_t s) const {
    if (smem > 48 * 1024 && !armed) {
      attribute = cudaFuncSetAttribute(net_kernel<Base, Mode, Relu, Act16>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
      armed = true;
    }
    net_kernel<Base, Mode, Relu, Act16><<<grid, block, smem, s>>>(params, kp);
  }
};

}  // namespace ajp::gpu::cuda::span_fp8_kernels
