// Ce que partagent le réseau du RIFE natif (rife_native_net.cu), ses lancements (rife_native_pick.cu) et le
// rangement de ses poids (rife_native_weights.cu) : les façons de lancer une couche, la forme qu'un réglage
// au chargement choisit pour chacune, et ses poids en FP16. Les convolutions du GMFSS natif
// (gmfss_conv_plan.cu) prennent les mêmes noyaux et le même rangement. Lu par nvcc seulement (C++20).
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <span>
#include <vector>

#include "gpu/cuda/backend/rife_fused_kernels.h"
#include "gpu/cuda/backend/rife_native_kernels.cuh"
#include "gpu/cuda/backend/rife_native_net.h"

namespace kirari::gpu::cuda::rife_native {

// Les façons de lancer une couche.
enum class Kind : std::uint8_t {
  kSame,     // 3 × 3, pas 1
  kHalf,     // 3 × 3, pas 2
  kImage,    // 3 × 3, pas 2, depuis les trois plans d'une image (première couche de l'encodeur)
  kPhase,    // transposée 4 × 4 de pas 2, une phase par blockIdx.z, sortie par pixel
  kPlanes4,  // transposée de l'encodeur, 3 × 3 aux 16 sorties phase · 4 + c, sortie en quatre plans
  kShuffle,  // GMFSS natif (gmfss_conv_plan.cuh) : 3 × 3 de pas 1 suivie du PixelShuffle(2), par phases
};

// La forme d'une couche : tranche des canaux d'entrée (kc), tuile des canaux de sortie (ct), lignes par
// warp (rows : la tuile de pixels fait 4 · rows lignes sur 32 colonnes), parts de K (ksplit), poids de toute
// la prise suivante chargés pendant la prise courante (ahead), au prix d'un second jeu de registres. bands :
// la forme en flux (conv_stream_kernels.cuh), bands bandes au plus par bloc ; 0, la tuile de conv_kernel.
struct Config {
  int kc = 16;
  int ct = 16;
  int rows = 2;
  int ksplit = 1;
  bool ahead = false;
  int bands = 0;
};

constexpr int kernel_size(Kind kind) {
  return kind == Kind::kPhase ? 2 : 3;
}
constexpr int stride_of(Kind kind) {
  return kind == Kind::kHalf || kind == Kind::kImage ? 2 : 1;
}

using Launch = void (*)(dim3 grid, int smem, cudaStream_t stream, const Params& p);

// Ce qui borne les formes d'une couche : canaux d'entrée et de sortie (complétés à 16), grille de sortie,
// phases (4 pour une transposée de bloc).
struct Extent {
  int cin = 16;
  int cout = 16;
  int oh = 0;
  int ow = 0;
  int phases = 1;
};

// Le noyau de cette forme, ou nul s'il n'existe pas : tuile de 64 colonnes hors sommes f16 (ses 128
// registres d'accumulateurs f32 débordent), une ligne par warp en f32 ou en pas 2 (non construits), poids à
// l'avance seulement en pas 1 (kSame), tranche de 32 ou 64, hors f32 : les ResConv des blocs, où les poids
// d'une couche large viennent du L2.
Launch pick_for(RifeNativeSums sums, Kind kind, int kc, int ct, int rows, bool ahead = false);
// Les formes à essayer au réglage : toute tranche, tuile, hauteur et avance qu'un noyau fait, chacune sans
// découpe de K, et découpée assez pour deux vagues sur sms multiprocesseurs quand sa grille en fait moins
// (les blocs grossiers, 30 × 17 pixels au 1/32 de 1080p), au plus une part par prise et tranche. Ni
// l'encodeur ni une couche lue dans l'image ne découpent K : pleine taille, leur grille remplit déjà la
// carte.
std::vector<Config> configs(RifeNativeSums sums, Kind kind, const Extent& extent, int sms);
// La forme en flux (Config::bands > 0) : son noyau, ou nul s'il n'existe pas (3 × 3 de pas 1, tranche de 64
// et une ligne par warp ; tuile de 64 en sommes f16, de 32 en f16 ou mixtes) ; ses formes à essayer, quand
// toute l'entrée tient en une tranche (bandes par bloc : conv_stream::band_counts). Le GMFSS natif les ajoute
// à celles de configs ; le RIFE natif ne les prend pas.
Launch pick_stream(RifeNativeSums sums, Kind kind, const Config& cfg);
std::vector<Config> stream_configs(RifeNativeSums sums, Kind kind, const Extent& extent, int sms);
const char* sums_name(RifeNativeSums sums);
// La fin d'une couche à K découpé (reduce_kernel), width : canaux de la grille z.
void launch_reduce(bool phase, int width, cudaStream_t stream, const Params& p);

// La première couche du dernier bloc, quand il est en pleine résolution (d = 1) : une convolution 3 × 3 de
// pas 2 sur les 28 canaux de X (32 rangés), que rife_native_glue.cu calcule à la volée au lieu de les lire
// (la colle, rife_glue.cuh). La tuile d'entrée d'une ligne par warp fait 65 × 9 pixels ; le relèvement de
// chaque source (le Y d'un bloc précédent, d'échelle scale) y prend au plus glue_capacity(scale) pixels de Y
// : sur len + 1 points pleine résolution, le Z agrandi couvre au plus ⌈len / scale⌉ + 2 points (coarse_span),
// le Y la moitié arrondie au-dessus, plus un.
__host__ __device__ constexpr int glue_span(int len, int scale) {
  const int z = (len + scale - 1) / scale + 1;
  return (z + 1) / 2 + 1;
}
__host__ __device__ constexpr int glue_capacity(int scale) {
  return glue_span(64, scale) * glue_span(8, scale);
}
using GlueLaunch = void (*)(dim3 grid, int smem, cudaStream_t stream, const Params& p,
                            const FusedGlueParams& g);
// Le noyau de cette première couche : flows blocs d'avant (3 ou 4), une tuile de ct sorties (16 ou 32, la
// couche entière) ; nul sinon. Sa grille : (⌈ow / 32⌉, ⌈oh / 4⌉, 1).
GlueLaunch pick_glue(RifeNativeSums sums, int flows, int ct);
// Ses octets de mémoire partagée, pour des sources d'échelles scales (la dernière porte 52 canaux).
int glue_smem(std::span<const int> scales);

// La dernière couche du dernier bloc (la transposée), fin comprise (rife_native_finish.cu) : sa première
// tuile de 32 canaux seulement (le flot du bloc et le masque y sont), puis fused_finish dans son épilogue au
// lieu d'écrire Y. Le noyau de la tranche kc et de rows lignes par warp, pour des sommes de tête (f16 ou f32,
// une ligne par warp en f16 seulement) ; nul sinon. Sa grille : (⌈ow / 32⌉, ⌈oh / (4 · rows)⌉, 4) ; ses
// paramètres ceux de la couche, poids rangés pour une tuile de 32, sans découpe de K.
using FinishLaunch = void (*)(dim3 grid, int smem, cudaStream_t stream, const Params& p,
                              const FusedFinishParams& f);
FinishLaunch pick_finish(RifeNativeSums sums, int kc, int rows);

// Les poids d'une couche en FP16, [phase][cout][prise][cin] (cout et cin complétés à 16 par des zéros), le
// bêta d'un résiduel replié (w · s, b · s), et ses biais [phase][cout]. Une seule phase hors transposée de
// bloc ; la transposée de l'encodeur en 16 sorties phase · 4 + c sur 9 prises.
struct Dense {
  int phases = 1;
  int cout = 0;
  int taps = 9;
  int cin = 0;
  std::vector<__half> w;
  std::vector<float> bias;
};
Dense dense_weights(const RifeNativeConv& conv, Kind kind, int cin);
// Les poids rangés dans l'ordre des fragments B du mma pour une tuile de ct sorties : un paquet par phase et
// par tuile.
std::vector<std::uint32_t> packed_weights(const Dense& dense, int ct);

}  // namespace kirari::gpu::cuda::rife_native
