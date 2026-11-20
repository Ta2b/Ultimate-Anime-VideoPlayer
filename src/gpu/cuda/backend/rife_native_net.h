// Le RIFE natif (--rife-native, docs/rife-natif.md) : l'encodeur et les convolutions des blocs de RIFE
// fusionné par les noyaux de rife_native_kernels.cuh, à la place des six moteurs TensorRT. Les formats sont
// ceux des moteurs en E/S HWC8 : l'encodeur lit les trois plans d'une image et écrit ses quatre plans
// d'encodage, un bloc lit X_k rangé par pixel (pas 16 ou 32) et écrit Y_k par pixel au pas 56. La colle, les
// warps et la fin restent ceux de rife_fused_kernels.h : RifeFusedEngine appelle ce réseau là où il appelait
// TensorRT. La colle du dernier bloc (pleine résolution) et de l'avant-dernier (moitié) fait exception quand
// la première couche du bloc sait la refaire à la volée (rife_native_glue.cu) et que l'essai garde cette
// variante (use_glue). En-tête sans std::expected : il est lu par nvcc en C++20.
#pragma once

#include <array>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "gpu/cuda/backend/native_sums.h"
#include "gpu/cuda/backend/rife_fused_kernels.h"

namespace kirari::gpu::cuda {

// Une couche, telle que prepare::RifeNativeLayer la lit (poids [cout][kernel²][cin] en flottants).
struct RifeNativeConv {
  bool transposed = false;
  int cin = 0;
  int cout = 0;
  int stride = 1;
  std::vector<float> weights;
  std::vector<float> bias;
  std::vector<float> slope;  // vide : pas d'activation
  std::vector<float> scale;  // vide : pas de résiduel
};
using RifeNativePiece = std::vector<RifeNativeConv>;

// La forme qu'un réglage a retenue pour une couche (rife_native_plan.h) : un réseau créé après un autre de
// même taille reprend celle des couches de mêmes sommes sans les chronométrer de nouveau.
struct RifeNativeTuned {
  int piece = 0;
  int layer = 0;
  RifeNativeSums sums = RifeNativeSums::kF16;
  int kc = 16;
  int ct = 16;
  int rows = 2;
  int ksplit = 1;
  bool ahead = false;  // poids une prise à l'avance (rife_native_kernels.cuh, accumulate)
  int bands = 0;       // bandes par bloc en flux ; 0 pour les tuiles
  bool compact_encoder = false;
};

class RifeNativeNet {
 public:
  static constexpr std::array<int, 4> kIfnetScales{8, 4, 2, 1};

  virtual ~RifeNativeNet() = default;
  // pieces : l'encodeur puis les blocs ; reduce : d_k de chaque bloc ; height × width : la taille remplie.
  // Chaque couche prend la forme (tranche, tuile, lignes par warp, parts de K, poids à l'avance ou flux) la
  // plus rapide sur cette carte, chronométrée ici sur la voie stream (comme AnimeJaNai), ou celle que tuned
  // garde déjà ; tuned reçoit les nouvelles. Rien et error rempli si une couche sort de ce que les noyaux
  // font, si les poids ne s'envoient pas, ou sur une erreur CUDA du réglage.
  static std::unique_ptr<RifeNativeNet> create(const std::vector<RifeNativePiece>& pieces,
                                               const std::vector<int>& reduce, int height, int width,
                                               RifeNativePrecision precision, cudaStream_t stream,
                                               std::vector<RifeNativeTuned>& tuned, std::string& error);
  // La transposée de l'encodeur Lite, depuis les seize canaux par pixel du préfixe TensorRT.
  static std::unique_ptr<RifeNativeNet> create_encode_tail(const RifeNativeConv& tail, int height, int width,
                                                           cudaStream_t stream, std::string& error);
  // L'IFNet de GMFSS utilise les mêmes convolutions, sans encodeur ni colle RIFE.
  static std::unique_ptr<RifeNativeNet> create_ifnet(const std::array<RifeNativePiece, 4>& pieces, int height,
                                                     int width, cudaStream_t stream, std::string& error);
  // La zone de travail (activations entre les couches) : octets, et son adresse pour les appels suivants.
  virtual std::int64_t workspace_bytes() const = 0;
  // L'encodeur sur trois plans vers quatre canaux, en plans ou groupés par pixel.
  virtual cudaError_t encode(cudaStream_t stream, const void* image, void* encoded, void* workspace,
                             bool hwc4 = false) const = 0;
  // Le bloc k : x (X_k par pixel), y (Y_k par pixel, pas 56). glue, pour un bloc dont ce réseau calcule
  // l'entrée à la volée (glues) : sa première couche lit la colle au lieu de x et n'écrit dans glue->x (X_k)
  // que le flot accumulé, que lit la fin ; nul sinon.
  // finish, pour le dernier bloc d'un réseau qui sait faire cette fin (can_finish) : sa dernière couche fait
  // fused_finish dans son épilogue, au bit près, et n'écrit pas y ; nul sinon.
  virtual cudaError_t block(cudaStream_t stream, int k, const void* x, void* y, void* workspace,
                            const FusedGlueParams* glue, const FusedFinishParams* finish) const = 0;
  virtual bool can_finish(int k, const FusedFinishParams& finish) const = 0;
  // Ce réseau calcule-t-il l'entrée du bloc k à la volée (le dernier bloc en pleine résolution,
  // l'avant-dernier à moitié) ? can_glue : il en a le noyau ; glues : il s'en sert (use_glue, vrai au départ
  // pour chaque bloc). Sans lui, le bloc lit le X_k que la colle à part a écrit, par la forme réglée de sa
  // première couche ; l'essai garde la plus rapide, bloc par bloc.
  virtual bool can_glue(int k) const = 0;
  virtual bool glues(int k) const = 0;
  virtual void use_glue(int k, bool on) = 0;
  // Choisit la forme étroite de la colle D1, si le réseau la propose.
  virtual bool use_staged_glue(int k, bool on) = 0;
  // Le profil d'un morceau (0 : l'encodeur, 1 + k : le bloc k) : ses couches comme encode ou block (glue
  // comme pour block), marks[0] enregistré avant la première et marks[i + 1] après la couche i
  // (cudaEventRecordExternal : la capture d'un graphe les garde). marks.size() doit valoir layers(piece) + 1.
  virtual int layers(int piece) const = 0;
  virtual cudaError_t profile(cudaStream_t stream, int piece, const void* in, void* out, void* workspace,
                              std::span<const cudaEvent_t> marks, const FusedGlueParams* glue) const = 0;
  // La couche i du morceau pour le journal : sa sorte, ses canaux, la forme retenue au réglage.
  virtual std::string layer_label(int piece, int i) const = 0;
};

}  // namespace kirari::gpu::cuda
