// Le RIFE natif (--rife-native, docs/rife-natif.md) : l'encodeur et les convolutions des blocs de RIFE
// fusionné par les noyaux de rife_native_kernels.cuh, à la place des six moteurs TensorRT. Les formats sont
// ceux des moteurs en E/S HWC8 : l'encodeur lit les trois plans d'une image et écrit ses quatre plans
// d'encodage, un bloc lit X_k rangé par pixel (pas 16 ou 32) et écrit Y_k par pixel au pas 56. La colle, les
// warps et la fin restent ceux de rife_fused_kernels.h : RifeFusedEngine appelle ce réseau là où il appelait
// TensorRT, sauf la colle du dernier bloc en pleine résolution, que sa première couche refait à la volée
// (rife_native_glue.cu). En-tête sans std::expected : il est lu par nvcc en C++20.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <span>
#include <string>
#include <vector>

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

// Les sommes des convolutions : f16 (le débit plein des tensor cores des GeForce), mixtes (chaque prise en
// f16, les prises entre elles en f32, au débit du f16), f32 (moitié du débit sur GeForce).
enum class RifeNativeSums : std::uint8_t { kF16, kMixed, kF32 };
// Les sommes des couches d'un morceau : la dernière (la tête : le flot et le masque d'un bloc, l'encodage de
// l'encodeur) peut sommer plus finement que les autres.
struct RifeNativePrecision {
  RifeNativeSums body = RifeNativeSums::kF16;
  RifeNativeSums head = RifeNativeSums::kF16;
};

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
};

class RifeNativeNet {
 public:
  virtual ~RifeNativeNet() = default;
  // pieces : l'encodeur puis les blocs ; reduce : d_k de chaque bloc ; height × width : la taille remplie.
  // Chaque couche prend la forme (tranche, tuile, lignes par warp, parts de K, poids à l'avance) la plus
  // rapide sur cette carte, chronométrée ici sur la voie stream (comme AnimeJaNai), ou celle que tuned garde
  // déjà ; tuned reçoit les nouvelles. Rien et error rempli si une couche sort de ce que les noyaux font, si
  // les poids ne s'envoient pas, ou sur une erreur CUDA du réglage.
  static std::unique_ptr<RifeNativeNet> create(const std::vector<RifeNativePiece>& pieces,
                                               const std::vector<int>& reduce, int height, int width,
                                               RifeNativePrecision precision, cudaStream_t stream,
                                               std::vector<RifeNativeTuned>& tuned, std::string& error);
  // La zone de travail (activations entre les couches) : octets, et son adresse pour les appels suivants.
  virtual std::int64_t workspace_bytes() const = 0;
  // L'encodeur sur une image (3 plans height × width) vers encoded (4 plans).
  virtual cudaError_t encode(cudaStream_t stream, const void* image, void* encoded,
                             void* workspace) const = 0;
  // Le bloc k : x (X_k par pixel), y (Y_k par pixel, pas 56). glue, pour un bloc dont ce réseau calcule
  // l'entrée à la volée (glues) : sa première couche lit la colle au lieu de x et n'écrit dans glue->x (X_k)
  // que le flot accumulé, que lit la fin ; nul sinon.
  virtual cudaError_t block(cudaStream_t stream, int k, const void* x, void* y, void* workspace,
                            const FusedGlueParams* glue) const = 0;
  // Ce réseau calcule-t-il l'entrée du bloc k à la volée (le dernier bloc, en pleine résolution) ?
  virtual bool glues(int k) const = 0;
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
