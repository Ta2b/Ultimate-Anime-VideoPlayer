// Les noyaux de TemporalFix fusionné (kernels/spec/tfix_fused.h), côté hôte. Tous lisent et écrivent du FP16
// rangé en plans, calculent en FP32. Les images de la fenêtre passent par une table d'adresses en mémoire de
// l'appareil (TfixTable), posée par le moteur avant chaque fenêtre : le graphe CUDA capturé ne dépend ainsi
// pas des tampons que la fenêtre choisit, qui tournent d'une image à l'autre.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "kernels/spec/tfix_fused.h"

namespace kirari::gpu::cuda {

// Les adresses d'une fenêtre : les sept images à la taille de travail (rgb) et leurs tampons gardés
// (packed, kernels::TfixPacked), dans l'ordre de la fenêtre ; la sortie à la taille de travail.
// Tableaux C : lus tels quels par les noyaux.
struct TfixTable {
  const void* rgb[kernels::kTfixFrames] = {};     // NOLINT(modernize-avoid-c-arrays)
  const void* packed[kernels::kTfixFrames] = {};  // NOLINT(modernize-avoid-c-arrays)
  void* out = nullptr;
};
static_assert(sizeof(TfixTable) <= 256);

// Ce que tous les noyaux de la fenêtre savent : la taille du réseau, la demi-taille, la disposition des
// tampons gardés et de la zone.
// Les décalages de kernels::TfixPacked et kernels::TfixLayout recopiés en tableaux C (le code de l'appareil
// n'appelle pas les fonctions constexpr de l'hôte) : tfix_shape les remplit.
struct TfixShape {
  int height = 0;  // le réseau (la moitié de la taille de travail en demi-taille)
  int width = 0;
  bool half = false;
  int lh[kernels::kTfixLevels] = {};  // NOLINT(modernize-avoid-c-arrays) les niveaux (tfix_level)
  int lw[kernels::kTfixLevels] = {};  // NOLINT(modernize-avoid-c-arrays)
  int yh[kernels::kTfixLevels] = {};  // NOLINT(modernize-avoid-c-arrays) Y_b (tfix_block_y)
  int yw[kernels::kTfixLevels] = {};  // NOLINT(modernize-avoid-c-arrays)
  std::size_t image = 0;              // TfixPacked
  std::size_t traits = 0;
  std::size_t pyramid[kernels::kTfixLevels] = {};  // NOLINT(modernize-avoid-c-arrays)
  std::size_t x[kernels::kTfixLevels] = {};        // NOLINT(modernize-avoid-c-arrays) TfixLayout
  std::size_t y[kernels::kTfixLevels] = {};        // NOLINT(modernize-avoid-c-arrays)
  std::size_t conf_in = 0;
  std::size_t mask_half = 0;
  std::size_t conf_out = 0;
  std::size_t net_out = 0;
  // Le rangement de X_b, Y_b, de l'entrée et de la sortie de la tête : en plans (TensorRT), ou par pixel au
  // pas *_ld (noyaux natifs, kernels::tfix_in_ld et tfix_out_ld). En plans, le pas est le nombre de canaux.
  bool native = false;
  int xld[kernels::kTfixLevels] = {};  // NOLINT(modernize-avoid-c-arrays)
  int yld = 0;
  int conf_ld = 0;
  int conf_out_ld = 0;
  void* zone = nullptr;              // le début des tampons de la fenêtre (TfixLayout)
  const TfixTable* table = nullptr;  // dans la zone (TfixLayout::table)
};

TfixShape tfix_shape(int height, int width, bool half, void* zone, bool native = false);

// L'encodeur fusionné, avant le moteur de l'encodeur (demi-taille) : l'image de travail (3 plans,
// 2H × 2W) moyennée 2 × 2 dans packed (TfixPacked::image).
cudaError_t launch_tfix_pool(const void* rgb, void* packed, int height, int width, cudaStream_t stream);

// L'encodeur fusionné, après le moteur de l'encodeur : les trois niveaux de pyramide de l'image (image, ou
// TfixPacked::image en demi-taille) et de ses traits (TfixPacked::traits), dans packed.
cudaError_t launch_tfix_pyramid(const void* image, void* packed, int height, int width, bool half,
                                cudaStream_t stream);

// X0, l'entrée du bloc 0.
cudaError_t launch_tfix_first(const TfixShape& shape, cudaStream_t stream);
// X_level (1 ou 2) depuis Y_{level-1} et X_{level-1} : flot agrandi, appuis déplacés, Concat.
cudaError_t launch_tfix_glue(const TfixShape& shape, int level, cudaStream_t stream);
// L'entrée de la tête de confiance et le masque moyenné, depuis Y2 et X2 : flot à pleine taille, appuis
// déplacés, leur flou réduit à demi-résolution.
cudaError_t launch_tfix_conf(const TfixShape& shape, cudaStream_t stream);
// Le mélange final depuis la sortie de la tête : la sortie (table->out), ou en demi-taille la sortie du
// réseau puis son report sur l'image entière.
cudaError_t launch_tfix_blend(const TfixShape& shape, cudaStream_t stream);

}  // namespace kirari::gpu::cuda
