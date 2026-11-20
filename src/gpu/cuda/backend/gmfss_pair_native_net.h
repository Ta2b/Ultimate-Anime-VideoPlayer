// Le FeatureNet et le MetricNet natifs de GMFSS (--gmfss-native, docs/gmfss-natif.md), à la place de leur
// part du moteur TensorRT de la paire ; GMFlow et l'entrée du MetricNet restent le moteur du « flot »
// (prepare/gmfss_pair_split.h). Les entrées et les sorties sont celles de la paire, en plans FP16, pour que
// les splats et la fusion les lisent comme avant. En-tête sans std::expected : il est lu par nvcc en C++20.
//
// Les caractéristiques d'une image ne dépendent que d'elle : le moteur peut reprendre les trois sorties
// d'une image déjà passée dans features() si le lecteur a identifié la même source.
#pragma once

#include <array>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

#include "gpu/cuda/backend/gmfss_native_parts.h"
#include "prepare/gmfss_pair_grid.h"

namespace kirari::gpu::cuda {

// Les sorties du FeatureNet pour une image : feature_1 (64 plans à la taille de la fusion), source_2 (128
// plans à la moitié) et source_3 (192 au quart).
struct GmfssPairFeatures {
  std::array<void*, prepare::kGmfssFeatureLevels> levels{};
};

// Les mouvements motion{k}_{l} de la paire (3 plans : le flot, que le moteur du flot écrit, puis la métrique,
// que metric() écrit), rangés k · 3 + l - 1.
struct GmfssPairMotion {
  std::array<void*, 6> motion{};
};

// Les morceaux (GmfssNativeParts) : le FeatureNet (0), dont la tête écrit les trois sorties, et le MetricNet
// (1), dont la tête est metric_out.
class GmfssPairNativeNet : public GmfssNativeParts {
 public:
  static constexpr int kFeatureNet = 0;
  static constexpr int kMetricNet = 1;
  // height × width : la taille de la fusion (multiples de 32), les images étant au double. Une zone pour les
  // deux réseaux, qui tournent l'un après l'autre sur un même flux ; chaque convolution est réglée au banc
  // sur stream pour les trois sommes, celles posées étant f32 partout. Rien et error rempli si une
  // convolution sort de ce que les noyaux font, ou sur une erreur CUDA (mémoire, banc).
  static std::unique_ptr<GmfssPairNativeNet> create(const prepare::GmfssPairNets& nets, int height, int width,
                                                    cudaStream_t stream, std::string& error);
  // Le FeatureNet d'une image : trois plans FP16 de 2·height × 2·width (frames, ou frames + 3 plans pour la
  // seconde image).
  virtual cudaError_t features(cudaStream_t stream, const void* image,
                               const GmfssPairFeatures& out) const = 0;
  // Le MetricNet : in, les 14 plans de height × width que rend le moteur du flot ; la métrique, tanh × 10, va
  // au troisième plan des six mouvements, réduite de moitié (moyenne 2 × 2) et au quart (moyenne des quatre
  // pixels du centre de chaque 4 × 4) comme le Resize linéaire de la paire.
  virtual cudaError_t metric(cudaStream_t stream, const void* in, const GmfssPairMotion& motion) const = 0;
};

}  // namespace kirari::gpu::cuda
