// Le FeatureNet et le MetricNet de la paire de GMFSS Fortuna Union tels que les noyaux natifs les font
// (--gmfss-native, docs/gmfss-natif.md) : deux petits réseaux aux types du GridNet (gmfss_grid.h), lus dans
// l'ONNX de la paire par prepare/gmfss_pair_native.h. GMFlow et ce qui prépare l'entrée du MetricNet
// (backwarp, distances, cohérence des flots) restent un moteur TensorRT (prepare/gmfss_pair_split.h). En-tête
// sans std::expected : nvcc le lit en C++20.
#pragma once

#include "prepare/gmfss_grid.h"

namespace kirari::prepare {

// FeatureNet, pour une image : le tenseur 0 est l'image, trois plans FP16 au double de la taille de la fusion
// (kGmfssFullLevel), lue en 16 canaux dont 13 nuls ; 1 à 3 sont ses sorties, feature_1 (64 canaux à la taille
// de la fusion), source_2 (128, à la moitié) et source_3 (192, au quart) ; les suivants, les sorties des
// premières convolutions des blocs.
inline constexpr int kGmfssFeatureImage = 0;
inline constexpr int kGmfssFeatureLevels = 3;  // les sorties : tenseurs 1 à 3, niveaux 0 à 2

// MetricNet : le tenseur 0 est l'entrée de metric_in, ses 14 plans (half0, half1, -métrique0, -métrique1, les
// deux flots normés, les deux occultations) rangés en 16 canaux par pixel ; le 1, la sortie de metric_out
// avant tanh × 10, 2 canaux utiles sur 8 (la métrique de chaque côté) ; les suivants, les sommes du réseau.
inline constexpr int kGmfssMetricInput = 0;
inline constexpr int kGmfssMetricOutput = 1;
inline constexpr int kGmfssMetricPlanes = 14;
inline constexpr float kGmfssMetricScale = 10.F;  // la sortie : tanh(x) × 10

struct GmfssPairNets {
  GmfssGrid features;
  GmfssGrid metric;
};

}  // namespace kirari::prepare
