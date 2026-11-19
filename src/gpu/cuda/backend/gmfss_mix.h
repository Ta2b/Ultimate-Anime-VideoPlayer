// Les options des morceaux du GMFSS natif et le problème de leur mélange (docs/gmfss-natif.md, « Essai par
// morceaux »), sur le choix commun de native_mix.h. Les morceaux : les trois lignes et la queue du GridNet
// de la fusion ; le FeatureNet et le MetricNet de la paire. TensorRT n'a pas de moteur par morceau (la
// fusion et la paire sont chacune un seul moteur) : il n'est pas une option d'un morceau, il garde la fusion
// ou la paire entière quand aucun mélange ne tient le seuil ou ne le bat. La base de la mesure des effets est
// donc le natif f32 partout, le calcul le plus juste. En-tête sans CUDA : les tests sur hôte le lisent.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "gpu/cuda/backend/native_mix.h"
#include "gpu/cuda/backend/native_sums.h"

namespace kirari::gpu::cuda::gmfss_mix {

// Les crans d'un morceau, du plus rapide au plus juste : sommes f16 ; f16 et tête f32 ; mixtes ; f32. La
// tête : les couches qui écrivent une sortie du réseau (l'image de la fusion, les caractéristiques, la
// métrique) ; « f16 et tête f32 » n'est proposé qu'à un morceau qui en a une.
inline constexpr std::array<RifeNativePrecision, 4> kLadder{{
    {.body = RifeNativeSums::kF16, .head = RifeNativeSums::kF16},
    {.body = RifeNativeSums::kF16, .head = RifeNativeSums::kF32},
    {.body = RifeNativeSums::kMixed, .head = RifeNativeSums::kMixed},
    {.body = RifeNativeSums::kF32, .head = RifeNativeSums::kF32},
}};
inline constexpr std::array<const char*, 4> kLadderNames{"f16", "f16 et tête f32", "mixtes", "f32"};
inline constexpr int kF16 = 0;
inline constexpr int kHeadF32 = 1;
inline constexpr int kMixed = 2;
inline constexpr int kF32 = 3;

// Le natif n'est gardé que s'il bat TensorRT d'au moins kMinGain sur la mesure du temps : en deçà, l'écart
// tient de la mesure.
inline constexpr double kMinGain = 0.02;
constexpr bool beats(double trt_ms, double native_ms) {
  return native_ms > 0.0 && native_ms <= trt_ms * (1.0 - kMinGain);
}

// Les options de chaque morceau (heads : a-t-il une tête ?) : chaque cran, « f16 et tête f32 » seulement
// avec une tête.
std::vector<std::vector<int>> candidates(const std::vector<bool>& heads);
// Le même cran pour chaque morceau.
native_mix::Options uniform(std::size_t pieces, int rung);
// Les sommes de chaque morceau pour des options.
std::vector<RifeNativePrecision> precisions(const native_mix::Options& options);
// Le problème du mélange : times[p][o], le temps seul de chaque option de candidates(heads) (les autres ne
// sont pas lues) ; la base au f32 ; les crans f16 et mixtes de tout le réseau mesurés tels quels (les
// variantes de l'ancien essai : le mélange ne peut pas faire moins bien qu'elles) ; peak et floor, le pic
// de chaque sortie de TensorRT et l'erreur du natif f32 contre elle.
native_mix::Problem problem(native_mix::Grid times, const std::vector<bool>& heads, std::vector<double> peak,
                            std::vector<double> floor, double min_db);
// « ligne 0 f16, ligne 1 mixtes, … » : le nom de chaque morceau et son cran.
std::string mix_name(const std::vector<std::string>& pieces, const native_mix::Options& options);

}  // namespace kirari::gpu::cuda::gmfss_mix
