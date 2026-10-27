// GMFSS Fortuna (kernels/spec/gmfss.h) : ses deux graphes ONNX, exportés par
// tools/gmfss/export_gmfss.py et rangés dans un dossier des modèles RIFE, puis leurs variantes aux
// tailles fixées pour la taille remplie de la lecture (derived_model.h), que TensorRT construit
// sans profil.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "prepare/onnx_dims.h"

namespace ajp::app::play {

// Les deux modèles de vsgmfss_fortuna : base, et union, dont la fusion mêle aussi l'image de
// RIFE (IFNet, dans le graphe « fusion », qui reçoit alors l'instant « time »).
enum class GmfssVariant : std::uint8_t { Base, Union };

inline constexpr std::string_view kGmfssPairFile = "gmfss_fortuna_base_pair.onnx";
inline constexpr std::string_view kGmfssFusionFile = "gmfss_fortuna_base_fusion.onnx";
inline constexpr std::string_view kGmfssUnionPairFile = "gmfss_fortuna_union_pair.onnx";
inline constexpr std::string_view kGmfssUnionFusionFile = "gmfss_fortuna_union_fusion.onnx";
// GMFlow à l'échelle 0,5 (export_gmfss.py --scale 0.5) : le flot à un quart de la taille de travail,
// 3,5 fois moins cher pour la paire (8,3 ms contre 29,4 à 720p sur une 4070 Ti) et la même qualité
// mesurée (21,2 dB sur 19 triplets de Re:Zero) ; plans remplis au multiple de 128. Préférés quand
// ils sont là.
inline constexpr std::string_view kGmfssHalfPairFile = "gmfss_fortuna_base_s05_pair.onnx";
inline constexpr std::string_view kGmfssHalfFusionFile = "gmfss_fortuna_base_s05_fusion.onnx";
inline constexpr std::string_view kGmfssUnionHalfPairFile = "gmfss_fortuna_union_s05_pair.onnx";
inline constexpr std::string_view kGmfssUnionHalfFusionFile = "gmfss_fortuna_union_s05_fusion.onnx";
inline constexpr int kGmfssHalfAlign = 128;

struct GmfssFiles {
  std::string_view pair;
  std::string_view fusion;
};
constexpr GmfssFiles gmfss_files(GmfssVariant variant, bool half_flow = false) {
  if (half_flow) {
    return variant == GmfssVariant::Union ? GmfssFiles{kGmfssUnionHalfPairFile, kGmfssUnionHalfFusionFile}
                                          : GmfssFiles{kGmfssHalfPairFile, kGmfssHalfFusionFile};
  }
  return variant == GmfssVariant::Union ? GmfssFiles{kGmfssUnionPairFile, kGmfssUnionFusionFile}
                                        : GmfssFiles{kGmfssPairFile, kGmfssFusionFile};
}

struct GmfssModels {
  std::filesystem::path pair;
  std::filesystem::path fusion;
  int align = 64;  // multiple des plans remplis : 128 pour le flot à l'échelle 0,5
};

// Les deux graphes de variant du premier de ces dossiers qui les a tous les deux, ceux du flot à
// l'échelle 0,5 d'abord si prefer_half (la règle de DistilDRBA : drba_half_flow), les autres
// d'abord sinon ; l'autre jeu sert de repli.
std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs,
                                      GmfssVariant variant = GmfssVariant::Base, bool prefer_half = true);

// Les dimensions nommées des deux graphes pour des plans remplis de padded_width × padded_height
// (multiples de 64) : « H », « W », puis « H/2 » à « W/8 ».
std::vector<prepare::NamedDim> gmfss_dims(int padded_width, int padded_height);

// Les deux graphes aux tailles fixées pour ces plans remplis, dérivés (s'ils manquent) dans dir.
// Refus : ceux de prepare::with_fixed_dims et de derived_model.
std::expected<GmfssModels, Refusal> static_gmfss(const GmfssModels& models, int padded_width,
                                                 int padded_height, const std::filesystem::path& dir);

}  // namespace ajp::app::play
