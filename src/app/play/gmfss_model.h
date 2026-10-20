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

struct GmfssFiles {
  std::string_view pair;
  std::string_view fusion;
};
constexpr GmfssFiles gmfss_files(GmfssVariant variant) {
  return variant == GmfssVariant::Union ? GmfssFiles{kGmfssUnionPairFile, kGmfssUnionFusionFile}
                                        : GmfssFiles{kGmfssPairFile, kGmfssFusionFile};
}

struct GmfssModels {
  std::filesystem::path pair;
  std::filesystem::path fusion;
};

// Les deux graphes de variant du premier de ces dossiers qui les a tous les deux.
std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs,
                                      GmfssVariant variant = GmfssVariant::Base);

// Les dimensions nommées des deux graphes pour des plans remplis de padded_width × padded_height
// (multiples de 64) : « H », « W », puis « H/2 » à « W/8 ».
std::vector<prepare::NamedDim> gmfss_dims(int padded_width, int padded_height);

// Les deux graphes aux tailles fixées pour ces plans remplis, dérivés (s'ils manquent) dans dir.
// Refus : ceux de prepare::with_fixed_dims et de derived_model.
std::expected<GmfssModels, Refusal> static_gmfss(const GmfssModels& models, int padded_width,
                                                 int padded_height, const std::filesystem::path& dir);

}  // namespace ajp::app::play
