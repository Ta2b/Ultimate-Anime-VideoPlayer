// GMFSS Fortuna Union (kernels/spec/gmfss.h) : ses deux graphes ONNX, exportés par
// tools/gmfss/export_gmfss.py (poids de la roue vsgmfss_fortuna 1.0.0, MIT ; flot entier) et rangés dans un
// dossier des modèles RIFE, puis leurs variantes aux tailles fixées pour les plans remplis de la lecture
// (derived_model.h), que TensorRT construit sans profil.
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "prepare/onnx_dims.h"

namespace ajp::app::play {

inline constexpr std::string_view kGmfssPairFile = "gmfss_fortuna_union_pair.onnx";
inline constexpr std::string_view kGmfssFusionFile = "gmfss_fortuna_union_fusion.onnx";

struct GmfssModels {
  std::filesystem::path pair;
  std::filesystem::path fusion;
};

// Les deux graphes du premier de ces dossiers qui les a tous les deux.
std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs);

// Les dimensions nommées des deux graphes pour des plans remplis de padded_width × padded_height (multiples
// de 64) : « H », « W », puis « H/2 » à « W/8 ».
std::vector<prepare::NamedDim> gmfss_dims(int padded_width, int padded_height);

// Les deux graphes aux tailles fixées pour ces plans remplis, dérivés (s'ils manquent) dans dir ; les tailles
// d'une autre vidéo s'en vont. Refus : ceux de prepare::with_fixed_dims et de derived_model.
std::expected<GmfssModels, Refusal> static_gmfss(const GmfssModels& models, int padded_width,
                                                 int padded_height, const std::filesystem::path& dir);

}  // namespace ajp::app::play
