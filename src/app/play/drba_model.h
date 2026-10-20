// DistilDRBA (kernels/spec/distil_drba.h) : son graphe ONNX, exporté par
// tools/drba/export_distil_drba.py et rangé dans un dossier des modèles RIFE (comme GMFSS), puis sa
// variante à la taille fixée pour la taille remplie de la lecture (derived_model.h).
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

#include "core/types.h"

namespace ajp::app::play {

inline constexpr std::string_view kDistilDrbaFile = "distil_drba_v2_lite.onnx";

// Le graphe du premier de ces dossiers qui l'a.
std::optional<std::filesystem::path> find_distil_drba(std::span<const std::filesystem::path> dirs);

// Le graphe aux dimensions « H » et « W » fixées pour ces plans remplis, dérivé (s'il manque) dans
// dir. Refus : ceux de prepare::with_fixed_dims et de derived_model.
std::expected<std::filesystem::path, Refusal> static_distil_drba(const std::filesystem::path& model,
                                                                 int padded_width, int padded_height,
                                                                 const std::filesystem::path& dir);

}  // namespace ajp::app::play
