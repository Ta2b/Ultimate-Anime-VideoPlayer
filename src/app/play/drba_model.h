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
// Le même réseau avec le flot estimé à l'échelle 0,5 (export_distil_drba.py --scale 0.5), conseillé
// par l'auteur de DistilDRBA en 4K : plans remplis à un multiple de 128.
inline constexpr std::string_view kDistilDrbaHalfFile = "distil_drba_v2_lite_s05.onnx";
inline constexpr int kDistilDrbaHalfAlign = 128;

struct DrbaModel {
  std::filesystem::path path;
  int align = 64;  // multiple des plans remplis : 128 pour le flot à l'échelle 0,5
};

// Le graphe du premier de ces dossiers qui l'a : celui du flot à l'échelle 0,5 d'abord si prefer_half,
// l'autre d'abord sinon ; l'un sert de repli à l'autre.
std::optional<DrbaModel> find_distil_drba(std::span<const std::filesystem::path> dirs,
                                          bool prefer_half = false);

// Le flot à l'échelle 0,5 pour cette hauteur de travail : à partir de 1440p (la 4K).
bool drba_half_flow(int work_height);

// Le graphe aux dimensions « H » et « W » fixées pour ces plans remplis, dérivé (s'il manque) dans
// dir. Refus : ceux de prepare::with_fixed_dims et de derived_model.
std::expected<std::filesystem::path, Refusal> static_distil_drba(const std::filesystem::path& model,
                                                                 int padded_width, int padded_height,
                                                                 const std::filesystem::path& dir);

}  // namespace ajp::app::play
