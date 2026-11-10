// DistilDRBA (kernels/spec/distil_drba.h) : son graphe ONNX, exporté par
// tools/drba/export_distil_drba.py et rangé dans un dossier des modèles RIFE, puis sa
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
  // Le graphe préféré manquait : c'est l'autre (A09-21). À dire au journal, le repli joue dans les deux sens.
  bool fallback = false;
};

// Le graphe du premier de ces dossiers qui l'a : celui du flot à l'échelle 0,5 d'abord si prefer_half,
// l'autre d'abord sinon ; l'un sert de repli à l'autre.
std::optional<DrbaModel> find_distil_drba(std::span<const std::filesystem::path> dirs,
                                          bool prefer_half = false);

// Le flot à l'échelle 0,5 pour cette hauteur d'estimation : à partir de 2160 lignes, le conseil de l'auteur
// de DistilDRBA. Le seuil était à 1440 lignes sans mesure (A09-24). Mesuré au banc de qualité
// (docs/banc-qualite.md, « DistilDRBA, demi-flot ») : le demi-flot fait quatre fois moins d'opérations et
// 34 à 41 % de temps en moins sur processeur, pour 3,8 à 4,4 dB de moins sur un mouvement simple et autant
// que le flot entier sur un extrait d'action. À 1440 lignes le flot entier reste léger : la perte n'y est pas
// justifiée.
inline constexpr int kDrbaHalfFlowHeight = 2160;
bool drba_half_flow(int work_height);

// La variante « mouvement » de model (prepare::with_drba_motion_outputs) : flot et masque au lieu de l'image,
// pour le warp après SPAN. Dérivée (si elle manque) dans dir. Refus : ceux de la dérivation.
std::expected<std::filesystem::path, Refusal> drba_motion_model(const std::filesystem::path& model,
                                                                const std::filesystem::path& dir);

// Le graphe aux dimensions « H » et « W » fixées pour ces plans remplis, dérivé (s'il manque) dans
// dir. Refus : ceux de prepare::with_fixed_dims et de derived_model.
std::expected<std::filesystem::path, Refusal> static_distil_drba(const std::filesystem::path& model,
                                                                 int padded_width, int padded_height,
                                                                 const std::filesystem::path& dir);

}  // namespace ajp::app::play
