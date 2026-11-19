// GMFSS Fortuna Union (kernels/spec/gmfss.h) : ses deux graphes ONNX, exportés par
// tools/gmfss/export_gmfss.py (poids de la roue vsgmfss_fortuna 1.0.0, MIT ; flot entier) et rangés dans le
// dossier de --gmfss-dir, un dossier des modèles RIFE ou son voisin « gmfss », puis leurs variantes aux
// tailles fixées pour les plans remplis de la lecture (derived_model.h), que TensorRT construit sans profil.
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace kirari::app::play {

inline constexpr std::string_view kGmfssPairFile = "gmfss_fortuna_union_pair.onnx";
inline constexpr std::string_view kGmfssFusionFile = "gmfss_fortuna_union_fusion.onnx";
// La paire au demi-flot (export_gmfss.py --scale 0.5) : mêmes entrées et sorties, GMFlow sur les images
// réduites de moitié ; la même fusion la suit (PlaySettings::gmfss_half_flow).
inline constexpr std::string_view kGmfssHalfPairFile = "gmfss_fortuna_union_s05_pair.onnx";

struct GmfssModels {
  std::filesystem::path pair;
  std::filesystem::path fusion;
  std::filesystem::path half_pair;  // vide sans le graphe au demi-flot à côté des deux autres
};

// Où chercher les graphes : chaque dossier, puis son voisin « gmfss » (~/kirari/models/gmfss à côté de
// ~/kirari/models/rife, le --out que propose tools/gmfss/export_gmfss.py), sans doublon.
std::vector<std::filesystem::path> gmfss_places(std::span<const std::filesystem::path> dirs);

// Les deux graphes du premier dossier de gmfss_places(dirs) qui les a tous les deux, et la paire au demi-flot
// de ce dossier si elle y est.
std::optional<GmfssModels> find_gmfss(std::span<const std::filesystem::path> dirs);

// Les deux graphes à construire : la paire au demi-flot si half et qu'elle est là, sinon celle au flot
// entier.
GmfssModels gmfss_chosen(const GmfssModels& found, bool half);

// Les deux graphes aux tailles fixées pour ces plans remplis, dérivés (s'ils manquent) dans dir ; les tailles
// d'une autre vidéo s'en vont. Refus : ceux de prepare::with_fixed_dims et de derived_model.
std::expected<GmfssModels, Refusal> static_gmfss(const GmfssModels& models, int padded_width,
                                                 int padded_height, const std::filesystem::path& dir);

}  // namespace kirari::app::play
