// Les modèles de la stabilisation temporelle (temporal_fix.h) : les ONNX de pifroggi/vs_temporalfix (Apache
// 2.0), livrés dans models/temporalfix, puis leur variante de lecture (derived_model.h) : l'entrée des sept
// images empilées (1 × 21 × H × W) coupée en sept entrées 1 × 3 × H × W (prepare/onnx_frames.h), que l'étage
// gpu fait tourner sans copie, et, par défaut, le réseau à demi-taille. H et W restent libres : figées, les
// grilles du réseau deviennent 6,5 Go de constantes dans le moteur TensorRT (docs/temporalfix.md).
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>

#include "app/play/temporal_fix.h"
#include "core/types.h"
#include "prepare/onnx_cache.h"

namespace kirari::app::play {

// Le nom de l'entrée des modèles livrés, et le dossier où ils sont rangés sous les modèles du lecteur.
inline constexpr std::string_view kTemporalFixInput = "input";
inline constexpr std::string_view kTemporalFixDir = "temporalfix";

// Le modèle de cette force dans le premier de ces dossiers qui l'a ; rien éteint, ou absent partout.
std::optional<std::filesystem::path> find_temporal_fix(TemporalFix fix,
                                                       std::span<const std::filesystem::path> dirs);

// Le nom des entrées de traits de la fenêtre (prepare/onnx_cache.h).
inline constexpr std::string_view kTemporalFixFeatures = "features";

// La découpe de l'encodeur (prepare/onnx_cache.h) : chaque image n'est encodée qu'une fois, ses traits
// gardés à côté d'elle dans la fenêtre (temporal_window.h).
prepare::EncoderCache temporal_fix_cache(bool half);

// La fenêtre à construire : entrée coupée en kTemporalFixFrames images, suivies de leurs kTemporalFixFrames
// traits gardés (sans eux si whole : le graphe en un morceau, --no-tfix-cache), à demi-taille si half,
// dans dir. Un fichier par modèle et par variante, quelle que soit la taille de travail. Refus : ceux de
// derived_model et des découpes.
std::expected<std::filesystem::path, Refusal> temporal_fix_graph(const std::filesystem::path& model,
                                                                 bool half, const std::filesystem::path& dir,
                                                                 bool whole = false);
// L'encodeur à construire : une image 1 × 3 × H × W → ses traits (prepare::feature_channels canaux).
std::expected<std::filesystem::path, Refusal> temporal_fix_encoder(const std::filesystem::path& model,
                                                                   bool half,
                                                                   const std::filesystem::path& dir);

// La taille de travail que la stabilisation prend : paire, et multiple de 4 à demi-taille (le réseau lit
// alors une image paire). Faux : la lecture se fait sans elle.
constexpr bool temporal_fix_fits(int width, int height, bool half) {
  const int step = half ? 4 : 2;
  return width > 0 && height > 0 && width % step == 0 && height % step == 0;
}

}  // namespace kirari::app::play
