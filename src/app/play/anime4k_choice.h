// Anime4K dans la liste des modèles (Vidéo > Modèle) : deux entrées à côté des modèles ONNX de SPAN, qui ne
// demandent aucun fichier ni moteur de réseau. Le réglage `model` de play.toml les nomme « anime4k-fast » et
// « anime4k-hq » ; un lecteur plus ancien, qui ne les connaît pas, les lit comme un modèle hors catalogue et
// reprend le modèle par défaut. `--model=anime4k-fast` les choisit aussi pour une lecture.
//
// Anime4K agrandit comme SPAN (render/anime4k.h) : la même hauteur de travail, la même réduction, le même ×2,
// mais dans la présentation (libplacebo) au lieu du thread gpu ; la chaîne joue sans SPAN et l'écran pose
// les shaders sur l'image de travail. Le mode (A, B ou C) suit la source et la réduction.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>

#include "render/anime4k.h"

namespace ajp::app::play {

inline constexpr std::string_view kAnime4kFast = "anime4k-fast";   // réseaux M
inline constexpr std::string_view kAnime4kQuality = "anime4k-hq";  // réseaux VL
inline constexpr std::array<std::string_view, 2> kAnime4kModels{kAnime4kFast, kAnime4kQuality};

// La taille des réseaux qu'un nom de modèle désigne ; nullopt : un modèle ONNX (ou vide, le défaut).
std::optional<render::Anime4kSize> anime4k_size(std::string_view model);
// « Anime4K Fast », « Anime4K HQ » : le libellé du menu (model_label).
std::string anime4k_label(render::Anime4kSize size);

// Le choix d'Anime4K pour une lecture : model en est un, et l'agrandissement tourne (span_runs : sinon la
// source est gardée telle quelle, sans agrandissement, comme SPAN). source_height et work_height : celles de
// la chaîne (work_height, la hauteur de l'image de travail après réduction).
std::optional<render::Anime4kChoice> anime4k_choice(std::string_view model, bool upscales, int source_height,
                                                    int work_height);

}  // namespace ajp::app::play
