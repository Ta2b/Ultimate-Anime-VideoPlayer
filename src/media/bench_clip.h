// La séquence de l'évaluation de la machine (« Évaluer ma machine », ajp bench) : un court clip synthétique,
// écrit une fois par le lecteur lui-même, pour que l'évaluation n'ait besoin d'aucune vidéo
// (docs/plan-v0.6.md). Version 3 : des plans de deux secondes, calme puis combat rapide puis combat sombre,
// en boucle ; fond qui défile (40 px par image en combat), personnages qui traversent l'écran en une seconde
// et se masquent, lame qui tourne, hachures fines, dégradés sombres et grain sur toute l'image.
#pragma once

#include <expected>
#include <filesystem>

#include "core/types.h"

namespace ajp::media {

struct BenchClipSpec {
  int width = 1920;
  int height = 1080;
  int fps = 24;
  int seconds = 12;
};

// Écrit le clip (MKV, MPEG-2, ~30 Mb/s : un décodeur que toutes les machines ont) ; remplace le
// fichier s'il existe. Refus : LibraryUnavailable si FFmpeg n'a pas l'encodeur ou le conteneur,
// FileUnwritable si le fichier ne s'écrit pas.
std::expected<void, Refusal> write_bench_clip(const std::filesystem::path& file,
                                              const BenchClipSpec& spec = {});

}  // namespace ajp::media
