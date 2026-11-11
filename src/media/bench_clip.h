// La séquence de l'évaluation de la machine (« Évaluer ma machine », ajp bench) : un court clip synthétique,
// écrit une fois par le lecteur lui-même, pour que l'évaluation n'ait besoin d'aucune vidéo
// (docs/plan-v0.6.md). Version 4 (bench-clip-v4.mkv) : des plans de deux secondes, calme puis combat rapide
// puis combat sombre, en boucle ; fond qui défile (40 px par image en combat), personnages qui traversent
// l'écran en une seconde et se masquent, lame qui tourne, hachures fines, dégradés sombres et grain sur toute
// l'image.
//
// Écart avec un épisode réel (B07-15) : MPEG-2 8 bits SDR, et chaque image change (panoramique, grain). Un
// épisode est le plus souvent en HEVC ou AV1 10 bits, animé en deux ou trois images. Le décodage (NVDEC sur
// une RTX) ne prend pas le GPU de calcul, mais l'envoi d'une image 10 bits double la bande passante d'entrée
// ; les dessins tenus, eux, épargnent à RIFE des paires entières. La séquence est donc plus dure que la
// plupart des épisodes pour l'interpolation, et un peu plus légère pour l'entrée ; la marge réelle n'est pas
// mesurée.
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
