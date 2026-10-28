// Le conteneur d'un export, choisi par l'extension du fichier. MKV, le défaut, porte tout (sous-
// titres ASS, polices jointes) mais date chaque paquet à la milliseconde : à 120 i/s, les images
// tombent à 8, 8 puis 9 ms au lieu de 8,333. MP4 et MOV gardent chaque instant exact, sans sous-
// titres ASS ni polices, et avec moins de formats audio (TrueHD : MKV seulement ; FLAC et Opus :
// pas en MOV) ; FFV1 ne va pas dans un MOV. Logique pure, sans FFmpeg.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

#include "export/encoder_choice.h"

namespace ajp::exporter {

enum class Container : std::uint8_t { Matroska, Mp4, Mov };

// .mkv, .mp4 ou .mov, quelle que soit la casse ; rien pour une autre extension.
std::optional<Container> container_of(const std::filesystem::path& output);
std::string_view label(Container container);        // « MKV »
std::string_view format_name(Container container);  // le nom du format avformat : « matroska »
bool carries(Container container, Codec codec);     // FFV1 : MKV et MP4 seulement
bool carries_attachments(Container container);      // les polices jointes : MKV seulement

// L'échelle de temps de la piste vidéo d'un MP4 ou d'un MOV : la fréquence, doublée jusqu'à 10 000
// au moins, pour que chaque période tombe sur un nombre entier de pas (1001 à 120000/1001 i/s) et
// que l'arrondi des instants à la microseconde disparaisse. 0 : fréquence inconnue, ou trop fine
// pour cet arrondi, la microseconde de l'encodeur.
int video_timescale(int fps_num, int fps_den);

}  // namespace ajp::exporter
