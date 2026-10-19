// Traduction des métadonnées HDR10 vers les structures de FFmpeg. Interne à export.
#pragma once

#include <expected>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
}

#include "core/types.h"
#include "export/hdr10.h"

namespace ajp::exporter {

// Affichage de mastering dans les données annexes d'entrée d'un encodeur
// (AVCodecContext::decoded_side_data) : l'encodeur l'écrit dans le flux quand il le sait
// (SEI de HEVC, OBU de métadonnées d'AV1).
std::expected<void, Refusal> set_encoder_mastering(AVCodecContext& encoder, const Hdr10Mastering& mastering);

// Mastering et niveaux de lumière dans les données annexes du flux d'un conteneur
// (AVCodecParameters::coded_side_data) : Matroska les écrit dans Colour, MP4 dans mdcv et clli.
// Une entrée du même type déjà présente est remplacée.
std::expected<void, Refusal> set_stream_hdr10(AVCodecParameters& par, const Hdr10Mastering& mastering,
                                              const LightLevel& light);

}  // namespace ajp::exporter
