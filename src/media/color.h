// Traduction des descriptions FFmpeg (format de pixel, couleur, HDR10) vers le
// vocabulaire de core/frame.h. Interne au module. Une valeur que ce vocabulaire
// ne sait pas dire est refusée, jamais rapprochée en silence.
#pragma once

#include <expected>
#include <optional>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

#include "core/frame.h"
#include "core/types.h"

namespace ajp::media {

std::expected<PixelFormat, Refusal> map_pixel_format(AVPixelFormat format);

// Couleur d'une image décodée, sw_format étant le format des pixels (celui du
// pool pour une image matérielle). Les champs non renseignés sont devinés comme
// mpv le fait ; les métadonnées HDR10 viennent des side data de l'image.
std::expected<ColorInfo, Refusal> color_info(const AVFrame& frame, AVPixelFormat sw_format);

// Affichage de mastering (SMPTE ST 2086) et niveaux de lumière ; std::nullopt
// si l'image n'en porte aucun. Une valeur hors des bornes physiques est refusée.
std::expected<std::optional<HdrMetadata>, Refusal> hdr_metadata(const AVFrame& frame);

}  // namespace ajp::media
