// Une image fixe en mémoire (JPEG ou PNG, embarquée dans le lecteur) décodée en RGBA 16 bits : les images
// d'exemple de la calibration du grade (app/play/calibration_images.h). FFmpeg reste dans le .cpp.
#pragma once

#include <cstdint>
#include <expected>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::media {

struct Picture {
  int width = 0;
  int height = 0;
  std::vector<std::uint16_t> rgba;  // width × height × 4, codes de l'image (BT.1886 pour un anime)
};

// Refus FormatUnsupported : ni JPEG ni PNG ; DecodeFailed : l'image ne se décode pas.
std::expected<Picture, Refusal> decode_picture(std::string_view bytes);

}  // namespace ajp::media
