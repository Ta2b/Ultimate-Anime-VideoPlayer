// Une image de la calibration du grade adaptatif (calibration_images.h), telle que l'écran la montre : ses
// pixels, sa place (à gauche du panneau des paramètres) et le grade de la route HDR adaptatif qui la rend.
// À la place de la vidéo, des sous-titres et de l'OSD ; l'interface reste par-dessus.
#pragma once

#include <cstdint>
#include <span>

#include "render/grade_params.h"

namespace ajp::app::play {

struct CalibrationShot {
  int id = -1;  // l'image : la texture n'est envoyée au GPU qu'au changement
  int width = 0;
  int height = 0;
  std::span<const std::uint16_t> rgba;  // RGBA16, BT.709, codes BT.1886 (une image SDR, comme un anime)
  int x0 = 0;                           // sa place à l'écran, en pixels
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
  render::GradeChoice grade;  // le grade de la route HDR adaptatif, scène forcée
};

}  // namespace ajp::app::play
