// Référence du mouvement d'une caméra affine (spec/motion_camera.h). Tests :
// tests/unit/kernels/motion_camera_test.cpp.
#pragma once

#include <array>

#include "kernels/ref/warp.h"

namespace kirari::kernels::ref {

// Le mouvement de la caméra dans des plans plane_width × plane_height, fenêtre area, scale pixels de la
// fenêtre par pixel de travail. Refus : plans vides, fenêtre hors des plans ou vide, scale ≤ 0 ou non fini,
// caméra non finie.
Result<Motion> motion_camera(const std::array<float, 6>& camera, float scale, int plane_width,
                             int plane_height, const Window& area);

}  // namespace kirari::kernels::ref
