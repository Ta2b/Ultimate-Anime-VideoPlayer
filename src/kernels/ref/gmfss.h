// Référence du softmax splatting de GMFSS Fortuna (spec/gmfss.h), dans l'ordre des pixels
// source. Repris de vsgmfss_fortuna 1.0.0 softsplat.py (softsplat, softsplat_out).
// Tests : tests/unit/kernels/gmfss_test.cpp.
#pragma once

#include <array>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/gmfss.h"

namespace ajp::kernels::ref {

// Le mouvement d'un côté à un niveau : flot fx, fy en pixels du niveau, métrique z (pour t = 1).
struct SplatMotion {
  std::array<PlaneF, kGmfssMotionChannels> planes;
};

// Les plans de source splattés à l'instant s du côté (gmfss_side_time). Refus : aucun plan,
// plans ou mouvement de tailles différentes, s hors de [0, 1].
Result<std::vector<PlaneF>> soft_splat(const std::vector<PlaneF>& source, const SplatMotion& motion, float s);

}  // namespace ajp::kernels::ref
