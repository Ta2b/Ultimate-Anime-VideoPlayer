// Référence du softmax splatting de GMFSS Fortuna (spec/gmfss.h), dans l'ordre des pixels
// source, et de la composition après SPAN. Repris de vsgmfss_fortuna 1.0.0 softsplat.py
// (softsplat, softsplat_out). Tests : tests/unit/kernels/gmfss_test.cpp.
#pragma once

#include <array>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/warp.h"

namespace ajp::kernels::ref {

// Le mouvement d'un côté à un niveau : flot fx, fy en pixels du niveau, métrique z (pour t = 1).
struct SplatMotion {
  std::array<PlaneF, kGmfssMotionChannels> planes;
};

// Les plans de source splattés à l'instant s du côté (gmfss_side_time). Refus : aucun plan,
// plans ou mouvement de tailles différentes, s hors de [0, 1].
Result<std::vector<PlaneF>> soft_splat(const std::vector<PlaneF>& source, const SplatMotion& motion, float s);

// Le même splat, le mouvement lu sur sa propre grille (spec MotionGrid : scale > 0), et la
// couverture en plus. Refus : ceux de soft_splat sauf la taille du mouvement, scale non positif.
struct Splatted {
  std::vector<PlaneF> planes;
  PlaneF coverage;
};
Result<Splatted> soft_splat_at(const std::vector<PlaneF>& source, const SplatMotion& motion, float s,
                               const MotionGrid& grid);

// Déformation après SPAN (spec gmfss_compose) : high_a, high_b les sources agrandies splattées
// (3 plans et leur couverture), low_a, low_b les mêmes à la taille de travail, fusion 3 plans
// remplis dont window est la fenêtre, de la taille de travail ; rend H + F − L borné à [0, 1].
// Refus : autre chose que 3 plans, tailles mêlées, sortie qui n'est pas un multiple entier de la
// taille de travail, fenêtre hors de fusion ou d'une autre taille, t hors de [0, 1].
Result<std::vector<PlaneF>> gmfss_compose(const Splatted& high_a, const Splatted& high_b,
                                          const Splatted& low_a, const Splatted& low_b,
                                          const std::vector<PlaneF>& fusion, const Window& window, float t);

}  // namespace ajp::kernels::ref
