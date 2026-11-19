// Référence du softmax splatting de GMFSS Fortuna (spec/gmfss.h), dans l'ordre des pixels source. Repris de
// vsgmfss_fortuna 1.0.0 softsplat.py (softsplat, softsplat_out). Tests : tests/unit/kernels/gmfss_test.cpp.
#pragma once

#include <array>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/warp.h"

namespace kirari::kernels::ref {

// Le mouvement d'un côté à un niveau : flot fx, fy en pixels du niveau, métrique z (pour t = 1).
struct SplatMotion {
  std::array<PlaneF, kGmfssMotionChannels> planes;
};

// Les plans de source splattés à l'instant s du côté. Refus : aucun plan, plans ou mouvement de tailles
// différentes, s hors de [0, 1].
Result<std::vector<PlaneF>> soft_splat(const std::vector<PlaneF>& source, const SplatMotion& motion, float s);

// Un splat et sa couverture (spec MotionGrid).
struct Splatted {
  std::vector<PlaneF> planes;
  PlaneF coverage;
};

// Le splat de source par un mouvement lu sur sa grille (spec MotionGrid), avec sa couverture. Refus : ceux de
// soft_splat, mouvement hors contrat, échelle non positive.
Result<Splatted> soft_splat_at(const std::vector<PlaneF>& source, const SplatMotion& motion, float s,
                               const MotionGrid& grid);

// Le splat tiré (spec MotionGrid, splat tiré) : la source lue en Catmull-Rom là d'où vient chaque pixel de
// sortie, avec la couverture. Refus : ceux de soft_splat_at.
Result<Splatted> soft_pull_at(const std::vector<PlaneF>& source, const SplatMotion& motion, float s,
                              const MotionGrid& grid);

// La composition à la taille de travail (spec gmfss_compose) : high_* à la taille de sortie, low_* à celle de
// RIFE (au plus la sortie), fusion les trois plans remplis et window l'image de RIFE dedans. Refus : tailles
// incohérentes, t hors de [0, 1].
Result<std::vector<PlaneF>> gmfss_compose(const Splatted& high_a, const Splatted& high_b,
                                          const Splatted& low_a, const Splatted& low_b,
                                          const std::vector<PlaneF>& fusion, const Window& window, float t);

// Le décor fixe repris des sources (spec gmfss_still) : key, a et b de 3 plans de même taille. Refus :
// tailles différentes, t hors de [0, 1].
Result<std::vector<PlaneF>> gmfss_still(const std::vector<PlaneF>& key, const std::vector<PlaneF>& a,
                                        const std::vector<PlaneF>& b, float t);

}  // namespace kirari::kernels::ref
