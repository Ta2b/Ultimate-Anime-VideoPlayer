#include "render/display_model.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace ajp::render {
namespace {

struct AblPoint {
  double area;
  double nits;
};
// RTINGS, QD-OLED 27" 4K (PG27UCDM) : fenêtres de 2, 10, 25, 50 et 100 %.
constexpr std::array<AblPoint, 5> kQdOled{
    {{0.02, 1016.0}, {0.10, 460.0}, {0.25, 367.0}, {0.50, 310.0}, {1.00, 261.0}}};

}  // namespace

KwinHint kwin_hint(double content_peak_nits, double reference_nits) {
  const DisplayModel model{.panel_peak_nits = content_peak_nits * reference_nits / kContentReferenceNits,
                           .reference_nits = reference_nits};
  return {.panel_peak_nits = model.panel_peak_nits, .max_content_nits = content_peak_nits};
}

double oled_abl_ratio(double area) {
  const double a = std::clamp(area, kQdOled.front().area, 1.0);
  // Interpolée en log de la surface et en log de la luminance : la courbe d'un limiteur est à peu près une
  // loi de puissance d'un relevé au suivant.
  for (std::size_t i = 1; i < kQdOled.size(); ++i) {
    if (a > kQdOled[i].area) continue;
    const AblPoint& p = kQdOled[i - 1];
    const AblPoint& q = kQdOled[i];
    const double t = std::log(a / p.area) / std::log(q.area / p.area);
    return std::exp(std::log(p.nits) + (t * std::log(q.nits / p.nits))) / kQdOled.front().nits;
  }
  return kQdOled.back().nits / kQdOled.front().nits;
}

double DisplayModel::scale() const {
  return reference_nits > 0.0 ? reference_nits / kContentReferenceNits : 1.0;
}

double DisplayModel::content_peak_nits() const {
  return panel_peak_nits / scale();
}

double DisplayModel::max_nits(double area) const {
  return oled ? panel_peak_nits * oled_abl_ratio(area) : panel_peak_nits;
}

std::optional<double> DisplayModel::frame_average_content_nits() const {
  if (!oled) return std::nullopt;
  return max_nits(1.0) / scale();
}

}  // namespace ajp::render
