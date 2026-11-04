#include "app/play/grade_mosaic.h"

#include <algorithm>

#include "render/curve.h"

namespace ajp::app::play {
namespace {

constexpr double kVibranceSpan = render::kCurveMaxVibrance - 1.0;

}  // namespace

TasteVector reduce(const GradeTaste& taste) {
  TasteVector v{};
  for (std::size_t c = 0; c < kSceneClasses; ++c) v.at(c) = taste.e.at(c) / render::kCurveMaxE;
  v.at(kSceneClasses) = taste.toe / render::kCurveMaxToe;
  v.at(kSceneClasses + 1) = (taste.vibrance - 1.0) / kVibranceSpan;
  return v;
}

GradeTaste expand(const TasteVector& v) {
  GradeTaste t;
  for (std::size_t c = 0; c < kSceneClasses; ++c) t.e.at(c) = v.at(c) * render::kCurveMaxE;
  t.toe = v.at(kSceneClasses) * render::kCurveMaxToe;
  t.vibrance = 1.0 + v.at(kSceneClasses + 1) * kVibranceSpan;
  return clamped(t);
}

GradeMosaic::GradeMosaic(const GradeTaste& start) : start_(reduce(clamped(start))), centre_(start_) {
  step_.fill(kMosaicStep);
}

MosaicRound GradeMosaic::round() const {
  const int index = std::min(round_, kMosaicRounds - 1);
  const auto axis = static_cast<std::size_t>(index) % kTasteAxes;
  TasteVector one = centre_;
  TasteVector two = centre_;
  one.at(axis) -= step_.at(axis);
  two.at(axis) += step_.at(axis);
  return MosaicRound{.index = index, .one = expand(one), .two = expand(two)};
}

void GradeMosaic::choose(int which) {
  if (done() || (which != 1 && which != 2)) return;
  const auto axis = static_cast<std::size_t>(round_) % kTasteAxes;
  const double lo = axis < kSceneClasses + 1 ? -1.0 : 0.0;
  centre_.at(axis) = std::clamp(centre_.at(axis) + (which == 2 ? 1.0 : -1.0) * step_.at(axis), lo, 1.0);
  step_.at(axis) *= 0.5;
  ++seen_.at(axis);
  ++round_;
}

GradeTaste GradeMosaic::result() const {
  // Moindres carrés sur l'axe : n bornes au centre trouvé (poids 1 chacune) et le départ (kMosaicPrior).
  TasteVector v{};
  for (std::size_t a = 0; a < kTasteAxes; ++a) {
    const double n = seen_.at(a);
    v.at(a) = (kMosaicPrior * start_.at(a) + n * centre_.at(a)) / (kMosaicPrior + n);
  }
  return expand(v);
}

}  // namespace ajp::app::play
