// Méga plan B2 : la mosaïque de calibration du goût (grade_mosaic.h) : un utilisateur simulé qui choisit
// toujours la version la plus proche de son goût le retrouve en dix-huit tours.
#include "app/play/grade_mosaic.h"

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "render/curve.h"

namespace ajp::app::play {
namespace {

using Catch::Approx;

double distance(const GradeTaste& a, const GradeTaste& b) {
  const TasteVector x = reduce(a);
  const TasteVector y = reduce(b);
  double d = 0.0;
  for (std::size_t i = 0; i < kTasteAxes; ++i) d += (x.at(i) - y.at(i)) * (x.at(i) - y.at(i));
  return std::sqrt(d);
}

GradeTaste run(const GradeTaste& truth, const GradeTaste& start = {}) {
  GradeMosaic mosaic(start);
  int rounds = 0;
  while (!mosaic.done()) {
    const MosaicRound r = mosaic.round();
    CHECK(r.index == rounds);
    mosaic.choose(distance(r.one, truth) <= distance(r.two, truth) ? 1 : 2);
    ++rounds;
  }
  CHECK(rounds == kMosaicRounds);
  return mosaic.result();
}

}  // namespace

TEST_CASE("mosaïque : les coordonnées réduites vont et reviennent", "[app][mosaic]") {
  GradeTaste t;
  t.e = {0.1, -0.2, 0.05, 0.25};
  t.toe = -0.1;
  t.vibrance = 1.1;
  const GradeTaste back = expand(reduce(t));
  for (std::size_t c = 0; c < kSceneClasses; ++c) CHECK(back.e.at(c) == Approx(t.e.at(c)));
  CHECK(back.toe == Approx(t.toe));
  CHECK(back.vibrance == Approx(t.vibrance));
  CHECK(reduce(GradeTaste{}) == TasteVector{});
}

TEST_CASE("mosaïque : chaque tour ne change qu'un axe, de part et d'autre du goût estimé", "[app][mosaic]") {
  GradeMosaic mosaic;
  const MosaicRound first = mosaic.round();
  CHECK(first.one.e.at(0) == Approx(-kMosaicStep * render::kCurveMaxE));
  CHECK(first.two.e.at(0) == Approx(kMosaicStep * render::kCurveMaxE));
  CHECK(first.one.e.at(1) == 0.0);
  CHECK(first.one.toe == 0.0);
  mosaic.choose(2);
  const MosaicRound second = mosaic.round();
  CHECK(second.one.e.at(0) == Approx(kMosaicStep * render::kCurveMaxE));  // l'axe 0 garde son estimation
  CHECK(second.one.e.at(1) == Approx(-kMosaicStep * render::kCurveMaxE));
  mosaic.choose(7);  // ignoré
  CHECK(mosaic.round().index == 1);
}

TEST_CASE("mosaïque : un utilisateur simulé retrouve son goût", "[app][mosaic]") {
  for (const double scale : {-0.8, -0.3, 0.0, 0.4, 0.8}) {
    GradeTaste truth;
    truth.e = {0.25 * scale, -0.2 * scale, 0.1 * scale, 0.22 * scale};
    truth.toe = 0.15 * scale;
    truth.vibrance = 1.0 + 0.15 * std::abs(scale);
    const GradeTaste found = run(truth);
    INFO("échelle " << scale);
    for (std::size_t c = 0; c < kSceneClasses; ++c)
      CHECK(std::abs(found.e.at(c) - truth.e.at(c)) <= 0.2 * render::kCurveMaxE);
    CHECK(std::abs(found.toe - truth.toe) <= 0.2 * render::kCurveMaxToe);
    CHECK(std::abs(found.vibrance - truth.vibrance) <= 0.25 * (render::kCurveMaxVibrance - 1.0));
  }
}

TEST_CASE("mosaïque : sans choix, le goût de départ ; le résultat reste dans les bornes", "[app][mosaic]") {
  GradeTaste start;
  start.e = {0.1, 0.0, 0.0, 0.0};
  CHECK(GradeMosaic(start).result() == clamped(start));
  GradeMosaic mosaic;
  while (!mosaic.done()) mosaic.choose(2);
  const GradeTaste top = mosaic.result();
  for (const double e : top.e) CHECK(e <= render::kCurveMaxE);
  CHECK(top.toe <= render::kCurveMaxToe);
  CHECK(top.vibrance <= render::kCurveMaxVibrance);
}

}  // namespace ajp::app::play
