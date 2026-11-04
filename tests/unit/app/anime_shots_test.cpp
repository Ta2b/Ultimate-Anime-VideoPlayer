// Méga plan A2 (--anime-shots) : la méthode « HDR anime » décidée par plan, glissée aux coupes
// (anime_shots.h), et l'exposition reçue par le grade (GradeParams::an_lift).
#include "app/play/anime_shots.h"

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "render/curve.h"

namespace ajp::app::play {
namespace {

using Catch::Approx;

media::Shot shot_of(float apl) {
  media::Shot s;
  s.meta.apl_mean_nits = apl;
  return s;
}

render::GradeContext pq_context() {
  return render::GradeContext{.route = core::HdrMode::Native,
                              .source_hdr = false,
                              .output_pq = true,
                              .enabled = true,
                              .peak_nits = 1000.0};
}

}  // namespace

TEST_CASE("anime par plan : l'exposition suit la règle de la courbe par plan, bornée", "[app][anime]") {
  CHECK(anime_exposure(40.0, 0.0) == 0.0);
  CHECK(anime_exposure(0.0, 40.0) == 0.0);
  CHECK(anime_exposure(40.0, 40.0) == Approx(0.0));
  // Un diaphragme plus sombre que la série : +0,25 ; deux plus clair : borné à −0,25.
  CHECK(anime_exposure(20.0, 40.0) == Approx(render::kCurveSlope));
  CHECK(anime_exposure(160.0, 40.0) == Approx(-render::kCurveMaxE));
  CHECK(anime_exposure(1.0, 400.0) == Approx(render::kCurveMaxE));
}

TEST_CASE("anime par plan : la série est la médiane des APL connues", "[app][anime]") {
  CHECK(series_apl_of({}) == 0.0);
  CHECK(series_apl_of({shot_of(10.0F), shot_of(0.0F), shot_of(80.0F), shot_of(30.0F)}) == Approx(30.0));
}

TEST_CASE("anime par plan : une coupe glisse, le même plan suit aussitôt", "[app][anime]") {
  AnimeGlide glide;
  CHECK_FALSE(glide.known());
  const AnimeShotLook dark{.e = 0.25, .light = {.apl_nits = 10.0, .lit = 0.0}};
  const AnimeShotLook bright{.e = -0.25, .light = {.apl_nits = 90.0, .lit = 0.4}};
  glide.aim(dark, 1'000'000, true);  // le premier plan : aucun glissé depuis rien
  CHECK(glide.at(1'000'000) == dark);
  CHECK_FALSE(glide.gliding(1'000'000));

  glide.aim(bright, 2'000'000, true);
  CHECK(glide.gliding(2'000'000));
  CHECK(glide.at(2'000'000) == dark);
  const AnimeShotLook mid = glide.at(2'000'000 + kAnimeGlideUs / 2);
  CHECK(mid.e == Approx(0.0).margin(1e-9));
  CHECK(mid.light.apl_nits == Approx(50.0));
  CHECK(glide.at(2'000'000 + kAnimeGlideUs) == bright);
  // Monotone tout du long.
  double last = 1.0;
  for (std::int64_t t = 0; t <= kAnimeGlideUs; t += kAnimeGlideUs / 20) {
    const double e = glide.at(2'000'000 + t).e;
    CHECK(e <= last + 1e-12);
    last = e;
  }

  // Le même plan mieux connu, en plein glissé : la cible bouge, pas le départ.
  AnimeShotLook better = bright;
  better.light.apl_nits = 70.0;
  glide.aim(better, 2'000'000 + kAnimeGlideUs / 2, false);
  CHECK(glide.at(2'000'000 + kAnimeGlideUs / 2).light.apl_nits == Approx(40.0));
  CHECK(glide.at(2'000'000 + kAnimeGlideUs) == better);
  // Hors glissé, hors coupe : aussitôt.
  glide.aim(dark, 3'000'000, false);
  CHECK(glide.at(3'000'000) == dark);
}

TEST_CASE("anime par plan : l'exposition relève les surfaces du grade, rien sans elle", "[app][anime]") {
  render::GradeSettings s;
  s.scene_light = render::SceneLight{.apl_nits = 30.0, .lit = 0.1};
  const auto plain = render::decide_grade(s, pq_context());
  REQUIRE(plain.has_value());
  CHECK(plain->params.an_lift == Approx(render::kAnimeLift));
  s.anime_exposure = 0.25;
  const auto lifted = render::decide_grade(s, pq_context());
  REQUIRE(lifted.has_value());
  CHECK(lifted->params.an_lift == Approx(render::kAnimeLift * std::exp2(0.25)));
  s.anime_exposure = -3.0;  // borné
  const auto low = render::decide_grade(s, pq_context());
  REQUIRE(low.has_value());
  CHECK(low->params.an_lift == Approx(render::kAnimeLift * std::exp2(-render::kCurveMaxE)));
}

TEST_CASE("anime : le peps en % devient contraste, saturation et part des éclats du grade", "[app][anime]") {
  render::GradeSettings s;
  const auto plain = render::decide_grade(s, pq_context());
  REQUIRE(plain.has_value());
  CHECK(plain->params.an_punch == Approx(0.0));
  CHECK(plain->params.an_chroma == Approx(1.0));
  CHECK(plain->params.an_spec == Approx(1.0));
  s.anime_contrast = 150;
  s.anime_saturation = 120;
  s.anime_highlights = 0;
  const auto peps = render::decide_grade(s, pq_context());
  REQUIRE(peps.has_value());
  CHECK(peps->params.an_punch == Approx(0.5));
  CHECK(peps->params.an_chroma == Approx(1.2));
  CHECK(peps->params.an_spec == Approx(0.0));
}

}  // namespace ajp::app::play
