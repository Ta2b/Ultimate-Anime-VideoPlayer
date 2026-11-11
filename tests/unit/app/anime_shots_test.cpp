// Méga plan A2 (--anime-shots) : la méthode « HDR anime » décidée par plan, glissée aux coupes
// (anime_shots.h), et l'exposition reçue par le grade (GradeParams::an_lift).
#include "app/play/anime_shots.h"

#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "app/play/chain.h"
#include "app/play/settings.h"
#include "render/grade_params.h"

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
  // Audit F3 : les surfaces restent ancrées, ±5 % au plus d'un plan à l'autre (|E| ≤ 0,07).
  CHECK(anime_exposure(36.0, 40.0) == Approx(-render::kAnimeExposureSlope * std::log2(0.9)));
  CHECK(anime_exposure(20.0, 40.0) == Approx(kAnimeSurfaceMaxE));
  CHECK(anime_exposure(160.0, 40.0) == Approx(-kAnimeSurfaceMaxE));
  CHECK(std::exp2(kAnimeSurfaceMaxE) <= 1.05 + 1e-9);
}

TEST_CASE("anime par plan : la série est la médiane des APL connues", "[app][anime]") {
  CHECK(series_apl_of({}) == 0.0);
  CHECK(series_apl_of({shot_of(10.0F), shot_of(0.0F), shot_of(80.0F), shot_of(30.0F)}) == Approx(30.0));
}

TEST_CASE("anime par plan : une coupe change à l'image de coupe, sans glissé après elle", "[app][anime]") {
  AnimeGlide glide;
  CHECK_FALSE(glide.known());
  const AnimeShotLook dark{.e = 0.07, .light = {.apl_nits = 10.0, .lit = 0.0}};
  const AnimeShotLook bright{.e = -0.07, .light = {.apl_nits = 90.0, .lit = 0.4}};
  glide.aim(dark, 1'000'000, true);  // le premier plan
  CHECK(glide.at(1'000'000) == dark);
  CHECK_FALSE(glide.gliding(1'000'000));
  glide.aim(bright, 2'000'000, true);  // la coupe : aussitôt
  CHECK_FALSE(glide.gliding(2'000'000));
  CHECK(glide.at(2'000'000) == bright);
  // B42-08 : le même plan mieux connu (ou la jointure de deux analyses) glisse vers sa nouvelle valeur ; le
  // glissé n'avait jamais lieu.
  AnimeShotLook better = bright;
  better.light.apl_nits = 70.0;
  glide.aim(better, 2'100'000, false);
  CHECK(glide.gliding(2'100'000));
  CHECK(glide.at(2'100'000) == bright);
  const AnimeShotLook half = glide.at(2'100'000 + (kAnimeGlideUs / 2));
  CHECK(half.light.apl_nits < 90.0);
  CHECK(half.light.apl_nits > 70.0);
  CHECK_FALSE(glide.gliding(2'100'000 + kAnimeGlideUs));
  CHECK(glide.at(2'100'000 + kAnimeGlideUs) == better);
  // La même valeur visée de nouveau ne relance rien.
  glide.aim(better, 2'500'000, false);
  CHECK_FALSE(glide.gliding(2'500'000));
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
  CHECK(low->params.an_lift == Approx(render::kAnimeLift * std::exp2(-render::kAnimeExposureMaxE)));
}

TEST_CASE("anime par plan : en HDR, la luminosité ne suit ni le plan ni la scène ; en SDR, si",
          "[app][anime]") {
  // Retour d'essai de la rc.4 : l'exposition par plan et la marge des éclats selon la scène, amplifiées par
  // la pile de LUT, faisaient sauter un même décor d'un plan à l'autre. La dalle limite les scènes claires.
  PlaySettings s;
  s.hdr_native = true;
  const render::GradeChoice still = grade_of(s, 1000);
  REQUIRE(still.pq.has_value());
  REQUIRE(still.sdr_anime.has_value());
  s.grade_values.scene_light = render::SceneLight{.apl_nits = 150.0, .lit = 0.6};  // une scène très claire
  s.grade_values.anime_exposure = -0.25;
  const render::GradeChoice bright = grade_of(s, 1000);
  REQUIRE(bright.pq.has_value());
  REQUIRE(bright.sdr_anime.has_value());
  CHECK(bright.pq->an_lift == Approx(render::kAnimeLift));
  CHECK(bright.pq->fs_apl < 0.0);  // clé inconnue : la marge des éclats au pic
  CHECK(*bright.pq == *still.pq);
  // La sortie SDR garde sa courbe par plan.
  CHECK(bright.sdr_anime->an_lift < still.sdr_anime->an_lift);
  CHECK(bright.sdr_anime->tm_top < still.sdr_anime->tm_top);
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
