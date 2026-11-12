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

TEST_CASE("anime par plan : la série est la médiane des APL connues", "[app][anime]") {
  CHECK(series_apl_of({}) == 0.0);
  CHECK(series_apl_of({shot_of(10.0F), shot_of(0.0F), shot_of(80.0F), shot_of(30.0F)}) == Approx(30.0));
}

TEST_CASE("anime par plan : une coupe change à l'image de coupe, sans glissé après elle", "[app][anime]") {
  AnimeGlide glide;
  CHECK_FALSE(glide.known());
  const AnimeShotLook dark{.light = {.apl_nits = 10.0, .lit = 0.0}};
  const AnimeShotLook bright{.light = {.apl_nits = 90.0, .lit = 0.4}};
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

// C41-01 : l'exposition par plan est retirée ; les surfaces gardent le même relèvement quelle que soit la
// lumière du plan.
TEST_CASE("anime par plan : les surfaces gardent le relèvement fixe, quelle que soit la scène",
          "[app][anime]") {
  render::GradeSettings s;
  for (const double apl : {5.0, 30.0, 150.0}) {
    s.scene_light = render::SceneLight{.apl_nits = apl, .lit = 0.1};
    const auto decision = render::decide_grade(s, pq_context());
    REQUIRE(decision.has_value());
    CHECK(decision->params.an_lift == render::kAnimeLift);
  }
}

TEST_CASE("anime par plan : en HDR, la luminosité ne suit ni le plan ni la scène ; en SDR, la marge seule",
          "[app][anime]") {
  // Retour d'essai de la rc.4 : l'exposition par plan et la marge des éclats selon la scène, amplifiées par
  // la pile de LUT, faisaient sauter un même décor d'un plan à l'autre. La dalle limite les scènes claires.
  PlaySettings s;
  s.hdr_native = true;
  const render::GradeChoice still = grade_of(s, 1000);
  REQUIRE(still.pq.has_value());
  REQUIRE(still.sdr_anime.has_value());
  s.grade_values.scene_light = render::SceneLight{.apl_nits = 150.0, .lit = 0.6};  // une scène très claire
  const render::GradeChoice bright = grade_of(s, 1000);
  REQUIRE(bright.pq.has_value());
  REQUIRE(bright.sdr_anime.has_value());
  CHECK(bright.pq->an_lift == Approx(render::kAnimeLift));
  CHECK(bright.pq->fs_apl < 0.0);  // clé inconnue : la marge des éclats au pic
  CHECK(*bright.pq == *still.pq);
  // La sortie SDR garde la marge des éclats par plan ; ses surfaces ne bougent pas (C41-01).
  CHECK(bright.sdr_anime->an_lift == still.sdr_anime->an_lift);
  CHECK(bright.sdr_anime->tm_white == still.sdr_anime->tm_white);
  CHECK(bright.sdr_anime->tm_top < still.sdr_anime->tm_top);
}

// C41-10 : la sortie SDR de l'anime n'a pas la pile de LUT (étalonnée pour des codes PQ), la sortie PQ si ;
// une source HDR n'est pas gradée.
TEST_CASE("anime : la sortie SDR coupe la pile de LUT, la sortie PQ la garde", "[app][anime]") {
  PlaySettings s;
  s.hdr_native = true;
  const render::GradeChoice choice = grade_of(s, 1000);
  REQUIRE(choice.pq.has_value());
  REQUIRE(choice.sdr_anime.has_value());
  bool any_look = false;
  for (const double mix : choice.pq->look_mixes) any_look = any_look || mix > 0.0;
  CHECK(any_look);  // la pile par défaut
  for (const double mix : choice.sdr_anime->look_mixes) CHECK(mix == 0.0);
  CHECK(choice.sdr_anime->look_pq == 0.0);
  CHECK(choice.sdr_anime->sdr_tm == 1.0);
  CHECK(choice.pq->sdr_tm == 0.0);
  const render::GradeChoice hdr_source = grade_of(s, 1000, {}, {}, true);
  CHECK_FALSE(hdr_source.pq.has_value());
  CHECK_FALSE(hdr_source.sdr_anime.has_value());
}

TEST_CASE("flashs dangereux : le grade PQ plafonné à 250 nits, l'annonce garde le pic", "[app][anime]") {
  PlaySettings s;
  s.hdr_native = true;
  render::GradeChoice choice = grade_of(s, 1000);
  REQUIRE(choice.pq.has_value());
  const render::GradeParams sdr = *choice.sdr_anime;
  cap_flash_peak(choice);
  CHECK(choice.pq->peak == 250.0);
  CHECK(choice.pq->uncapped_peak == 1000.0);
  CHECK(*choice.sdr_anime == sdr);
  render::GradeChoice low = grade_of(s, 200);
  REQUIRE(low.pq.has_value());
  cap_flash_peak(low);
  CHECK(low.pq->peak == 200.0);
  CHECK(low.pq->uncapped_peak == 0.0);
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
