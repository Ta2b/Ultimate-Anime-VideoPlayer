#include "app/play/chain_shots.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

TEST_CASE("Pré-analyse : aucun plan à lire pour une source HDR sans grade natif", "[app][shots]") {
  PlaySettings settings;
  settings.grade = true;
  settings.hdr_native = false;
  ShotWork work;
  open_shots(work, "inexistant.mkv", settings, 0, ShotOutput{.source_hdr = true});
  CHECK(work.current.probe == nullptr);
  CHECK(work.current.timeline == nullptr);
}

TEST_CASE("Pré-analyse : rien sans grade", "[app][shots]") {
  PlaySettings settings;
  settings.grade = false;
  ShotWork work;
  open_shots(work, "inexistant.mkv", settings, 0, ShotOutput{});
  CHECK(work.current.probe == nullptr);
  CHECK(work.current.timeline == nullptr);
}

TEST_CASE("Pré-analyse : sans grade, elle part quand l'interpolation lit les coupes des plans (A42-09)",
          "[app][shots]") {
  PlaySettings settings;
  settings.grade = false;
  settings.rife_factor = 2;
  ShotWork work;
  work.shot_cuts = true;
  open_shots(work, "inexistant.mkv", settings, 0, ShotOutput{.source_hdr = true});
  REQUIRE(work.current.probe != nullptr);  // elle échouera sur ce fichier, sans gêner la lecture
  CHECK(work.current.timeline != nullptr);
  // Ni grade ni interpolation : elle s'arrête.
  settings.rife_factor = 0;
  open_shots(work, "inexistant.mkv", settings, 0, ShotOutput{.source_hdr = true});
  CHECK(work.current.probe == nullptr);
}

// B42-02 : en sortie HDR, le grade efface l'exposition par plan et la marge selon la scène ; il ne lit plus
// des plans que les flashs dangereux. Sans le limiteur, le second décodage de tout le fichier ne part pas.
TEST_CASE("Pré-analyse : en sortie HDR, seulement avec le limiteur de flashs", "[app][shots]") {
  PlaySettings settings;
  settings.grade = true;
  settings.hdr_native = true;
  settings.flash_guard = false;
  ShotWork work;
  open_shots(work, "inexistant.mkv", settings, 0, ShotOutput{.hdr_output = true});
  CHECK(work.current.probe == nullptr);
  settings.flash_guard = true;
  open_shots(work, "inexistant.mkv", settings, 0, ShotOutput{.hdr_output = true});
  CHECK(work.current.probe != nullptr);
  // En sortie SDR, la courbe suit le plan : l'analyse part sans le limiteur.
  settings.flash_guard = false;
  ShotWork sdr;
  open_shots(sdr, "inexistant.mkv", settings, 0, ShotOutput{.hdr_output = false});
  CHECK(sdr.current.probe != nullptr);
}

TEST_CASE("Limiteur de flashs : l'échelle du grade suit son pic", "[app][shots]") {
  CHECK(flash_gain_of(1000.0) == Catch::Approx(690.0 / 203.0));
  CHECK(flash_gain_of(400.0) == Catch::Approx(320.0 / 203.0));
  CHECK(flash_gain_of(2000.0) == Catch::Approx(1304.0 / 203.0));
  CHECK(flash_gain_of(100.0) == 1.0);  // jamais sous la source
  // B42-07 : continue et croissante (652 nits à 1 001 contre 690 à 1 000).
  double before = flash_gain_of(100.0);
  for (double peak = 100.0; peak <= 4000.0; peak += 1.0) {
    const double gain = flash_gain_of(peak);
    CHECK(gain >= before);
    CHECK(gain - before < 0.01);
    before = gain;
  }
}

// B42-01 : au point de reprise d'une analyse (un saut, une reprise), le plan d'après commence sans vraie
// coupe là où finissait le plan suivi. Avant, chaque numéro de plan nouveau était une coupe : la lumière
// sautait au milieu d'un plan réel.
TEST_CASE("Grade : un plan coupé au point de reprise de l'analyse n'est pas une coupe", "[app][shots]") {
  const ShotScene resumed{.start_us = 5'000'000, .end_us = 9'000'000, .cut_before = false, .planned = true};
  CHECK(continues_shot(resumed, 5'000'000));
  CHECK(continues_shot(resumed, 4'960'000));  // l'image de reprise retombe à 0,1 s près
  // Une vraie coupe, un plan loin du précédent, rien de suivi avant, une scène vue à l'ingestion : des
  // coupes.
  ShotScene cut = resumed;
  cut.cut_before = true;
  CHECK_FALSE(continues_shot(cut, 5'000'000));
  CHECK_FALSE(continues_shot(resumed, 3'000'000));
  CHECK_FALSE(continues_shot(resumed, -1));
  ShotScene live = resumed;
  live.planned = false;
  CHECK_FALSE(continues_shot(live, 5'000'000));
}

}  // namespace ajp::app::play
