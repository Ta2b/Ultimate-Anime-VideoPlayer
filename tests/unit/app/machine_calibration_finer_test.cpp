// La calibration au-delà des modèles du niveau (C07-03), la priorité Image (C07-04) et le rejeu de justesse
// (C07-05) : à côté de machine_calibration_test.cpp, qui approche la limite de taille.
#include <filesystem>
#include <stop_token>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/machine_calibration.h"

using namespace ajp::app::play;

namespace {

template <typename Holds>
std::vector<CandidateTrial> search(Holds holds, int start_height, bool try_finer) {
  std::vector<CandidateTrial> tried;
  while (const auto chain = next_candidate(tried, start_height, try_finer)) {
    REQUIRE(tried.size() < 20);
    tried.push_back({.chain = *chain, .holds = holds(*chain)});
  }
  return tried;
}

}  // namespace

TEST_CASE("C07-03 : une carte qui tient le plus haut palier essaie les modèles de Qualité",
          "[calibration-machine]") {
  using C = ChainCandidate;
  const C finer{.span = true, .rife = true, .height = 1080, .finer = true};
  // Une RTX 3080 estimée Normal (720p) tient tout : SPAN seul monte à 1080, SPAN + RIFE tient à 1080, puis
  // les modèles de Qualité y sont essayés, et gardés.
  auto tried = search([](const C&) { return true; }, 720, true);
  REQUIRE(tried.size() == 4);
  CHECK(tried.back().chain == finer);
  CHECK(best_candidate(tried) == finer);
  // Ils ne tiennent pas : la chaîne du niveau reste recommandée.
  tried = search([](const C& c) { return !c.finer; }, 720, true);
  CHECK(tried.back().chain == finer);
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 1080});
  // La chaîne ne tient qu'à 720 : rien de plus lourd n'est essayé.
  tried = search([](const C& c) { return c.height <= 720; }, 720, true);
  for (const CandidateTrial& trial : tried) CHECK_FALSE(trial.chain.finer);
  // Modèles déjà de Qualité (try_finer faux) : aucun essai de plus.
  tried = search([](const C&) { return true; }, 1080, false);
  CHECK(tried.size() == 2);
  // Les modèles de Qualité sont ceux du niveau Qualité.
  CHECK(finer_models() == calibration_models(Tier::Quality));
  CHECK(finer_models() != calibration_models(Tier::Normal));
}

TEST_CASE("C07-03 : la recommandation aux modèles de Qualité se relit et l'Auto les suit",
          "[calibration-machine]") {
  using C = ChainCandidate;
  const C finer{.span = true, .rife = true, .height = 1080, .finer = true};
  MachineCalibration c{.chosen = finer,
                       .tried = {{.chain = {.span = true, .rife = false, .height = 1080}, .holds = true},
                                 {.chain = {.span = true, .rife = true, .height = 1080}, .holds = true},
                                 {.chain = finer, .holds = true}},
                       .gpu = "RTX 3080",
                       .display_height = 1440,
                       .models = finer_models(),
                       .engine = "tensorrt"};
  const auto read = parse_calibration(format_calibration(c));
  REQUIRE(read);
  CHECK(read->chosen == finer);
  REQUIRE(read->tried.size() == 3);
  CHECK(read->tried[2].chain.finer);
  CHECK_FALSE(read->tried[1].chain.finer);
  const CalibrationHost host{.gpu = "RTX 3080", .display_height = 1440, .tensorrt = true};
  const auto followed = followed_chain(*read, host);
  REQUIRE(followed);
  CHECK(followed->height == 1080);
  CHECK(followed->span_model == kQualitySpanModel);
  CHECK(followed->rife_model == kQualityRifeModel);
  // Un fichier d'avant C07-03 (sans « finer ») se relit comme avant.
  const auto old = parse_calibration(
      "version = 4\nchosen = { span = true, rife = true, height = 720 }\n"
      "[[trials]]\nspan = true\nrife = true\nheight = 720\nholds = true\n");
  REQUIRE(old);
  CHECK_FALSE(old->chosen.finer);
}

TEST_CASE("C07-04 : la calibration garde le plus haut SPAN seul pour la priorité Image",
          "[calibration-machine]") {
  using C = ChainCandidate;
  // SPAN seul tient 1080p, SPAN + RIFE seulement 480p.
  const auto tried = search([](const C& c) { return !c.rife || c.height <= 480; }, 720, true);
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 480});
  CHECK(best_span_alone(tried) == 1080);
  const MachineCalibration c{.chosen = best_candidate(tried),
                             .tried = tried,
                             .gpu = "RTX 3060",
                             .display_height = 1080,
                             .models = calibration_models(Tier::Normal),
                             .engine = "tensorrt"};
  const auto followed = followed_chain(c, {.gpu = "RTX 3060", .display_height = 1080, .tensorrt = true});
  REQUIRE(followed);
  CHECK(followed->height == 480);
  CHECK(followed->span_height == 1080);
  // RIFE seul recommandé : le modèle de SPAN reste connu pour la priorité Image.
  const auto rife_only = search([](const C& chain) { return !(chain.span && chain.rife); }, 720, true);
  MachineCalibration r = c;
  r.chosen = best_candidate(rife_only);
  r.tried = rife_only;
  r.models = calibration_models(Tier::Quality);
  const auto back = followed_chain(r, {.gpu = "RTX 3060", .display_height = 1080, .tensorrt = true});
  REQUIRE(back);
  CHECK_FALSE(back->span);
  CHECK(back->span_height == 1080);
  CHECK(back->span_model == kQualitySpanModel);
}

TEST_CASE("C07-05 : le rejeu de justesse compte les pertes par image visée", "[calibration-machine]") {
  // Tient, avec 0,3 % des images visées perdues (plus de la moitié des 0,5 % permis) : rejoué.
  CandidateTrial close{.chain = {.span = true, .rife = true, .height = 720},
                       .holds = true,
                       .lost_percent = 0.25,
                       .fps = 47.9,
                       .target_fps = 48.0,
                       .lost_per_image = 0.003};
  CHECK(needs_second_run(close));
  close.lost_per_image = 0.001;  // une image sur mille : net
  CHECK_FALSE(needs_second_run(close));
  // La recherche le rejoue : 0,4 % des 480 images visées en dix secondes à 60 Hz (deux pertes) tient de
  // justesse ; avant, il fallait 1 % des présentations, au-delà de ce que le verdict laisse tenir.
  int played = 0;
  const TrialPlayer play = [&](const ChainCandidate& chain, int,
                               bool) -> std::expected<BenchReading, ajp::Refusal> {
    BenchReading reading{.gpu = "RTX 4070 Ti", .driver = "581", .engine = "tensorrt"};
    reading.run = {.presents = 600, .lag_p95_us = 4000, .period_us = 16'667};
    const double fps = chain.rife ? 48.0 : 24.0;
    reading.interval_mean_us = static_cast<std::int64_t>(1e6 / fps);
    reading.new_frames = static_cast<std::int64_t>(fps * 10.0);
    if (chain.rife && chain.height == 1080) {
      ++played;
      reading.run.dropped = 2;
    }
    return reading;
  };
  const auto trials = run_calibration_trials(play, std::stop_token{}, {}, 1080);
  REQUIRE(trials);
  CHECK(played == 2);
}
