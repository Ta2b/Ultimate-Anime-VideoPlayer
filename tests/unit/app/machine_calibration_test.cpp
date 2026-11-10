#include "app/play/machine_calibration.h"

#include <filesystem>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;

namespace {

// Joue la recherche contre une machine qui tient ce que holds accepte ; rend les essais faits.
template <typename Holds>
std::vector<CandidateTrial> search(Holds holds) {
  std::vector<CandidateTrial> tried;
  while (const auto chain = next_candidate(tried)) {
    REQUIRE(tried.size() < 20);
    tried.push_back({.chain = *chain, .holds = holds(*chain)});
  }
  return tried;
}

}  // namespace

TEST_CASE("les paliers montent jusqu'à la 1080p de la séquence, sur tout écran (A49-02)",
          "[calibration-machine]") {
  CHECK(calibration_heights() == std::vector<int>{1080, 720, 540, 480, 360});
}

TEST_CASE("les modèles de la calibration sont ceux du niveau, jamais en dessous (A49-01, A38-02)",
          "[calibration-machine]") {
  for (const Tier tier : {Tier::Fast, Tier::Ultra}) {
    CHECK(calibration_models(tier).span.find("Balanced_SPANF3_b8f64") != std::string::npos);
    CHECK(calibration_models(tier).rife == "rife_v4.25.onnx");
  }
  for (const Tier tier : {Tier::Economy, Tier::Light, Tier::Normal}) {
    CHECK(calibration_models(tier).span.empty());  // le SPAN par défaut, Performance Sharp
    CHECK(calibration_models(tier).rife == "rife_v4.22_lite.onnx");
  }
}

TEST_CASE("la calibration monte SPAN seul, puis descend SPAN + RIFE depuis son plus haut palier",
          "[calibration-machine]") {
  using C = ChainCandidate;
  // Tout tient : SPAN seul 360 → 1080, puis SPAN + RIFE à 1080 d'emblée ; la source 1080p est gardée.
  auto tried = search([](const ChainCandidate&) { return true; });
  REQUIRE(tried.size() == 6);
  CHECK(tried[0].chain == C{.span = true, .rife = false, .height = 360});
  CHECK(tried[4].chain == C{.span = true, .rife = false, .height = 1080});
  CHECK(tried[5].chain == C{.span = true, .rife = true, .height = 1080});
  CHECK(best_candidate(tried) == tried[5].chain);
  // 540p tient au plus : SPAN seul s'arrête au premier échec (720), SPAN + RIFE part de 540.
  tried = search([](const ChainCandidate& c) { return c.height <= 540; });
  REQUIRE(tried.size() == 5);
  CHECK(tried[3].chain == C{.span = true, .rife = false, .height = 720});
  CHECK(tried[4].chain == C{.span = true, .rife = true, .height = 540});
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 540});
}

TEST_CASE("la calibration préfère RIFE seul à SPAN seul, SPAN seul quand RIFE ne tient pas",
          "[calibration-machine]") {
  using C = ChainCandidate;
  // RIFE ne tient nulle part : SPAN + RIFE descend jusqu'au bout, RIFE seul est essayé, SPAN seul est gardé.
  auto tried = search([](const ChainCandidate& c) { return c.span && !c.rife && c.height <= 540; });
  CHECK(tried.size() == 4 + 3 + 1);
  CHECK(best_candidate(tried) == C{.span = true, .rife = false, .height = 540});
  // SPAN + RIFE nulle part mais RIFE seul tient : la fluidité passe avant SPAN seul (P3.11).
  tried = search([](const ChainCandidate& c) { return !(c.span && c.rife); });
  CHECK(tried.back().chain == C{.span = false, .rife = true, .height = 0});
  CHECK(best_candidate(tried) == C{.span = false, .rife = true, .height = 0});
  // La fluidité d'abord (P3.11) : 480p + RIFE passe avant 1080p seul, qui reste au tableau.
  tried = search([](const ChainCandidate& c) { return !c.rife || c.height <= 480; });
  CHECK(tried.back().chain == C{.span = true, .rife = true, .height = 480});
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 480});
  // SPAN nulle part : RIFE seul, sans SPAN + RIFE.
  tried = search([](const ChainCandidate& c) { return !c.span; });
  REQUIRE(tried.size() == 2);
  CHECK(best_candidate(tried) == C{.span = false, .rife = true, .height = 0});
  // Rien ne tient : SPAN seul au plus léger, puis RIFE seul ; ni l'un ni l'autre.
  tried = search([](const ChainCandidate&) { return false; });
  CHECK(tried.size() == 2);
  CHECK_FALSE(best_candidate(tried).span);
  CHECK_FALSE(best_candidate(tried).rife);
}

TEST_CASE("les réglages d'une chaîne : Personnalisé, Hermite, modèles du niveau en FP16, auto Rapide",
          "[calibration-machine]") {
  PlaySettings base;
  base.true_hdr_peak = 800;
  const PlaySettings s = candidate_settings({.span = true, .rife = true, .height = 540}, base);
  CHECK(s.preset == Preset::Custom);
  CHECK(s.upscale);
  CHECK(s.work_height == 540);
  CHECK(s.downscale == Downscale::Hermite);
  CHECK(s.rife_model == "rife_v4.22_lite.onnx");
  CHECK(s.rife_auto);
  CHECK(s.auto_free);
  CHECK(s.rife_factor == 2);
  CHECK(s.true_hdr_peak == 800);  // le reste des réglages de l'utilisateur
  CHECK(s.model.empty());         // SPAN par défaut (Performance Sharp) en FP16
  CHECK(s.span_precision == PrecisionTier::Balanced);
  // Niveau Qualité : le b8f64 et RIFE 4.25, toujours en FP16 (A49-03 : plus de FP8 posé par la calibration).
  const PlaySettings fine =
      candidate_settings({.span = true, .rife = true, .height = 1080}, base, calibration_models(Tier::Fast));
  CHECK(fine.model == kQualitySpanModel);
  CHECK(fine.rife_model == "rife_v4.25.onnx");
  CHECK(fine.span_precision == PrecisionTier::Balanced);
  CHECK(fine.work_height == 1080);
  const PlaySettings plain = candidate_settings({.span = false, .rife = false, .height = 0}, base);
  CHECK_FALSE(plain.upscale);
  CHECK_FALSE(plain.rife_auto);
  CHECK(plain.rife_factor == 0);
}

TEST_CASE("un essai joue la chaîne que l'Auto calibré écrira, pas les réglages de l'utilisateur (A07-01)",
          "[calibration-machine]") {
  PlaySettings base;
  base.span_order = SpanOrder::Before;
  base.keep_poses = false;
  base.motion_precision = PrecisionTier::Performance;
  base.restore_lines = false;  // A07-06 : sans lui un essai SPAN sous 1080p jouerait la source sans SPAN
  const ChainCandidate chain{.span = true, .rife = true, .height = 720};
  const CalibrationModels models = calibration_models(Tier::Fast);
  const PlaySettings trial = candidate_settings(chain, base, models);
  CHECK(span_runs(trial, 1080));
  PlaySettings automatic;
  automatic.preset = Preset::Auto;
  const PlaySettings played = apply_preset(automatic,
                                           Machine{.tier = Tier::Fast,
                                                   .display_height = 1440,
                                                   .display_hz = 60.0,
                                                   .calibrated = CalibratedChain{.span = true,
                                                                                 .rife = true,
                                                                                 .height = chain.height,
                                                                                 .span_model = models.span,
                                                                                 .rife_model = models.rife}},
                                           ajp::core::SourceInfo{.width = 1920, .height = 1080, .fps = 24.0});
  CHECK(trial.upscale == played.upscale);
  CHECK(trial.work_height == played.work_height);
  CHECK(trial.downscale == played.downscale);
  CHECK(trial.model == played.model);
  CHECK(trial.span_precision == played.span_precision);
  CHECK(trial.span_order == played.span_order);
  CHECK(trial.motion_engine == played.motion_engine);
  CHECK(trial.motion_precision == played.motion_precision);
  CHECK(trial.rife_model == played.rife_model);
  CHECK(trial.keep_poses == played.keep_poses);
  CHECK(trial.auto_free == played.auto_free);
  CHECK(trial.rife_factor == played.rife_factor);  // 60 Hz sur 24 i/s : ×2, comme l'essai
}

TEST_CASE("calibration.toml relit les essais et la recommandation", "[calibration-machine]") {
  MachineCalibration c{.chosen = {.span = true, .rife = true, .height = 540},
                       .tried = {{.chain = {.span = true, .rife = true, .height = 720},
                                  .holds = false,
                                  .lost_percent = 12.5,
                                  .fps = 33.0,
                                  .target_fps = 48.0},
                                 {.chain = {.span = true, .rife = true, .height = 540},
                                  .holds = true,
                                  .lost_percent = 0.4,
                                  .fps = 47.9,
                                  .low_fps = 41.0,
                                  .target_fps = 48.0}},
                       .gpu = "RTX 4070 Ti",
                       .date = "2026-11-01",
                       .display_height = 1440,
                       .models = calibration_models(Tier::Fast)};
  const auto read = parse_calibration(format_calibration(c));
  REQUIRE(read);
  CHECK(read->chosen == c.chosen);
  REQUIRE(read->tried.size() == 2);
  CHECK(read->tried[1].holds);
  CHECK(read->tried[0].lost_percent == 12.5);
  CHECK(read->tried[1].fps == 47.9);
  CHECK(read->tried[1].low_fps == 41.0);
  CHECK(read->tried[1].target_fps == 48.0);
  CHECK(read->gpu == "RTX 4070 Ti");
  CHECK(read->display_height == 1440);
  CHECK(read->version == kCalibrationVersion);
  CHECK(read->models == c.models);
  CHECK_FALSE(read->on_battery);
  CHECK_FALSE(parse_calibration("version = 1\n"));
  // Rejoués dans les deux sens (P3.12) : l'échec, et le succès de justesse ; pas le succès net.
  CHECK(needs_second_run(c.tried[0]));
  CHECK_FALSE(needs_second_run(c.tried[1]));
  CandidateTrial close = c.tried[1];
  close.lost_percent = 1.6;
  CHECK(needs_second_run(close));
  // Sur secteur, à cette version : l'Auto la suit, avec ses modèles.
  const auto dir = std::filesystem::temp_directory_path() / "ajp-calibration-battery";
  std::filesystem::create_directories(dir);
  REQUIRE(save_calibration(calibration_file(dir), c));
  const auto followed = calibrated_chain(dir, "RTX 4070 Ti", 1440);
  REQUIRE(followed);
  CHECK(followed->height == 540);
  CHECK(followed->span_model == kQualitySpanModel);
  CHECK(followed->rife_model == "rife_v4.25.onnx");
  CHECK_FALSE(calibrated_chain(dir, "RTX 4090", 1440));  // une autre carte
  // Mesurée sur batterie : relue, mais l'Auto ne la suit pas.
  c.on_battery = true;
  REQUIRE(save_calibration(calibration_file(dir), c));
  CHECK(load_calibration(calibration_file(dir))->on_battery);
  CHECK_FALSE(calibrated_chain(dir, "RTX 4070 Ti", 1440));
  std::filesystem::remove_all(dir);
}

TEST_CASE("un calibration.toml d'avant la version 2 se relit mais l'Auto ne le suit plus",
          "[calibration-machine]") {
  // Tel qu'écrit par la 0.10 : essais bornés à la moitié de l'écran, SPAN FP8, RIFE 4.22 lite.
  constexpr std::string_view kOld = R"(version = 1
gpu = "RTX 4070 Ti"
date = "2026-11-01"
display_height = 1440
span_fp8 = true
on_battery = false

[chosen]
span = true
rife = true
height = 720

[[trials]]
span = true
rife = true
height = 720
holds = true
lost_percent = 0.4
fps = 47.9
low_fps = 41.0
target_fps = 48.0
)";
  const auto read = parse_calibration(kOld);
  REQUIRE(read);
  CHECK(read->version == 1);
  CHECK(read->chosen == ChainCandidate{.span = true, .rife = true, .height = 720});
  CHECK(read->models.span.empty());
  const auto dir = std::filesystem::temp_directory_path() / "ajp-calibration-v1";
  std::filesystem::create_directories(dir);
  REQUIRE(save_calibration(calibration_file(dir), *read));  // garde sa version : toujours ancienne
  CHECK(load_calibration(calibration_file(dir))->version == 1);
  CHECK_FALSE(calibrated_chain(dir, "RTX 4070 Ti", 1440));
  std::filesystem::remove_all(dir);
}

TEST_CASE("la durée annoncée et la barre qui avance pendant une construction (P3.13)",
          "[calibration-machine]") {
  const int plain = calibration_minutes(false, false);
  CHECK(plain >= 3);
  CHECK(plain <= 8);
  CHECK(calibration_minutes(true, false) > plain + 10);  // TensorRT construit ses moteurs
  CHECK(calibration_minutes(true, true) > calibration_minutes(true, false));
  CHECK(creeping_fraction(0.05F, 0.0) == 0.05F);
  CHECK(creeping_fraction(0.05F, 300.0) > 0.1F);   // cinq minutes de construction : la barre a bougé
  CHECK(creeping_fraction(0.05F, 1e6) <= 0.131F);  // sans rattraper l'essai suivant
  CHECK(creeping_fraction(0.95F, 1e6) <= 0.97F);
}
