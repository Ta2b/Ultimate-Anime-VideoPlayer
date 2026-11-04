#include "app/play/machine_calibration.h"

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;

namespace {

// Joue la recherche contre une machine qui tient ce que holds accepte ; rend les essais faits.
template <typename Holds>
std::vector<CandidateTrial> search(int display_height, Holds holds) {
  std::vector<CandidateTrial> tried;
  while (const auto chain = next_candidate(tried, display_height)) {
    REQUIRE(tried.size() < 20);
    tried.push_back({.chain = *chain, .holds = holds(*chain)});
  }
  return tried;
}

}  // namespace

TEST_CASE("les paliers s'arrêtent à la moitié de l'écran", "[calibration-machine]") {
  CHECK(calibration_heights(1080) == std::vector<int>{540, 480, 360});
  CHECK(calibration_heights(1440) == std::vector<int>{720, 540, 480, 360});
  CHECK(calibration_heights(2160) == std::vector<int>{1080, 900, 720, 540, 480, 360});
  CHECK(calibration_heights(600) == std::vector<int>{360});
}

TEST_CASE("la calibration part du plus haut palier permis avec SPAN + RIFE et descend",
          "[calibration-machine]") {
  // 1440p, tout tient : 720p + SPAN + RIFE d'emblée, puis RIFE seul pour comparer.
  auto tried = search(1440, [](const ChainCandidate&) { return true; });
  REQUIRE(tried.size() == 2);
  CHECK(tried[0].chain == ChainCandidate{.span = true, .rife = true, .height = 720});
  CHECK(tried[1].chain == ChainCandidate{.span = false, .rife = true, .height = 0});
  CHECK(best_candidate(tried) == tried[0].chain);
  // 1080p : départ à 540p.
  CHECK(search(1080, [](const ChainCandidate&) { return true; })[0].chain.height == 540);
  // 1440p, 540p tient : 720, 540, puis RIFE seul.
  tried = search(1440, [](const ChainCandidate& c) { return c.height <= 540; });
  CHECK(tried.size() == 3);
  CHECK(best_candidate(tried) == ChainCandidate{.span = true, .rife = true, .height = 540});
}

TEST_CASE("la calibration lâche RIFE avant SPAN, puis tout", "[calibration-machine]") {
  // Rien avec RIFE : SPAN seul, du haut vers le bas.
  auto tried = search(1440, [](const ChainCandidate& c) { return c.span && !c.rife && c.height <= 540; });
  CHECK(best_candidate(tried) == ChainCandidate{.span = true, .rife = false, .height = 540});
  // SPAN nulle part : RIFE seul.
  tried = search(1080, [](const ChainCandidate& c) { return !c.span; });
  CHECK(best_candidate(tried) == ChainCandidate{.span = false, .rife = true, .height = 0});
  // Rien ne tient : 3 paliers par famille SPAN sur un écran 1080p, puis RIFE seul ; ni l'un ni l'autre.
  tried = search(1080, [](const ChainCandidate&) { return false; });
  CHECK(tried.size() == 7);
  CHECK_FALSE(best_candidate(tried).span);
  CHECK_FALSE(best_candidate(tried).rife);
}

TEST_CASE("les réglages d'une chaîne : Personnalisé, Hermite, RIFE 4.22 lite en auto Rapide",
          "[calibration-machine]") {
  PlaySettings base;
  base.true_hdr_contrast = 150;
  const PlaySettings s = candidate_settings({.span = true, .rife = true, .height = 540}, base);
  CHECK(s.preset == Preset::Custom);
  CHECK(s.upscale);
  CHECK(s.work_height == 540);
  CHECK(s.downscale == Downscale::Hermite);
  CHECK(s.rife_model == "rife_v4.22_lite.onnx");
  CHECK(s.rife_auto);
  CHECK(s.auto_free);
  CHECK(s.rife_factor == 2);
  CHECK(s.true_hdr_contrast == 150);  // le reste des réglages de l'utilisateur
  const PlaySettings plain = candidate_settings({.span = false, .rife = false, .height = 0}, base);
  CHECK_FALSE(plain.upscale);
  CHECK_FALSE(plain.rife_auto);
  CHECK(plain.rife_factor == 0);
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
                                  .target_fps = 48.0,
                                  .fp8 = true}},
                       .gpu = "RTX 4070 Ti",
                       .date = "2026-11-01",
                       .display_height = 1440};
  const auto read = parse_calibration(format_calibration(c));
  REQUIRE(read);
  CHECK(read->chosen == c.chosen);
  REQUIRE(read->tried.size() == 2);
  CHECK(read->tried[1].holds);
  CHECK(read->tried[0].lost_percent == 12.5);
  CHECK(read->tried[1].fps == 47.9);
  CHECK(read->tried[1].low_fps == 41.0);
  CHECK(read->tried[1].target_fps == 48.0);
  CHECK(read->tried[1].fp8);
  CHECK_FALSE(read->tried[0].fp8);  // absent du fichier (et des calibrations d'avant) : FP16
  CHECK(read->gpu == "RTX 4070 Ti");
  CHECK(read->display_height == 1440);
  CHECK_FALSE(parse_calibration("version = 1\n"));
}

TEST_CASE("un essai joué en FP8 garde le cran Performance de SPAN, la recommandation aussi",
          "[calibration-machine]") {
  PlaySettings base;
  base.span_precision = PrecisionTier::Balanced;
  const CandidateTrial fp8{.chain = {.span = true, .rife = true, .height = 540}, .holds = true, .fp8 = true};
  const CandidateTrial fp16{.chain = {.span = true, .rife = false, .height = 720}, .holds = true};
  CHECK(trial_settings(fp8, base).span_precision == PrecisionTier::Performance);
  CHECK(trial_settings(fp8, base).work_height == 540);
  CHECK(trial_settings(fp16, base).span_precision == PrecisionTier::Balanced);
  const MachineCalibration c{.chosen = fp8.chain, .tried = {fp16, fp8}};
  CHECK(chosen_settings(c, base).span_precision == PrecisionTier::Performance);
  const MachineCalibration other{.chosen = fp16.chain, .tried = {fp16, fp8}};
  CHECK(chosen_settings(other, base).span_precision == PrecisionTier::Balanced);
}
