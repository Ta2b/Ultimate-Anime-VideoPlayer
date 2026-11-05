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

TEST_CASE("la calibration monte SPAN seul, puis descend SPAN + RIFE depuis son plus haut palier",
          "[calibration-machine]") {
  using C = ChainCandidate;
  // 1440p, tout tient : SPAN seul 360 → 720, puis SPAN + RIFE à 720 d'emblée.
  auto tried = search(1440, [](const ChainCandidate&) { return true; });
  REQUIRE(tried.size() == 5);
  CHECK(tried[0].chain == C{.span = true, .rife = false, .height = 360});
  CHECK(tried[3].chain == C{.span = true, .rife = false, .height = 720});
  CHECK(tried[4].chain == C{.span = true, .rife = true, .height = 720});
  CHECK(best_candidate(tried) == tried[4].chain);
  // 1080p : SPAN seul s'arrête à 540p.
  tried = search(1080, [](const ChainCandidate&) { return true; });
  REQUIRE(tried.size() == 4);
  CHECK(tried.back().chain == C{.span = true, .rife = true, .height = 540});
  // 1440p, 540p tient au plus : SPAN seul s'arrête au premier échec (720), SPAN + RIFE part de 540.
  tried = search(1440, [](const ChainCandidate& c) { return c.height <= 540; });
  REQUIRE(tried.size() == 5);
  CHECK(tried[3].chain == C{.span = true, .rife = false, .height = 720});
  CHECK(tried[4].chain == C{.span = true, .rife = true, .height = 540});
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 540});
}

TEST_CASE("la calibration garde SPAN seul quand RIFE ne tient pas, RIFE seul en recours",
          "[calibration-machine]") {
  using C = ChainCandidate;
  // RIFE ne tient nulle part : SPAN + RIFE descend jusqu'au bout, SPAN seul est gardé.
  auto tried = search(1440, [](const ChainCandidate& c) { return c.span && !c.rife && c.height <= 540; });
  CHECK(tried.size() == 4 + 3);
  CHECK(best_candidate(tried) == C{.span = true, .rife = false, .height = 540});
  // SPAN haut sans RIFE passe avant SPAN bas avec RIFE : 720p seul est recommandé, 480p + RIFE reste au
  // tableau.
  tried = search(1440, [](const ChainCandidate& c) { return !c.rife || c.height <= 480; });
  CHECK(tried.back().chain == C{.span = true, .rife = true, .height = 480});
  CHECK(best_candidate(tried) == C{.span = true, .rife = false, .height = 720});
  // SPAN nulle part : RIFE seul, sans SPAN + RIFE.
  tried = search(1080, [](const ChainCandidate& c) { return !c.span; });
  REQUIRE(tried.size() == 2);
  CHECK(best_candidate(tried) == C{.span = false, .rife = true, .height = 0});
  // Rien ne tient : SPAN seul au plus léger, puis RIFE seul ; ni l'un ni l'autre.
  tried = search(1080, [](const ChainCandidate&) { return false; });
  CHECK(tried.size() == 2);
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
  CHECK(s.model.empty());             // SPAN par défaut (Performance Sharp) en FP16
  CHECK(s.span_precision == PrecisionTier::Balanced);
  const PlaySettings fp8 = candidate_settings({.span = true, .rife = true, .height = 540}, base, true);
  CHECK(fp8.model == kCalibrationSpanFp8);
  CHECK(fp8.span_precision == PrecisionTier::Performance);
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
                                  .target_fps = 48.0}},
                       .gpu = "RTX 4070 Ti",
                       .date = "2026-11-01",
                       .display_height = 1440,
                       .span_fp8 = true};
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
  CHECK(read->span_fp8);
  CHECK_FALSE(parse_calibration("version = 1\n"));
}
