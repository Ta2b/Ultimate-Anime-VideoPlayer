#include "app/play/machine_calibration.h"

#include <filesystem>

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

TEST_CASE("la calibration préfère RIFE seul à SPAN seul, SPAN seul quand RIFE ne tient pas",
          "[calibration-machine]") {
  using C = ChainCandidate;
  // RIFE ne tient nulle part : SPAN + RIFE descend jusqu'au bout, RIFE seul est essayé, SPAN seul est gardé.
  auto tried = search(1440, [](const ChainCandidate& c) { return c.span && !c.rife && c.height <= 540; });
  CHECK(tried.size() == 4 + 3 + 1);
  CHECK(best_candidate(tried) == C{.span = true, .rife = false, .height = 540});
  // SPAN + RIFE nulle part mais RIFE seul tient : la fluidité passe avant SPAN seul (P3.11).
  tried = search(1440, [](const ChainCandidate& c) { return !(c.span && c.rife); });
  CHECK(tried.back().chain == C{.span = false, .rife = true, .height = 0});
  CHECK(best_candidate(tried) == C{.span = false, .rife = true, .height = 0});
  // La fluidité d'abord (P3.11) : 480p + RIFE passe avant 720p seul, qui reste au tableau.
  tried = search(1440, [](const ChainCandidate& c) { return !c.rife || c.height <= 480; });
  CHECK(tried.back().chain == C{.span = true, .rife = true, .height = 480});
  CHECK(best_candidate(tried) == C{.span = true, .rife = true, .height = 480});
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
  CHECK_FALSE(read->on_battery);
  CHECK_FALSE(parse_calibration("version = 1\n"));
  // Rejoués dans les deux sens (P3.12) : l'échec, et le succès de justesse ; pas le succès net.
  CHECK(needs_second_run(c.tried[0]));
  CHECK_FALSE(needs_second_run(c.tried[1]));
  CandidateTrial close = c.tried[1];
  close.lost_percent = 1.6;
  CHECK(needs_second_run(close));
  // Mesurée sur batterie : relue, mais l'Auto ne la suit pas.
  c.on_battery = true;
  const auto dir = std::filesystem::temp_directory_path() / "ajp-calibration-battery";
  std::filesystem::create_directories(dir);
  REQUIRE(save_calibration(calibration_file(dir), c));
  CHECK(load_calibration(calibration_file(dir))->on_battery);
  CHECK_FALSE(calibrated_chain(dir, "RTX 4070 Ti", 1440));
  std::filesystem::remove_all(dir);
}

TEST_CASE("la durée annoncée et la barre qui avance pendant une construction (P3.13)",
          "[calibration-machine]") {
  const int plain = calibration_minutes(1440, false, false);
  CHECK(plain >= 3);
  CHECK(plain <= 8);
  CHECK(calibration_minutes(1440, true, false) > plain + 10);  // TensorRT construit ses moteurs
  CHECK(calibration_minutes(1440, true, true) > calibration_minutes(1440, true, false));
  CHECK(creeping_fraction(0.05F, 0.0) == 0.05F);
  CHECK(creeping_fraction(0.05F, 300.0) > 0.1F);   // cinq minutes de construction : la barre a bougé
  CHECK(creeping_fraction(0.05F, 1e6) <= 0.131F);  // sans rattraper l'essai suivant
  CHECK(creeping_fraction(0.95F, 1e6) <= 0.97F);
}
