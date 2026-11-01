#include "app/play/machine_calibration.h"

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;

namespace {

// Joue la recherche contre une machine qui tient tout ce que holds accepte ; rend les essais faits.
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

TEST_CASE("la calibration de la machine part de 720p + SPAN + RIFE et monte tant que ça tient",
          "[calibration-machine]") {
  const auto tried = search([](const ChainCandidate& c) { return c.span && c.rife && c.height <= 900; });
  REQUIRE(tried.size() == 3);  // 720, 900, 1080 (ne tient pas)
  CHECK(tried[0].chain == ChainCandidate{.span = true, .rife = true, .height = 720});
  CHECK(best_candidate(tried) == ChainCandidate{.span = true, .rife = true, .height = 900});
  // Tout tient : 1080p, rien de plus.
  CHECK(best_candidate(search([](const ChainCandidate&) { return true; })).height == 1080);
}

TEST_CASE("la calibration descend, puis lâche RIFE avant SPAN, puis tout", "[calibration-machine]") {
  // 540p avec RIFE tient : 720 non, 540 oui, on s'arrête.
  auto tried = search([](const ChainCandidate& c) { return c.height <= 540; });
  CHECK(tried.size() == 2);
  CHECK(best_candidate(tried) == ChainCandidate{.span = true, .rife = true, .height = 540});
  // Rien avec RIFE : SPAN seul, qui tient à 720p et 900p.
  tried = search([](const ChainCandidate& c) { return !c.rife && c.height <= 900; });
  CHECK(best_candidate(tried) == ChainCandidate{.span = true, .rife = false, .height = 900});
  // Ni SPAN + RIFE ni SPAN : RIFE seul.
  tried = search([](const ChainCandidate& c) { return !c.span; });
  CHECK(best_candidate(tried) == ChainCandidate{.span = false, .rife = true, .height = 0});
  // Rien ne tient : ni l'un ni l'autre.
  tried = search([](const ChainCandidate&) { return false; });
  CHECK(tried.size() == 7);  // 720, 540, 480 pour chaque famille SPAN, puis RIFE seul
  CHECK_FALSE(best_candidate(tried).span);
  CHECK_FALSE(best_candidate(tried).rife);
}

TEST_CASE("les réglages d'une chaîne : Personnalisé, Hermite, RIFE 4.22 lite en auto Rapide",
          "[calibration-machine]") {
  PlaySettings base;
  base.true_hdr = true;
  const PlaySettings s = candidate_settings({.span = true, .rife = true, .height = 540}, base);
  CHECK(s.preset == Preset::Custom);
  CHECK(s.upscale);
  CHECK(s.work_height == 540);
  CHECK(s.downscale == Downscale::Hermite);
  CHECK(s.rife_model == "rife_v4.22_lite.onnx");
  CHECK(s.rife_auto);
  CHECK(s.auto_free);
  CHECK(s.rife_factor == 2);
  CHECK(s.true_hdr);  // le reste des réglages de l'utilisateur
  const PlaySettings plain = candidate_settings({.span = false, .rife = false, .height = 0}, base);
  CHECK_FALSE(plain.upscale);
  CHECK_FALSE(plain.rife_auto);
  CHECK(plain.rife_factor == 0);
}
