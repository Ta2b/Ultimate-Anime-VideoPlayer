// RIFE fusionné : quand la lecture l'essaie (rife_fused_choice.h), le réglage « rife_fused » de play.toml et
// ce que son changement en cours de lecture coûte.
#include "app/play/rife_fused_choice.h"

#include <filesystem>
#include <optional>

#include <catch2/catch_test_macros.hpp>

#include "app/play/model_choice.h"
#include "app/play/settings.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {

TEST_CASE("RIFE fusionné : essayé par défaut, avec la raison de chaque renoncement", "[app][rife_fused]") {
  const PlaySettings defaults;
  CHECK(defaults.rife_fused);  // allumé par défaut
  CHECK(rife_fused_declined(defaults, true, kernels::kRifeInputChannels, defaults.motion_precision).empty());

  PlaySettings off;
  off.rife_fused = false;
  CHECK(rife_fused_declined(off, true, kernels::kRifeInputChannels, off.motion_precision) ==
        "réglage éteint");
  // Un moteur qui ne le fait pas (Vulkan, Core ML…) : le RIFE entier, sans erreur.
  CHECK(rife_fused_declined(defaults, false, kernels::kRifeInputChannels, defaults.motion_precision) ==
        "ce moteur ne le fait pas");
  // Les crans FP8 et FP32 chargent un autre graphe (« _fp8 », « _fp32 ») que la découpe ne connaît pas.
  PlaySettings fast;
  fast.motion_precision = PrecisionTier::Performance;
  CHECK_FALSE(rife_fused_declined(fast, true, kernels::kRifeInputChannels, fast.motion_precision).empty());
  PlaySettings quality;
  quality.motion_precision = PrecisionTier::Quality;
  CHECK_FALSE(
      rife_fused_declined(quality, true, kernels::kRifeInputChannels, quality.motion_precision).empty());
  // A31-06 : un cran dont le graphe manque retombe sur le modèle livré, que la découpe connaît ; c'est le
  // cran rendu qui décide, pas celui du réglage (aucun graphe FP8 de RIFE n'est publié).
  CHECK(rife_fused_declined(fast, true, kernels::kRifeInputChannels, PrecisionTier::Balanced).empty());
  CHECK(rife_fused_declined(quality, true, kernels::kRifeInputChannels, PrecisionTier::Balanced).empty());
  CHECK(model_tier(std::filesystem::path("m/rife_v4.25.onnx")) == PrecisionTier::Balanced);
  CHECK(model_tier(std::filesystem::path("m/rife_v4.25_fp8.onnx")) == PrecisionTier::Performance);
  CHECK(model_tier(std::filesystem::path("m/rife_v4.25_fp32.onnx")) == PrecisionTier::Quality);
  CHECK(model_tier(std::nullopt) == PrecisionTier::Balanced);
  // Les exports v2 (7 plans) et les autres moteurs du mouvement ne sont pas coupés.
  CHECK(rife_fused_declined(defaults, true, kernels::kRifeV2InputChannels, defaults.motion_precision) ==
        "modèle à entrée v2");
  PlaySettings gmfss;
  gmfss.motion_engine = MotionEngine::Gmfss;
  CHECK_FALSE(rife_fused_declined(gmfss, true, kernels::kRifeInputChannels, gmfss.motion_precision).empty());
}

TEST_CASE("RIFE fusionné : le réglage voyage dans play.toml et recharge la lecture",
          "[app][rife_fused][settings]") {
  PlaySettings s;
  s.rife_fused = false;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK_FALSE(back->rife_fused);
  CHECK(*back == s);
  // Un fichier d'avant sans la clé garde le défaut.
  const auto old_file = parse_settings("[interpolation]\nkeep_poses = true\n");
  REQUIRE(old_file);
  CHECK(old_file->rife_fused);
  const auto off_file = parse_settings("[interpolation]\nrife_fused = false\n");
  REQUIRE(off_file);
  CHECK_FALSE(off_file->rife_fused);

  // Le changer demande de rouvrir la lecture (les moteurs ne sont pas les mêmes), à condition d'interpoler
  // avec RIFE.
  PlaySettings running;
  running.rife_factor = 3;
  PlaySettings wanted = running;
  wanted.rife_fused = false;
  const Change change = change_between(running, wanted, true);
  CHECK(change.cost == ChangeCost::Reopen);
  CHECK(change.reason.find("whole RIFE") != std::string::npos);
  wanted.rife_factor = 0;
  running.rife_factor = 0;
  CHECK(change_between(running, wanted, false).cost != ChangeCost::Reopen);
}

}  // namespace ajp::app::play
