// DistilDRBA fusionné : quand la lecture l'essaie (drba_fused_choice.h), le réglage « drba_fused » de
// play.toml et ce que son changement en cours de lecture coûte.
#include "app/play/drba_fused_choice.h"

#include <catch2/catch_test_macros.hpp>

#include "app/play/settings.h"

namespace ajp::app::play {

TEST_CASE("DistilDRBA fusionné : essayé par défaut avec DistilDRBA, avec la raison de chaque renoncement",
          "[app][drba_fused]") {
  PlaySettings drba;
  drba.motion_engine = MotionEngine::DistilDrba;
  CHECK(drba.drba_fused);  // allumé par défaut
  CHECK(drba_fused_declined(drba, true).empty());

  PlaySettings off = drba;
  off.drba_fused = false;
  CHECK(drba_fused_declined(off, true) == "réglage éteint");
  // Un moteur qui ne le fait pas (Vulkan, Core ML…) : le DistilDRBA entier, sans erreur.
  CHECK(drba_fused_declined(drba, false) == "ce moteur ne le fait pas");
  // Les crans FP8 et FP32 chargent un autre graphe (« _fp8 », « _fp32 ») que la découpe ne connaît pas.
  PlaySettings fast = drba;
  fast.motion_precision = PrecisionTier::Performance;
  CHECK_FALSE(drba_fused_declined(fast, true).empty());
  PlaySettings quality = drba;
  quality.motion_precision = PrecisionTier::Quality;
  CHECK_FALSE(drba_fused_declined(quality, true).empty());
  // Les autres moteurs du mouvement ne sont pas coupés ici.
  PlaySettings rife;
  rife.motion_engine = MotionEngine::Rife;
  CHECK(drba_fused_declined(rife, true) == "moteur du mouvement autre que DistilDRBA");
  PlaySettings gmfss;
  gmfss.motion_engine = MotionEngine::Gmfss;
  CHECK_FALSE(drba_fused_declined(gmfss, true).empty());
}

TEST_CASE("DistilDRBA fusionné : le réglage voyage dans play.toml et recharge la lecture",
          "[app][drba_fused][settings]") {
  PlaySettings s;
  s.drba_fused = false;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK_FALSE(back->drba_fused);
  CHECK(*back == s);
  // Un fichier d'avant sans la clé garde le défaut.
  const auto old_file = parse_settings("[interpolation]\nkeep_poses = true\n");
  REQUIRE(old_file);
  CHECK(old_file->drba_fused);
  const auto off_file = parse_settings("[interpolation]\ndrba_fused = false\n");
  REQUIRE(off_file);
  CHECK_FALSE(off_file->drba_fused);

  // Le changer demande de rouvrir la lecture (les moteurs ne sont pas les mêmes), à condition d'interpoler
  // avec DistilDRBA.
  PlaySettings running;
  running.rife_factor = 3;
  running.motion_engine = MotionEngine::DistilDrba;
  PlaySettings wanted = running;
  wanted.drba_fused = false;
  const Change change = change_between(running, wanted, true);
  CHECK(change.cost == ChangeCost::Reopen);
  CHECK(change.reason.find("whole DistilDRBA") != std::string::npos);
  // Avec RIFE, le réglage DistilDRBA ne coûte rien.
  running.motion_engine = MotionEngine::Rife;
  wanted.motion_engine = MotionEngine::Rife;
  CHECK(change_between(running, wanted, true).cost != ChangeCost::Reopen);
  wanted.rife_factor = 0;
  running.rife_factor = 0;
  CHECK(change_between(running, wanted, false).cost != ChangeCost::Reopen);
}

}  // namespace ajp::app::play
