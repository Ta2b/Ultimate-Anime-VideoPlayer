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
  // Le cran ne se juge plus ici mais sur le modèle trouvé (drba_tier).
  PlaySettings fast = drba;
  fast.motion_precision = PrecisionTier::Performance;
  CHECK(drba_fused_declined(fast, true).empty());
  // Les autres moteurs du mouvement ne sont pas coupés ici.
  PlaySettings rife;
  rife.motion_engine = MotionEngine::Rife;
  CHECK(drba_fused_declined(rife, true) == "moteur du mouvement autre que DistilDRBA");
  PlaySettings apple;
  apple.motion_engine = MotionEngine::Apple;
  CHECK_FALSE(drba_fused_declined(apple, true).empty());
}

TEST_CASE("DistilDRBA : le cran rendu décide du graphe et du fusionné", "[app][drba_fused]") {
  const TierSupport none{};
  const TierSupport both{.performance = true, .quality = true};
  // Équilibre : le modèle livré, fusionné s'il s'essaie, sans un mot.
  CHECK(drba_tier(PrecisionTier::Balanced, true, both).fused);
  CHECK(drba_tier(PrecisionTier::Balanced, true, both).tier == PrecisionTier::Balanced);
  CHECK(drba_tier(PrecisionTier::Balanced, true, both).why.empty());
  CHECK_FALSE(drba_tier(PrecisionTier::Balanced, false, both).fused);
  // A31-06 : Performance sans graphe FP8 ou sans moteur FP8 (toujours sous Windows), Qualité sans graphe FP32
  // : le modèle livré, donc fusionné, au lieu du moteur entier plus lent qu'Équilibre.
  for (const PrecisionTier asked : {PrecisionTier::Performance, PrecisionTier::Quality}) {
    const DrbaTier plan = drba_tier(asked, true, none);
    CHECK(plan.tier == PrecisionTier::Balanced);
    CHECK(plan.fused);
    CHECK_FALSE(plan.why.empty());
  }
  // A31-07 : Performance avec le graphe FP8 présent : la chaîne fusionnée FP16 d'abord, plus rapide et plus
  // juste.
  const DrbaTier fast = drba_tier(PrecisionTier::Performance, true, both);
  CHECK(fast.tier == PrecisionTier::Balanced);
  CHECK(fast.fused);
  // Sans chaîne fusionnée (réglage éteint, moteur qui ne la fait pas), le graphe FP8 sert.
  const DrbaTier whole = drba_tier(PrecisionTier::Performance, false, both);
  CHECK(whole.tier == PrecisionTier::Performance);
  CHECK_FALSE(whole.fused);
  CHECK(whole.why.empty());
  // Qualité avec son graphe : le FP32 entier, jamais la chaîne coupée du modèle FP16.
  const DrbaTier best = drba_tier(PrecisionTier::Quality, true, both);
  CHECK(best.tier == PrecisionTier::Quality);
  CHECK_FALSE(best.fused);
  CHECK(drba_tier(PrecisionTier::Quality, false, none).tier == PrecisionTier::Balanced);
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
