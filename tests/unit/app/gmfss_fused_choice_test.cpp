// GMFSS fusionné : quand la lecture l'essaie (gmfss_fused_choice.h), le réglage « gmfss_fused » de play.toml,
// ce que son changement en cours de lecture coûte, et les tranches de caractéristiques (gmfss_features.h).
#include "app/play/gmfss_fused_choice.h"

#include <catch2/catch_test_macros.hpp>

#include "app/play/gmfss_features.h"
#include "app/play/settings.h"

namespace ajp::app::play {

TEST_CASE("GMFSS fusionné : essayé avec GMFSS, avec la raison de chaque renoncement", "[app][gmfss_fused]") {
  PlaySettings gmfss;
  gmfss.motion_engine = MotionEngine::Gmfss;
  gmfss.gmfss_fused = true;
  CHECK(gmfss_fused_declined(gmfss, true).empty());
  PlaySettings off = gmfss;
  off.gmfss_fused = false;
  CHECK(gmfss_fused_declined(off, true) == "réglage éteint");
  // Un moteur qui ne le fait pas (Vulkan, Core ML…) : la paire entière, sans erreur.
  CHECK(gmfss_fused_declined(gmfss, false) == "ce moteur ne le fait pas");
  // Les crans FP8 et FP32 chargent une autre paire (« _s05fp8 », « _s05q ») que la découpe ne connaît pas.
  PlaySettings fast = gmfss;
  fast.gmfss_precision = PrecisionTier::Performance;
  CHECK(gmfss_fused_declined(fast, true) == "cran de précision hors Équilibre");
  PlaySettings quality = gmfss;
  quality.gmfss_precision = PrecisionTier::Quality;
  CHECK(gmfss_fused_declined(quality, true) == "cran de précision hors Équilibre");
  // GMFSS Union a la même paire : essayé aussi, la reconnaissance du graphe tranche.
  PlaySettings ultra = gmfss;
  ultra.motion_engine = MotionEngine::GmfssUnion;
  CHECK(gmfss_fused_declined(ultra, true).empty());
  CHECK_FALSE(PlaySettings{}.gmfss_fused);  // éteint par défaut (docs/gmfss-fusionne.md)
  // Les autres moteurs du mouvement ne sont pas coupés ici.
  PlaySettings rife;
  rife.gmfss_fused = true;
  rife.motion_engine = MotionEngine::Rife;
  CHECK(gmfss_fused_declined(rife, true) == "moteur du mouvement autre que GMFSS");
  PlaySettings drba;
  drba.gmfss_fused = true;
  drba.motion_engine = MotionEngine::DistilDrba;
  CHECK_FALSE(gmfss_fused_declined(drba, true).empty());
}

TEST_CASE("GMFSS fusionné : le réglage voyage dans play.toml et recharge la lecture",
          "[app][gmfss_fused][settings]") {
  PlaySettings s;
  s.gmfss_fused = !PlaySettings{}.gmfss_fused;
  const auto back = parse_settings(format_settings(s));
  REQUIRE(back);
  CHECK(back->gmfss_fused == s.gmfss_fused);
  CHECK(*back == s);
  // Un fichier d'avant sans la clé garde le défaut.
  const auto old_file = parse_settings("[interpolation]\nkeep_poses = true\n");
  REQUIRE(old_file);
  CHECK(old_file->gmfss_fused == PlaySettings{}.gmfss_fused);
  const auto on_file = parse_settings("[interpolation]\ngmfss_fused = true\n");
  REQUIRE(on_file);
  CHECK(on_file->gmfss_fused);
  const auto off_file = parse_settings("[interpolation]\ngmfss_fused = false\n");
  REQUIRE(off_file);
  CHECK_FALSE(off_file->gmfss_fused);

  // Le changer demande de rouvrir la lecture (les moteurs ne sont pas les mêmes), à condition d'interpoler
  // avec GMFSS.
  PlaySettings running;
  running.rife_factor = 3;
  running.motion_engine = MotionEngine::Gmfss;
  running.gmfss_fused = true;
  PlaySettings wanted = running;
  wanted.gmfss_fused = false;
  const Change change = change_between(running, wanted, true);
  CHECK(change.cost == ChangeCost::Reopen);
  CHECK(change.reason.find("paire GMFSS entière") != std::string::npos);
  // Avec RIFE, le réglage GMFSS ne coûte rien.
  running.motion_engine = MotionEngine::Rife;
  wanted.motion_engine = MotionEngine::Rife;
  CHECK(change_between(running, wanted, true).cost != ChangeCost::Reopen);
  running.rife_factor = 0;
  wanted.rife_factor = 0;
  running.motion_engine = MotionEngine::Gmfss;
  wanted.motion_engine = MotionEngine::Gmfss;
  CHECK(change_between(running, wanted, true).cost != ChangeCost::Reopen);
}

TEST_CASE("GMFSS fusionné : les caractéristiques d'une image servent à deux paires de suite",
          "[app][gmfss_fused]") {
  const auto frame = [](std::uint32_t n, std::uint32_t epoch = 0) {
    return GmfssFrameKey{.id = SourceId{.value = n}, .epoch = epoch};
  };
  GmfssFeatureSlots slots;
  // Première paire (1, 2) : rien n'est gardé, les deux sont calculées ; les tranches sont celles des tampons
  // d'image (a dans 0, b dans 1).
  CHECK_FALSE(slots.reuse(0, frame(1)));
  slots.fill(0, frame(1));
  CHECK_FALSE(slots.reuse(1, frame(2)));
  slots.fill(1, frame(2));
  // Paire suivante (2, 3) : b de la première est a de celle-ci, dans la même tranche ; 3 prend la tranche
  // de 1.
  CHECK(slots.reuse(1, frame(2)));
  CHECK_FALSE(slots.reuse(0, frame(3)));
  slots.fill(0, frame(3));
  // Encore (3, 4) : 3 est là, 4 remplace 2.
  CHECK(slots.reuse(0, frame(3)));
  CHECK_FALSE(slots.reuse(1, frame(4)));
  slots.fill(1, frame(4));
  CHECK(slots.computed() == 4);
  CHECK(slots.reused() == 2);
  // Une paire sautée (rien pour 4 puis 5) : la paire (5, 6) refait 5 et 6, jamais une tranche d'une autre
  // image.
  CHECK_FALSE(slots.reuse(0, frame(5)));
  slots.fill(0, frame(5));
  CHECK_FALSE(slots.reuse(1, frame(6)));
  // Une tranche dont le remplissage n'a pas fini (erreur) ne passe pour personne.
  CHECK_FALSE(slots.reuse(1, frame(6)));
  slots.fill(1, frame(6));
  // Une autre époque (seek) : la même identité ne suffit pas.
  CHECK_FALSE(slots.reuse(1, frame(6, 1)));
  CHECK_FALSE(slots.reuse(1, frame(6)));
  slots.fill(1, frame(6));
  slots.clear();
  CHECK_FALSE(slots.reuse(1, frame(6)));
}

}  // namespace ajp::app::play
