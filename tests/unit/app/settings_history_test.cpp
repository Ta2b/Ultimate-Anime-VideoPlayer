// Annuler / rétablir du panneau (settings_history.h) : ce que le lecteur change seul n'y entre pas.
#include "app/play/settings_history.h"

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

TEST_CASE("Annuler : un volume changé aux touches n'est ni compté ni rejoué (B23-01)", "[app][settings]") {
  PlaySettings running;
  running.volume = 0.2F;
  SettingsHistory history;
  history.reset(running);
  PlaySettings draft = running;
  PlaySettings seen = running;

  draft.sharpness = 40;  // un réglage du panneau : une entrée d'annulation, avec le volume du moment
  history.settle(draft);
  REQUIRE(history.can_undo());

  // Le volume monte à la molette, hors du panneau : le brouillon le suit (follow_running), pas l'historique.
  running.sharpness = 40;
  running.volume = 1.2F;
  const ExternalChange external = external_changes(seen, running, draft);
  CHECK(external.volume);
  draft.volume = running.volume;
  history.adopt(external, running);
  history.settle(draft);  // rien de nouveau : pas d'entrée de plus
  PlaySettings restored = history.undo(draft);
  CHECK(restored.sharpness == PlaySettings{}.sharpness);  // la netteté revient
  CHECK(restored.volume == 1.2F);                         // le volume reste celui d'aujourd'hui
  CHECK_FALSE(history.can_undo());                        // un seul « Annuler » suffit
  // Rétablir reprend la netteté, toujours au volume en cours.
  PlaySettings again = history.redo(restored);
  CHECK(again.sharpness == 40);
  CHECK(again.volume == 1.2F);
}

// C23-05 : un réglage discret pris hors du niveau « Personnalisé » passe en Personnalisé avant l'historique :
// une seule entrée, un seul « Annuler ».
TEST_CASE("Annuler : un réglage et son passage en Personnalisé ne font qu'une entrée (C23-05)",
          "[app][settings]") {
  PlaySettings s;
  s.preset = Preset::Auto;
  SettingsHistory history;
  history.reset(s);
  s.restore_lines = !s.restore_lines;
  s.preset = Preset::Custom;  // le panneau bascule dans la même image, avant settle
  history.settle(s);
  history.settle(s);  // l'image suivante : rien de nouveau
  REQUIRE(history.can_undo());
  const PlaySettings back = history.undo(s);
  CHECK(back.preset == Preset::Auto);
  CHECK(back.restore_lines == PlaySettings{}.restore_lines);
  CHECK_FALSE(history.can_undo());
}

TEST_CASE("Annuler : un réglage que le panneau a lui-même fait reste annulable (B23-01)", "[app][settings]") {
  PlaySettings running;
  SettingsHistory history;
  history.reset(running);
  PlaySettings draft = running;
  draft.audio_delay_ms = 120;  // le curseur du délai
  history.settle(draft);
  // La part directe l'applique : running le rejoint, ce n'est pas un changement venu d'ailleurs.
  PlaySettings applied = running;
  applied.audio_delay_ms = 120;
  const ExternalChange external = external_changes(running, applied, draft);
  CHECK_FALSE(external.any());
  history.adopt(external, applied);
  CHECK(history.undo(draft).audio_delay_ms == 0);
}

TEST_CASE("Annuler : les états posés d'avant gardent le délai des touches et les statistiques",
          "[app][settings]") {
  PlaySettings running;
  SettingsHistory history;
  history.reset(running);
  PlaySettings draft = running;
  draft.deband = !draft.deband;
  history.settle(draft);
  PlaySettings later = running;
  later.deband = draft.deband;
  later.audio_delay_ms = -80;     // touche « - »
  later.subtitle_delay_ms = 300;  // touche « x »
  later.stats.all = true;         // Tab
  const ExternalChange external = external_changes(draft, later, draft);
  CHECK(external.audio_delay);
  CHECK(external.subtitle_delay);
  CHECK(external.stats);
  draft.audio_delay_ms = later.audio_delay_ms;
  draft.subtitle_delay_ms = later.subtitle_delay_ms;
  draft.stats = later.stats;
  history.adopt(external, later);
  history.settle(draft);
  const PlaySettings back = history.undo(draft);
  CHECK(back.deband == running.deband);
  CHECK(back.audio_delay_ms == -80);
  CHECK(back.subtitle_delay_ms == 300);
  CHECK(back.stats.all);
}

TEST_CASE("Annuler : un glissement compte pour un changement, la pile est bornée", "[app][settings]") {
  PlaySettings running;
  SettingsHistory history;
  history.reset(running);
  PlaySettings draft = running;
  for (int i = 1; i <= 150; ++i) {
    draft.sharpness = i % 100;
    history.settle(draft);
    history.settle(draft);  // le même état posé deux fois : une seule entrée
  }
  int steps = 0;
  while (history.can_undo()) {
    draft = history.undo(draft);
    ++steps;
  }
  CHECK(steps == 100);
}

}  // namespace
}  // namespace ajp::app::play
