#include "app/play/wizard.h"

#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;

TEST_CASE("l'assistant saute les écrans sans objet", "[wizard]") {
  CHECK(wizard_steps({.engine_choice = true, .hdr_display = true}).size() == 8);
  const auto plain = wizard_steps({.engine_choice = false, .hdr_display = false});
  REQUIRE(plain.size() == 6);
  CHECK(plain.front() == WizardStep::Welcome);
  CHECK(plain[1] == WizardStep::Chain);  // ni moteur ni HDR
  CHECK(plain.back() == WizardStep::Summary);
}

TEST_CASE("l'assistant avance, recule, passe et reprend en route", "[wizard]") {
  const WizardFacts facts{.engine_choice = true, .hdr_display = false};
  Wizard wizard = begin_wizard(facts, 0);
  CHECK(current(wizard) == WizardStep::Welcome);
  CHECK(is_first(wizard));
  previous(wizard);
  CHECK(current(wizard) == WizardStep::Welcome);
  next(wizard);
  CHECK(current(wizard) == WizardStep::Engines);
  next(wizard);
  skip(wizard);  // Image et fluidité passée
  CHECK(current(wizard) == WizardStep::Audio);  // HDR n'a pas lieu
  CHECK(progress_text(wizard) == "Étape 4 sur 7 : Audio");
  CHECK(wizard.skipped == std::vector<WizardStep>{WizardStep::Chain});
  // Fermé après l'écran 5 : on reprend à Langues.
  CHECK(current(begin_wizard(facts, 5)) == WizardStep::Languages);
  // Tout vu : le résumé.
  CHECK(current(begin_wizard(facts, 8)) == WizardStep::Summary);
  while (!is_last(wizard)) next(wizard);
  next(wizard);
  CHECK(current(wizard) == WizardStep::Summary);
}

TEST_CASE("Terminer écrit les choix et laisse intact ce qui est sauté", "[wizard]") {
  const WizardFacts facts{.engine_choice = true, .hdr_display = true};
  Wizard wizard = begin_wizard(facts, 0);
  wizard.preset = Preset::Quality;
  wizard.priority = Priority::Image;
  wizard.hdr = HdrChoice::Adaptive;
  wizard.listening = Listening::Headphones;
  wizard.update_mode = "auto";

  const ajp::core::Settings written = wizard_settings(wizard, {});
  CHECK(written.first_run_done);
  CHECK(written.first_run_step == kWizardSteps);

  PlaySettings play;
  play = wizard_play_settings(wizard, play);
  CHECK(play.listening == "headphones");
  CHECK(play.update_mode == "auto");
  CHECK(play.preset == Preset::Quality);
  CHECK(play.priority == Priority::Image);
  CHECK(play.hdr_native);
  CHECK_FALSE(play.true_hdr);

  // Écrans passés : Enceintes (rien d'écrit), Me prévenir (défaut), HDR inchangé, préréglage Auto.
  wizard.skipped = {WizardStep::Audio, WizardStep::Updates, WizardStep::Hdr, WizardStep::Chain};
  const ajp::core::Settings kept = wizard_settings(wizard, {});
  PlaySettings other;
  other.true_hdr = true;
  other = wizard_play_settings(wizard, other);
  CHECK(other.update_mode == "notify");
  CHECK(other.listening == "speakers");  // passé : Enceintes en rendu fidèle
  CHECK(other.true_hdr);
  CHECK(other.preset == Preset::Auto);
}

TEST_CASE("le résumé dit ce qui est choisi, sauté, et où le changer", "[wizard]") {
  Wizard wizard = begin_wizard({.engine_choice = true, .hdr_display = false}, 0);
  wizard.skipped = {WizardStep::Languages};
  wizard.listening = Listening::Cinema;
  const auto lines = wizard_summary(wizard);
  REQUIRE(lines.size() == 4);  // Chain, Audio, Languages, Updates (pas de HDR)
  CHECK(lines[0].contains("Général > Configuration"));
  CHECK(lines[1].contains("Home cinéma"));
  CHECK(lines[2].contains("sauté"));
  CHECK(lines[3].contains("me prévenir"));
  CHECK(listening_from_key("headphones") == Listening::Headphones);
  CHECK(listening_from_key("n'importe quoi") == Listening::Unsaid);
}
