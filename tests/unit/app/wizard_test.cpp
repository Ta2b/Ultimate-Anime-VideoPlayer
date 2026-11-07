#include "app/play/wizard.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "french_ui.h"

using namespace ajp::app::play;

TEST_CASE("l'assistant saute les écrans sans objet", "[wizard]") {
  ajp::app::play::FrenchUi french;
  // Sans les rangs 4 (HDR, il suit le système) et 5 (grade adaptatif), retirés.
  const auto all = wizard_steps({.engine_choice = true});
  CHECK(all.size() == 6);
  CHECK(std::ranges::find(all, WizardStep::Hdr) == all.end());
  const auto plain = wizard_steps({.engine_choice = false});
  REQUIRE(plain.size() == 5);
  CHECK(plain.front() == WizardStep::Welcome);
  CHECK(plain[1] == WizardStep::Chain);  // pas de moteur à choisir
  CHECK(plain.back() == WizardStep::Summary);
}

TEST_CASE("l'assistant avance, recule, passe et reprend en route", "[wizard]") {
  ajp::app::play::FrenchUi french;
  const WizardFacts facts{.engine_choice = true};
  Wizard wizard = begin_wizard(facts, 0);
  CHECK(current(wizard) == WizardStep::Welcome);
  CHECK(is_first(wizard));
  previous(wizard);
  CHECK(current(wizard) == WizardStep::Welcome);
  next(wizard);
  CHECK(current(wizard) == WizardStep::Engines);
  next(wizard);
  skip(wizard);                                     // la calibration de la machine passée
  CHECK(current(wizard) == WizardStep::Languages);  // l'écran HDR n'existe plus
  CHECK(progress_text(wizard) == "Étape 4 sur 6 : Langues");
  CHECK(wizard.skipped == std::vector<WizardStep>{WizardStep::Chain});
  // Fermé après le rang 4 ou 5 (retirés) : on reprend à Langues.
  CHECK(current(begin_wizard(facts, 3)) == WizardStep::Languages);
  CHECK(current(begin_wizard(facts, 5)) == WizardStep::Languages);
  // Tout vu : le résumé.
  CHECK(current(begin_wizard(facts, 8)) == WizardStep::Summary);
  while (!is_last(wizard)) next(wizard);
  next(wizard);
  CHECK(current(wizard) == WizardStep::Summary);
}

TEST_CASE("Terminer écrit les choix et laisse intact ce qui est sauté", "[wizard]") {
  ajp::app::play::FrenchUi french;
  const WizardFacts facts{.engine_choice = true};
  Wizard wizard = begin_wizard(facts, 0);
  wizard.update_mode = "auto";

  const ajp::core::Settings written = wizard_settings(wizard, {});
  CHECK(written.first_run_done);
  CHECK(written.first_run_step == kWizardSteps);

  PlaySettings play;
  play = wizard_play_settings(wizard, play);
  CHECK(play.update_mode == "auto");
  CHECK(wants_machine_calibration(wizard));
  // « Régler moi-même » (P3.1) : pas de calibration, le préréglage Auto.
  Wizard myself = wizard;
  set_up_myself(myself);
  CHECK_FALSE(wants_machine_calibration(myself));
  CHECK(wizard_play_settings(myself, PlaySettings{}).preset == Preset::Auto);

  // Écrans passés : Me prévenir (défaut), préréglage Auto, pas de calibration.
  wizard.skipped = {WizardStep::Updates, WizardStep::Chain};
  const ajp::core::Settings kept = wizard_settings(wizard, {});
  PlaySettings other;
  other = wizard_play_settings(wizard, other);
  CHECK(other.update_mode == "notify");
  CHECK(other.preset == Preset::Auto);
  CHECK_FALSE(wants_machine_calibration(wizard));
}

TEST_CASE("le résumé dit ce qui est choisi, sauté, et où le changer", "[wizard]") {
  ajp::app::play::FrenchUi french;
  Wizard wizard = begin_wizard({.engine_choice = true}, 0);
  wizard.skipped = {WizardStep::Languages};
  const auto lines = wizard_summary(wizard);
  REQUIRE(lines.size() == 3);  // Chain, Languages, Updates
  CHECK(lines[0].contains("Vidéo > Post-traitement"));
  CHECK(lines[1].contains("sauté"));
  CHECK(lines[2].contains("me prévenir"));
}

TEST_CASE("l'assistant repris en route retrouve ses choix (P3.3)", "[wizard]") {
  PlaySettings saved;
  saved.update_mode = "never";
  saved.language_presets = {{.audio = "fra", .subtitles = ""}};
  Wizard wizard = begin_wizard(WizardFacts{}, 3);
  resume_choices(wizard, saved);
  CHECK(wizard.update_mode == "never");
  CHECK(wizard.audio_language == "fra");
  CHECK(wizard_play_settings(wizard, PlaySettings{}).update_mode == "never");
}
