#include "app/play/wizard.h"

#include <catch2/catch_test_macros.hpp>

#include "french_ui.h"

using namespace ajp::app::play;

TEST_CASE("l'assistant saute les écrans sans objet", "[wizard]") {
  ajp::app::play::FrenchUi french;
  CHECK(wizard_steps({.engine_choice = true, .hdr_display = true}).size() == 7);  // sans le rang 5, retiré
  const auto plain = wizard_steps({.engine_choice = false, .hdr_display = false});
  REQUIRE(plain.size() == 5);
  CHECK(plain.front() == WizardStep::Welcome);
  CHECK(plain[1] == WizardStep::Chain);  // ni moteur ni HDR
  CHECK(plain.back() == WizardStep::Summary);
}

TEST_CASE("l'assistant avance, recule, passe et reprend en route", "[wizard]") {
  ajp::app::play::FrenchUi french;
  const WizardFacts facts{.engine_choice = true, .hdr_display = false};
  Wizard wizard = begin_wizard(facts, 0);
  CHECK(current(wizard) == WizardStep::Welcome);
  CHECK(is_first(wizard));
  previous(wizard);
  CHECK(current(wizard) == WizardStep::Welcome);
  next(wizard);
  CHECK(current(wizard) == WizardStep::Engines);
  next(wizard);
  skip(wizard);                                     // la calibration de la machine passée
  CHECK(current(wizard) == WizardStep::Languages);  // pas de HDR : l'écran n'est pas HDR
  CHECK(progress_text(wizard) == "Étape 4 sur 6 : Langues");
  CHECK(wizard.skipped == std::vector<WizardStep>{WizardStep::Chain});
  // Fermé après le rang 5 (retiré) : on reprend à Langues.
  CHECK(current(begin_wizard(facts, 5)) == WizardStep::Languages);
  // Tout vu : le résumé.
  CHECK(current(begin_wizard(facts, 8)) == WizardStep::Summary);
  while (!is_last(wizard)) next(wizard);
  next(wizard);
  CHECK(current(wizard) == WizardStep::Summary);
}

TEST_CASE("Terminer écrit les choix et laisse intact ce qui est sauté", "[wizard]") {
  ajp::app::play::FrenchUi french;
  const WizardFacts facts{.engine_choice = true, .hdr_display = true};
  Wizard wizard = begin_wizard(facts, 0);
  wizard.hdr = HdrChoice::Hdr;
  wizard.update_mode = "auto";

  const ajp::core::Settings written = wizard_settings(wizard, {});
  CHECK(written.first_run_done);
  CHECK(written.first_run_step == kWizardSteps);

  PlaySettings play;
  play = wizard_play_settings(wizard, play);
  CHECK(play.update_mode == "auto");
  CHECK(play.hdr_native);
  CHECK(wants_machine_calibration(wizard));

  // Écrans passés : Me prévenir (défaut), HDR inchangé, préréglage Auto, pas de calibration.
  wizard.skipped = {WizardStep::Updates, WizardStep::Hdr, WizardStep::Chain};
  const ajp::core::Settings kept = wizard_settings(wizard, {});
  PlaySettings other;
  other.hdr_native = false;
  other = wizard_play_settings(wizard, other);
  CHECK(other.update_mode == "notify");
  CHECK_FALSE(other.hdr_native);
  CHECK(other.preset == Preset::Auto);
  CHECK_FALSE(wants_machine_calibration(wizard));
}

TEST_CASE("le résumé dit ce qui est choisi, sauté, et où le changer", "[wizard]") {
  ajp::app::play::FrenchUi french;
  Wizard wizard = begin_wizard({.engine_choice = true, .hdr_display = false}, 0);
  wizard.skipped = {WizardStep::Languages};
  const auto lines = wizard_summary(wizard);
  REQUIRE(lines.size() == 3);  // Chain, Languages, Updates (pas de HDR)
  CHECK(lines[0].contains("Général > Configuration"));
  CHECK(lines[1].contains("sauté"));
  CHECK(lines[2].contains("me prévenir"));
}
