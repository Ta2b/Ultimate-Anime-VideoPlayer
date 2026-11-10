#include "app/play/wizard.h"

#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "french_ui.h"

using namespace ajp::app::play;

TEST_CASE("l'assistant : le léger d'abord, les moteurs en dernier, puis Prêt (P3.18)", "[wizard]") {
  const auto all = wizard_steps({.engine_choice = true});
  CHECK(all == std::vector<WizardStep>{WizardStep::Welcome, WizardStep::Languages, WizardStep::Updates,
                                       WizardStep::Engines, WizardStep::Summary});
  const auto plain = wizard_steps({.engine_choice = false});
  CHECK(plain.size() == 4);  // pas de moteur à choisir
  CHECK(std::ranges::find(plain, WizardStep::Engines) == plain.end());
}

TEST_CASE("l'assistant avance, recule, passe et reprend en route", "[wizard]") {
  ajp::app::play::FrenchUi french;
  const WizardFacts facts{.engine_choice = true};
  Wizard wizard = begin_wizard(facts, 0);
  CHECK(current(wizard) == WizardStep::Welcome);
  previous(wizard);
  CHECK(current(wizard) == WizardStep::Welcome);
  next(wizard);
  skip(wizard);  // les langues passées
  CHECK(current(wizard) == WizardStep::Updates);
  CHECK(progress_text(wizard) == "Étape 3 sur 5 : Intégration");
  CHECK(wizard.skipped == std::vector<WizardStep>{WizardStep::Languages});
  previous(wizard);  // revenue puis « Suivant » : elle n'est plus passée (P3.16)
  next(wizard);
  CHECK(wizard.skipped.empty());
  CHECK(current(begin_wizard(facts, 3)) == WizardStep::Engines);
  CHECK(current(begin_wizard(facts, 8)) == WizardStep::Summary);  // rang d'une version d'avant
}

TEST_CASE("Prêt : Optimiser maintenant ou Régler moi-même ; un écran sauté n'écrase rien", "[wizard]") {
  ajp::app::play::FrenchUi french;
  Wizard wizard = begin_wizard({.engine_choice = true}, 0);
  wizard.update_mode = UpdateMode::Never;
  wizard.interface_language = "fr";
  CHECK(wizard_settings(wizard, {}).first_run_step == kWizardSteps);
  const PlaySettings play = wizard_play_settings(wizard, PlaySettings{});
  CHECK(play.update_mode == UpdateMode::Never);
  CHECK(play.language == "fr");
  CHECK(wants_machine_calibration(wizard));
  Wizard myself = wizard;
  set_up_myself(myself);
  CHECK_FALSE(wants_machine_calibration(myself));
  CHECK(wizard_play_settings(myself, PlaySettings{}).preset == Preset::Auto);
  CHECK(wizard_summary(myself).front().contains("passée"));
  wizard.skipped = {WizardStep::Updates};
  CHECK(wizard_play_settings(wizard, PlaySettings{}).update_mode == UpdateMode::Notify);
}

TEST_CASE("l'assistant repris en route retrouve ses choix (P3.3)", "[wizard]") {
  PlaySettings saved;
  saved.update_mode = UpdateMode::Never;
  saved.language_presets = {{.audio = "fra", .subtitles = ""}};
  Wizard wizard = begin_wizard(WizardFacts{}, 3);
  resume_choices(wizard, saved);
  CHECK(wizard.update_mode == UpdateMode::Never);
  CHECK(wizard.audio_language == "fra");
  CHECK(wizard_play_settings(wizard, PlaySettings{}).update_mode == UpdateMode::Never);
}

TEST_CASE("l'assistant relancé part des réglages en place et ne les efface pas (A25-09)", "[wizard]") {
  PlaySettings saved;
  saved.update_mode = UpdateMode::Never;
  saved.language = "fr";
  saved.language_presets = {{.audio = "fra", .subtitles = "eng"}};
  // --first-run : depuis le premier écran, avec les choix du poste ; « Suivant » sur Bienvenue récrit
  // play.toml avec les écrans pas encore vus, qui gardent donc ce qui était réglé.
  Wizard wizard = begin_wizard(WizardFacts{}, 0);
  resume_choices(wizard, saved);
  const PlaySettings written = wizard_play_settings(wizard, saved);
  CHECK(written.update_mode == UpdateMode::Never);
  CHECK(written.language == "fr");
  REQUIRE(written.language_presets.size() == 1);
  CHECK(written.language_presets[0].audio == "fra");
  CHECK(written.language_presets[0].subtitles == "eng");
}

TEST_CASE("Lecteur par défaut : pas demandé quand l'écran Intégration est passé (A25-04)", "[wizard]") {
  Wizard wizard = begin_wizard({.engine_choice = true}, 0);
  CHECK_FALSE(make_default_requested(wizard));
  wizard.make_default = true;
  CHECK(make_default_requested(wizard));
  wizard.skipped.push_back(WizardStep::Updates);
  CHECK_FALSE(make_default_requested(wizard));
}
