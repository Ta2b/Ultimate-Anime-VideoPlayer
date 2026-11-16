#include "app/play/wizard.h"

#include <algorithm>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/engine_screen.h"
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

TEST_CASE("Prêt : le préréglage Auto de la carte, sans calibration ; un écran sauté n'écrase rien",
          "[wizard]") {
  ajp::app::play::FrenchUi french;
  Wizard wizard = begin_wizard({.engine_choice = true}, 0);
  wizard.update_mode = UpdateMode::Never;
  wizard.interface_language = "fr";
  CHECK(wizard_settings({}).first_run_step == kWizardSteps);
  const PlaySettings play = wizard_play_settings(wizard, PlaySettings{});
  CHECK(play.update_mode == UpdateMode::Never);
  CHECK(play.language == "fr");
  PlaySettings custom;
  custom.preset = Preset::Custom;
  CHECK(wizard_play_settings(wizard, custom).preset == Preset::Auto);
  // Les langues : la version choisie et sa langue, dites au résumé.
  wizard.own_language = "fre";
  wizard.language_version = LanguageVersion::Dubbed;
  wizard.language_presets = language_presets_for("fre", LanguageVersion::Dubbed);
  const PlaySettings dubbed = wizard_play_settings(wizard, PlaySettings{});
  CHECK(dubbed.language_version == LanguageVersion::Dubbed);
  CHECK(dubbed.language_presets.front() == ajp::media::LanguagePreset{"fre", ""});
  CHECK(wizard_summary(wizard).front() == "Langues : Français, doublé (Langues > Langues)");
  wizard.skipped = {WizardStep::Updates};
  CHECK(wizard_play_settings(wizard, PlaySettings{}).update_mode == UpdateMode::Notify);
}

TEST_CASE("l'assistant repris en route retrouve ses choix (P3.3)", "[wizard]") {
  PlaySettings saved;
  saved.update_mode = UpdateMode::Never;
  saved.language_presets = {{.audio = "fra", .subtitles = ""}};
  saved.language_version = LanguageVersion::Custom;
  Wizard wizard = begin_wizard(WizardFacts{}, 3);
  resume_choices(wizard, saved);
  CHECK(wizard.update_mode == UpdateMode::Never);
  CHECK(wizard.language_presets == saved.language_presets);
  CHECK(wizard.advanced_languages);  // des choix faits un à un se montrent un à un
  CHECK(wizard_play_settings(wizard, PlaySettings{}).update_mode == UpdateMode::Never);
}

TEST_CASE("l'assistant relancé part des réglages en place et ne les efface pas (A25-09)", "[wizard]") {
  PlaySettings saved;
  saved.update_mode = UpdateMode::Never;
  saved.language = "fr";
  saved.language_presets = {{.audio = "fra", .subtitles = "eng"}};
  saved.language_version = LanguageVersion::Custom;
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

TEST_CASE("Bienvenue : au repos l'écran ne se redessine plus 60 fois par seconde (A25-15)", "[app][wizard]") {
  using ajp::app::play::welcome_wait_ms;
  CHECK(welcome_wait_ms(0, false) == 16);
  CHECK(welcome_wait_ms(999'999, false) == 16);
  CHECK(welcome_wait_ms(1'000'000, false) == 250);
  // Un essai (--keys, --snapshot) garde sa précision.
  CHECK(welcome_wait_ms(5'000'000, true) == 16);
}

TEST_CASE("l'écran Moteur n'est vu que par « Suivant » (C25-01)", "[wizard]") {
  const WizardFacts facts{.engine_choice = true};
  Wizard wizard = begin_wizard(facts, 0);
  CHECK_FALSE(engines_reviewed(wizard, 0));  // rien de vu : aucune licence montrée
  skip(wizard);                              // Bienvenue passé, l'écran Moteur pas atteint
  CHECK_FALSE(engines_reviewed(wizard, 0));
  wizard = begin_wizard(facts, 3);
  REQUIRE(current(wizard) == WizardStep::Engines);
  skip(wizard);
  CHECK_FALSE(engines_reviewed(wizard, 0));  // passé : ni accord ni téléchargement
  previous(wizard);
  next(wizard);  // revu, puis « Suivant »
  CHECK(engines_reviewed(wizard, 0));
  // Repris en route : l'écran vu dans la séance précédente compte.
  CHECK(engines_reviewed(begin_wizard(facts, 4), static_cast<int>(WizardStep::Engines)));
  // Sans écran Moteur, rien à relire.
  CHECK(engines_reviewed(begin_wizard({.engine_choice = false}, 0), 0));
}
