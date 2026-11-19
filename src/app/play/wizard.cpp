#include "app/play/wizard.h"

#include <algorithm>

#include "app/play/auto_config.h"
#include "app/play/tracks.h"

namespace kirari::app::play {

std::string_view title(WizardStep step) {
  switch (step) {
    case WizardStep::Welcome:
      return i18n::tr("Welcome");
    case WizardStep::Languages:
      return i18n::tr("Languages");
    case WizardStep::Updates:
      return i18n::tr("Integration");
    case WizardStep::Engines:
      return i18n::tr("Engine");
    case WizardStep::Summary:
      return i18n::tr("Ready");
  }
  return {};
}

std::vector<WizardStep> wizard_steps(const WizardFacts& facts) {
  std::vector<WizardStep> steps;
  for (int i = 1; i <= kWizardSteps; ++i) {
    const auto step = static_cast<WizardStep>(i);
    if (step == WizardStep::Engines && !facts.engine_choice) continue;
    steps.push_back(step);
  }
  return steps;
}

Wizard begin_wizard(const WizardFacts& facts, int last_step_seen) {
  Wizard wizard;
  wizard.steps = wizard_steps(facts);
  // Le suivant du dernier écran vu : le premier écran de cette machine dont le rang dépasse last_step_seen.
  for (std::size_t i = 0; i < wizard.steps.size(); ++i) {
    if (static_cast<int>(wizard.steps[i]) > last_step_seen) {
      wizard.at = i;
      return wizard;
    }
  }
  wizard.at = wizard.steps.size() - 1;
  return wizard;
}

WizardStep current(const Wizard& wizard) {
  return wizard.steps[wizard.at];
}
bool is_first(const Wizard& wizard) {
  return wizard.at == 0;
}
bool is_last(const Wizard& wizard) {
  return wizard.at + 1 == wizard.steps.size();
}
void next(Wizard& wizard) {
  // « Suivant » sur un écran passé puis revu par « Précédent » : il ne compte plus comme passé (P3.16).
  std::erase(wizard.skipped, current(wizard));
  wizard.seen = std::max(wizard.seen, static_cast<int>(current(wizard)));
  if (!is_last(wizard)) ++wizard.at;
}
void previous(Wizard& wizard) {
  if (!is_first(wizard)) --wizard.at;
}
void skip(Wizard& wizard) {
  const WizardStep step = current(wizard);
  next(wizard);
  if (std::ranges::find(wizard.skipped, step) == wizard.skipped.end()) wizard.skipped.push_back(step);
}

std::string progress_text(const Wizard& wizard) {
  return i18n::tr_format("Step {} of {}: {}", wizard.at + 1, wizard.steps.size(), title(current(wizard)));
}

std::vector<std::string> wizard_summary(const Wizard& wizard) {
  const auto skipped = [&](WizardStep step) {
    return std::ranges::find(wizard.skipped, step) != wizard.skipped.end();
  };
  const auto has = [&](WizardStep step) {
    return std::ranges::find(wizard.steps, step) != wizard.steps.end();
  };
  std::vector<std::string> out;
  if (has(WizardStep::Languages)) {
    if (skipped(WizardStep::Languages))
      out.push_back(i18n::tr("Languages: skipped (Languages > Languages)"));
    else if (wizard.language_version == LanguageVersion::Dubbed)
      out.push_back(i18n::tr_format("Languages: {}, dubbed (Languages > Languages)",
                                    language_name(wizard.own_language)));
    else if (wizard.language_version == LanguageVersion::Original)
      out.push_back(i18n::tr_format("Languages: {}, original version with subtitles (Languages > Languages)",
                                    language_name(wizard.own_language)));
    else
      out.push_back(i18n::tr("Languages: chosen track by track (Languages > Languages)"));
  }
  if (has(WizardStep::Updates)) {
    std::string_view mode = i18n::tr("notify me");
    if (wizard.update_mode == UpdateMode::Never) mode = i18n::tr("never");
    out.push_back(i18n::tr_format("Updates: {} (General > Application)", mode));
    if (make_default_requested(wizard))
      out.push_back(
          i18n::tr("Default video player: asked at the end of the assistant (General > Application)"));
  }
  return out;
}

bool engines_reviewed(const Wizard& wizard, int saved_step) {
  if (std::ranges::find(wizard.steps, WizardStep::Engines) == wizard.steps.end()) return true;
  if (std::ranges::find(wizard.skipped, WizardStep::Engines) != wizard.skipped.end()) return false;
  const int engines = static_cast<int>(WizardStep::Engines);
  return wizard.seen >= engines || saved_step >= engines;
}

bool make_default_requested(const Wizard& wizard) {
  return wizard.make_default &&
         std::ranges::find(wizard.skipped, WizardStep::Updates) == wizard.skipped.end();
}

core::Settings wizard_settings(core::Settings settings) {
  settings.first_run_done = true;
  settings.first_run_step = kWizardSteps;
  return settings;
}

PlaySettings wizard_play_settings(const Wizard& wizard, PlaySettings settings) {
  const auto skipped = [&](WizardStep step) {
    return std::ranges::find(wizard.skipped, step) != wizard.skipped.end();
  };
  // Le niveau estimé pour la carte (gpu_tier.h) : Auto le prend à l'ouverture de la vidéo.
  settings.preset = Preset::Auto;
  if (!wizard.interface_language.empty()) settings.language = wizard.interface_language;
  if (!skipped(WizardStep::Updates)) {
    settings.update_mode = wizard.update_mode;
  }
  if (!skipped(WizardStep::Languages)) {
    settings.own_language = wizard.own_language;
    settings.language_version = wizard.language_version;
    settings.language_presets = wizard.language_presets;
  }
  return settings;
}

void resume_choices(Wizard& wizard, const PlaySettings& saved) {
  wizard.interface_language = saved.language;
  wizard.update_mode = saved.update_mode;
  wizard.own_language = saved.own_language;
  wizard.language_version = saved.language_version;
  wizard.language_presets = saved.language_presets;
  wizard.advanced_languages = saved.language_version == LanguageVersion::Custom;
}

}  // namespace kirari::app::play
