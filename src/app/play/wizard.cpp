#include "app/play/wizard.h"

#include <algorithm>
#include <format>

#include "app/play/auto_config.h"

namespace ajp::app::play {

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

void set_up_myself(Wizard& wizard) {
  wizard.calibrate = false;
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
  out.push_back(wizard.calibrate
                    ? i18n::tr("Machine calibration: with \"Calibrate now\", the best settings this "
                               "machine sustains (Video > Post-processing)")
                    : i18n::tr("Machine calibration: skipped, Auto preset (Video > Post-processing)"));
  if (has(WizardStep::Languages))
    out.push_back(skipped(WizardStep::Languages)
                      ? i18n::tr("Languages: skipped (Audio and playback > Playback)")
                      : i18n::tr("Languages: chosen (Audio and playback > Playback)"));
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

bool make_default_requested(const Wizard& wizard) {
  return wizard.make_default &&
         std::ranges::find(wizard.skipped, WizardStep::Updates) == wizard.skipped.end();
}

core::Settings wizard_settings([[maybe_unused]] const Wizard& wizard, core::Settings settings) {
  settings.first_run_done = true;
  settings.first_run_step = kWizardSteps;
  return settings;
}

PlaySettings wizard_play_settings(const Wizard& wizard, PlaySettings settings) {
  const auto skipped = [&](WizardStep step) {
    return std::ranges::find(wizard.skipped, step) != wizard.skipped.end();
  };
  // La calibration de la machine pose ensuite sa chaîne ; sans elle, le préréglage Auto estimé.
  if (!wizard.calibrate) settings.preset = Preset::Auto;
  if (!wizard.interface_language.empty()) settings.language = wizard.interface_language;
  if (!skipped(WizardStep::Updates)) {
    settings.update_mode = wizard.update_mode;
  }
  if (!skipped(WizardStep::Languages)) {
    if (settings.language_presets.empty()) settings.language_presets.resize(1);
    settings.language_presets[0].audio = wizard.audio_language;
    settings.language_presets[0].subtitles = wizard.subtitle_language;
  }
  return settings;
}

void resume_choices(Wizard& wizard, const PlaySettings& saved) {
  wizard.interface_language = saved.language;
  wizard.update_mode = saved.update_mode;
  if (!saved.language_presets.empty()) {
    wizard.audio_language = saved.language_presets[0].audio;
    wizard.subtitle_language = saved.language_presets[0].subtitles;
  }
}

bool wants_machine_calibration(const Wizard& wizard) {
  return wizard.calibrate;
}

}  // namespace ajp::app::play
