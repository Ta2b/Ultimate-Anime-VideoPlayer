#include "app/play/wizard.h"

#include <algorithm>
#include <format>

#include "app/play/auto_config.h"

namespace ajp::app::play {

std::string_view title(WizardStep step) {
  switch (step) {
    case WizardStep::Welcome:
      return i18n::tr("Welcome");
    case WizardStep::Engines:
      return i18n::tr("Engine");
    case WizardStep::Chain:
      return i18n::tr("Picture and smoothness");
    case WizardStep::Hdr:
      return i18n::tr("HDR");
    case WizardStep::Audio:
      return i18n::tr("Audio");
    case WizardStep::Languages:
      return i18n::tr("Languages");
    case WizardStep::Updates:
      return i18n::tr("Updates");
    case WizardStep::Summary:
      return i18n::tr("Summary");
  }
  return {};
}

std::vector<WizardStep> wizard_steps(const WizardFacts& facts) {
  std::vector<WizardStep> steps;
  for (int i = 1; i <= kWizardSteps; ++i) {
    const auto step = static_cast<WizardStep>(i);
    if (step == WizardStep::Engines && !facts.engine_choice) continue;
    if (step == WizardStep::Hdr && !facts.hdr_display) continue;
    steps.push_back(step);
  }
  return steps;
}

std::string_view key(Listening listening) {
  switch (listening) {
    case Listening::Unsaid:
      return {};
    case Listening::Headphones:
      return "headphones";
    case Listening::Speakers:
      return "speakers";
    case Listening::Cinema:
      return "cinema";
  }
  return {};
}

std::string_view label(Listening listening) {
  switch (listening) {
    case Listening::Unsaid:
      return i18n::tr("Not said yet");
    case Listening::Headphones:
      return i18n::tr("Headphones");
    case Listening::Speakers:
      return i18n::tr("Speakers");
    case Listening::Cinema:
      return i18n::tr("Home theater");
  }
  return {};
}

Listening listening_from_key(std::string_view text) {
  for (const Listening each : {Listening::Headphones, Listening::Speakers, Listening::Cinema}) {
    if (key(each) == text) return each;
  }
  return Listening::Unsaid;
}

std::string_view label(HdrChoice choice) {
  switch (choice) {
    case HdrChoice::Auto:
      return i18n::tr("Automatic");
    case HdrChoice::RtxHdr:
      return "RTX HDR";
    case HdrChoice::Adaptive:
      return i18n::tr("Adaptive HDR");
    case HdrChoice::Sdr:
      return "SDR";
  }
  return {};
}

bool rtx_ready(const std::vector<RtxCheck>& checks) {
  return !checks.empty() && std::ranges::all_of(checks, &RtxCheck::done);
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
  wizard.seen = std::max(wizard.seen, static_cast<int>(current(wizard)));
  if (!is_last(wizard)) ++wizard.at;
}
void previous(Wizard& wizard) {
  if (!is_first(wizard)) --wizard.at;
}
void skip(Wizard& wizard) {
  const WizardStep step = current(wizard);
  if (std::ranges::find(wizard.skipped, step) == wizard.skipped.end()) wizard.skipped.push_back(step);
  next(wizard);
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
  if (has(WizardStep::Chain)) {
    out.push_back(skipped(WizardStep::Chain)
                      ? i18n::tr("Picture and smoothness: skipped, Auto preset (General > Setup)")
                      : i18n::tr_format("Picture and smoothness: {}, priority {} (General > Setup)",
                                        label(wizard.preset), label(wizard.priority)));
  }
  if (has(WizardStep::Hdr)) {
    if (skipped(WizardStep::Hdr)) {
      out.push_back(i18n::tr("HDR: skipped, automatic (HDR and color > HDR)"));
    } else if (wizard.hdr == HdrChoice::RtxHdr && !wizard.rtx_checks.empty() &&
               !rtx_ready(wizard.rtx_checks)) {
      out.push_back(i18n::tr(
          "HDR: RTX HDR chosen but not ready (HDR and color > HDR); meanwhile adaptive HDR is used"));
    } else {
      out.push_back(i18n::tr_format("HDR: {} (HDR and color > HDR)", label(wizard.hdr)));
    }
  }
  if (has(WizardStep::Audio)) {
    out.push_back(wizard.listening == Listening::Unsaid || skipped(WizardStep::Audio)
                      ? i18n::tr("Audio: speakers, faithful sound (Audio > Output)")
                      : i18n::tr_format("Audio: {} (Audio > Output)", label(wizard.listening)));
  }
  if (has(WizardStep::Languages))
    out.push_back(skipped(WizardStep::Languages) ? i18n::tr("Languages: skipped (General > Playback)")
                                                 : i18n::tr("Languages: chosen (General > Playback)"));
  if (has(WizardStep::Updates)) {
    std::string_view mode = i18n::tr("notify me");
    if (wizard.update_mode == "auto") mode = i18n::tr("automatic");
    if (wizard.update_mode == "never") mode = i18n::tr("never");
    out.push_back(i18n::tr_format("Updates: {} (General > Application)", mode));
  }
  return out;
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
  if (!skipped(WizardStep::Chain)) {
    settings.preset = wizard.preset;
    settings.priority = wizard.priority;
  } else {
    settings.preset = Preset::Auto;  // le premier lancement choisit Auto
  }
  // Passé ou ignoré : Enceintes en rendu fidèle (le défaut).
  if (wizard.listening != Listening::Unsaid && !skipped(WizardStep::Audio))
    settings.listening = std::string(key(wizard.listening));
  if (!skipped(WizardStep::Updates)) {
    settings.update_mode = wizard.update_mode;
    settings.update_channel = wizard.update_channel;
  }
  if (!skipped(WizardStep::Languages)) {
    if (settings.language_presets.empty()) settings.language_presets.resize(1);
    settings.language_presets[0].audio = wizard.audio_language;
    settings.language_presets[0].subtitles = wizard.subtitle_language;
  }
  if (!skipped(WizardStep::Hdr)) {
    switch (wizard.hdr) {
      case HdrChoice::Auto:
        break;
      case HdrChoice::RtxHdr:
        // Pas prêt (4b non terminé) : le HDR adaptatif en attendant, RTX HDR se choisit ensuite dans HDR >
        // Conversion.
        settings.true_hdr = wizard.rtx_checks.empty() || rtx_ready(wizard.rtx_checks);
        settings.hdr_native = !settings.true_hdr;
        break;
      case HdrChoice::Adaptive:
        settings.true_hdr = false;
        settings.hdr_native = true;
        break;
      case HdrChoice::Sdr:
        settings.true_hdr = false;
        settings.hdr_native = false;
        break;
    }
  }
  return settings;
}

}  // namespace ajp::app::play
