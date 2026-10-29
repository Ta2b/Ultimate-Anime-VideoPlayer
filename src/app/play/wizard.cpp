#include "app/play/wizard.h"

#include <algorithm>
#include <format>

#include "app/play/auto_config.h"

namespace ajp::app::play {

std::string_view title(WizardStep step) {
  switch (step) {
    case WizardStep::Welcome:
      return "Bienvenue";
    case WizardStep::Engines:
      return "Moteur";
    case WizardStep::Chain:
      return "Image et fluidité";
    case WizardStep::Hdr:
      return "HDR";
    case WizardStep::Audio:
      return "Audio";
    case WizardStep::Languages:
      return "Langues";
    case WizardStep::Updates:
      return "Mises à jour";
    case WizardStep::Summary:
      return "Résumé";
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
      return "Pas encore dit";
    case Listening::Headphones:
      return "Casque";
    case Listening::Speakers:
      return "Enceintes";
    case Listening::Cinema:
      return "Home cinéma";
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
      return "Automatique";
    case HdrChoice::RtxHdr:
      return "RTX HDR";
    case HdrChoice::Adaptive:
      return "HDR adaptatif";
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
  return std::format("Étape {} sur {} : {}", wizard.at + 1, wizard.steps.size(), title(current(wizard)));
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
                      ? "Image et fluidité : sauté, préréglage Auto (Général > Configuration)"
                      : std::format("Image et fluidité : {}, priorité {} (Général > Configuration)",
                                    label(wizard.preset), label(wizard.priority)));
  }
  if (has(WizardStep::Hdr)) {
    if (skipped(WizardStep::Hdr)) {
      out.push_back("HDR : sauté, automatique (HDR et couleurs > HDR)");
    } else if (wizard.hdr == HdrChoice::RtxHdr && !wizard.rtx_checks.empty() &&
               !rtx_ready(wizard.rtx_checks)) {
      out.push_back(
          "HDR : RTX HDR choisi mais pas prêt (HDR et couleurs > HDR) : en attendant, "
          "l'écran reste en HDR adaptatif");
    } else {
      out.push_back(std::format("HDR : {} (HDR et couleurs > HDR)", label(wizard.hdr)));
    }
  }
  if (has(WizardStep::Audio)) {
    out.push_back(wizard.listening == Listening::Unsaid || skipped(WizardStep::Audio)
                      ? "Audio : enceintes en rendu fidèle (Audio > Sortie)"
                      : std::format("Audio : {} (Audio > Sortie)", label(wizard.listening)));
  }
  if (has(WizardStep::Languages))
    out.push_back(skipped(WizardStep::Languages) ? "Langues : sauté (Général > Lecture)"
                                                 : "Langues : choisies (Général > Lecture)");
  if (has(WizardStep::Updates)) {
    const std::string_view mode = wizard.update_mode == "auto"    ? "automatiques"
                                  : wizard.update_mode == "never" ? "jamais"
                                                                  : "me prévenir";
    out.push_back(std::format("Mises à jour : {} (Général > Application)", mode));
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
