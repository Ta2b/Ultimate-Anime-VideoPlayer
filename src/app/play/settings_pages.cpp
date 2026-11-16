#include "app/play/settings_pages.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <string_view>

#include <imgui.h>

#include "app/play/auto_config.h"
#include "app/play/i18n.h"
#include "app/play/panel_style.h"
#include "app/play/settings_color_tabs.h"
#include "app/play/settings_engines.h"
#include "app/play/tracks.h"
#include "render/grade_params.h"

namespace ajp::app::play {

using panel::kWarn;
using panel::note;
using panel::section;

namespace {

// Vidéo > Général : le niveau, d'Économe à Ultra, refait la chaîne (réduction, SPAN, interpolation) pour
// cette machine, cet écran et cette vidéo (auto_config.h : apply_preset). Auto prend le niveau de la machine
// ; un réglage fait à la main ailleurs passe en Personnalisé, montré seulement quand il est en cours.
void preset_section(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("Preset"));
  const auto detected = static_cast<Tier>(std::clamp(choices.machine_tier, 0, kTierCount - 1));
  // Le préréglage qui porte le niveau de la machine : le nom qu'Auto affiche entre parenthèses.
  constexpr std::array<Preset, kTierCount> kByTier{Preset::Economy, Preset::Fast, Preset::Normal,
                                                   Preset::Quality, Preset::Ultra};
  const auto text = [&](Preset preset) {
    if (preset != Preset::Auto) return std::string(label(preset));
    if (choices.calibrated) return std::string(i18n::tr("Auto (calibrated)"));
    return i18n::tr_format("Auto ({})", label(kByTier[static_cast<std::size_t>(detected)]));
  };
  if (ImGui::BeginCombo(panel::left(i18n::tr("Quality preset")).c_str(), text(d.preset).c_str())) {
    for (int i = 0; i < kPresetCount; ++i) {
      const auto preset = static_cast<Preset>(i);
      if (preset == Preset::Custom && d.preset != Preset::Custom) continue;
      if (ImGui::Selectable(text(preset).c_str(), preset == d.preset) && preset != d.preset &&
          preset != Preset::Custom) {
        PlaySettings next = d;
        next.preset = preset;
        d = apply_preset(
            next,
            Machine{.tier = detected,
                    .display_height = choices.display_height,
                    .display_hz = choices.display_hz,
                    .explored_work_height = choices.explored_work_height,
                    .explored_rife_model = choices.explored_rife_model,
                    .calibrated = choices.calibrated,
                    .display_width = choices.display_width,
                    .span_fp8 = choices.span_fp8},
            core::SourceInfo{
                .width = choices.source_width, .height = choices.source_height, .fps = choices.source_fps});
      }
    }
    ImGui::EndCombo();
  }
  panel::tip(
      i18n::tr("Quality preset"),
      i18n::tr("Sets upscaling and interpolation together, from Economy (light) to Ultra. Auto picks the "
               "level of this machine. Changing a setting by hand switches to Custom. Recommended: "
               "Auto."),
      panel::Apply::Reload);
}

// « Afficher les FPS » : la cadence seule, en haut à gauche de l'image.
void fps_section(PlaySettings& d) {
  section(i18n::tr("On screen"));
  ImGui::Checkbox(i18n::tr("Show FPS"), &d.stats.fps);
  panel::tip(i18n::tr("Show FPS"),
             i18n::tr("The frame rate on screen, top left. Tab shows every statistic. Recommended: off."));
}

}  // namespace

void application_page(PlaySettings& d, const SettingsChoices& choices, UpdateNote& updates,
                      SystemNote& system, std::vector<Input>& actions) {
  language_section(d);
  section(i18n::tr("Playback"));
  ImGui::Checkbox(i18n::tr("Autoplay next episode"), &d.autoplay_next);
  panel::tip(i18n::tr("Autoplay next episode"),
             i18n::tr("Plays the next file of the folder at the end. N: next, Shift+N: previous, O: skip "
                      "credits. Recommended: on."));
  updates_page(d, choices.place, updates);
  default_player_page(choices.place, system, choices.default_player_is_ajp);
  section(i18n::tr("This machine"));
  panel::wrapped(choices.gpu_name.empty() ? i18n::tr("Unknown graphics card") : choices.gpu_name.c_str());
  if (choices.display_hz > 0.0) note(i18n::tr_format("Display {:.0f} Hz", choices.display_hz));
  if (ImGui::Button(i18n::tr("Run the wizard again"))) actions.emplace_back(OpenEngineScreen{.wizard = true});
  panel::tip(i18n::tr("Run the wizard again"), i18n::tr("Nothing changes until \"Start\"."));
  report_problem_section(choices.place, choices.gpu_name);
}

bool reset_section(PlaySettings& draft) {
  bool reset_all = false;
  section(i18n::tr("All settings"));
  // Une confirmation avant de tout perdre (A23-05) : le premier clic demande, le second réinitialise.
  ImGuiStorage* storage = ImGui::GetStateStorage();
  const ImGuiID asking_id = ImGui::GetID("reset-all-asking");
  if (!storage->GetBool(asking_id, false)) {
    if (ImGui::Button(i18n::tr("Reset all settings"))) storage->SetBool(asking_id, true);
    panel::tip(i18n::tr("Reset all settings"),
               i18n::tr("Every setting goes back to its default. The engines folder and the language are "
                        "kept."),
               panel::Apply::Reload);
  } else {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    panel::wrapped(i18n::tr("Every setting goes back to its default. This cannot be undone."));
    ImGui::PopStyleColor();
    if (ImGui::Button(i18n::tr("Yes, reset everything"))) {
      PlaySettings fresh;
      fresh.engines_dir = draft.engines_dir;
      fresh.language = draft.language;
      draft = fresh;
      reset_all = true;
      storage->SetBool(asking_id, false);
    }
    ImGui::SameLine();
    if (ImGui::Button(i18n::tr("Keep my settings"))) storage->SetBool(asking_id, false);
  }
  return reset_all;
}

void statistics_page(PlaySettings& d) {
  section(i18n::tr("On-screen statistics"));
  ImGui::Checkbox(i18n::tr("All statistics (Tab)"), &d.stats.all);
  ImGui::BeginDisabled(d.stats.all);
  ImGui::Checkbox(i18n::tr("Pipeline"), &d.stats.chain);
  ImGui::Checkbox(i18n::tr("Frames"), &d.stats.frames);
  ImGui::Checkbox(i18n::tr("Interpolation"), &d.stats.rife);
  ImGui::Checkbox(i18n::tr("Tracks"), &d.stats.tracks);
  ImGui::Checkbox(i18n::tr("HDR output"), &d.stats.output);
  ImGui::Checkbox(i18n::tr("Exact position"), &d.stats.position);
  ImGui::Checkbox(i18n::tr("Video memory"), &d.stats.memory);
  ImGui::EndDisabled();
  panel::advanced_header();
  section(i18n::tr("Debugging"));
  ImGui::Checkbox(i18n::tr("Show held and cut pairs"), &d.stats.pairs);
  panel::tip(i18n::tr("Show held and cut pairs"),
             i18n::tr("Each pair judged a held drawing (blue) or a scene cut (red) tints the screen for a "
                      "moment and goes to ajp.log. This session only."));
}

void video_general_page(PlaySettings& d, const SettingsChoices& choices, std::vector<Input>& actions,
                        std::optional<core::EngineId>& engine_draft) {
  stages_note(choices, actions);
  preset_section(d, choices);
  engine_section(choices, engine_draft, actions, d.gpu);
  output_section(d, choices);
  colors_section(d);
  fps_section(d);
}

void display_section(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("Display"));
  if (ImGui::BeginCombo(panel::left(i18n::tr("Scaler")).c_str(),
                        std::string(label(d.final_scaler)).c_str())) {
    for (int i = 0; i < kFinalScalerCount; ++i) {
      const auto scaler = static_cast<FinalScaler>(i);
      if (ImGui::Selectable(std::string(label(scaler)).c_str(), scaler == d.final_scaler))
        d.final_scaler = scaler;
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("Scaler"),
             i18n::tr("How the picture is stretched to the window. EWA Lanczos is the sharpest but the "
                      "heaviest for the card. Recommended: the default."));
  ImGui::Checkbox(i18n::tr("Debanding"), &d.deband);
  panel::tip(i18n::tr("Debanding"),
             i18n::tr("Smooths visible steps in skies and dark gradients. Can soften fine grain. Speed: "
                      "minor cost. Quality impact: minor; can soften fine grain. Recommended: off."));
  if (ImGui::BeginCombo(panel::left(i18n::tr("True black")).c_str(),
                        std::string(label(d.oled_black)).c_str())) {
    for (int i = 0; i < kOledBlackCount; ++i) {
      const auto mode = static_cast<OledBlack>(i);
      if (ImGui::Selectable(std::string(label(mode)).c_str(), mode == d.oled_black)) d.oled_black = mode;
    }
    ImGui::EndCombo();
  }
  panel::tip(
      i18n::tr("True black"),
      i18n::tr("On an OLED, near-black areas become truly black: deeper blacks, no grey glow. The very "
               "darkest details can merge into black. Auto: on an OLED display. Recommended: Auto."));
  ImGui::BeginDisabled(d.oled_black == OledBlack::Off);
  grade_slider(i18n::tr("Threshold (code steps)"), d.oled_black_steps, render::grade_limits::kBlackSteps);
  panel::live_preview();
  ImGui::EndDisabled();
  note((i18n::tr("Display: ") + choices.display_text).c_str());
  if (!choices.black_status.empty()) note(choices.black_status.c_str());
  ImGui::Checkbox(i18n::tr("Hide the bar when paused (burn-in)"), &d.oled_care);
  panel::tip(i18n::tr("Hide the bar when paused (burn-in)"),
             i18n::tr("On an OLED display, the bar fades after 20 s of pause. Recommended: on."));
  ImGui::Checkbox(i18n::tr("Steady frame rate (VRR flicker)"), &d.steady_cadence);
  panel::tip(i18n::tr("Steady frame rate (VRR flicker)"),
             i18n::tr("On an OLED with VRR, keeps the refresh rate constant so dark scenes stop flickering. "
                      "Costs a little card work. Speed: one more composition per refresh. Recommended: off; "
                      "on if dark scenes flicker."));
}

}  // namespace ajp::app::play
