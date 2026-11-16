// L'onglet Audio du panneau des réglages : appareil et synchronisation.
#include "app/play/settings_sound.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>

#include "app/play/panel_style.h"
#include "app/play/settings_panel.h"
#include "app/play/text_fit.h"
#include "app/play/tracks.h"

namespace ajp::app::play {

using panel::note;
using panel::section;

// Un autre appareil choisi : le délai du précédent est gardé pour lui, celui du nouveau repris (P2.7). Suivre
// le système et nommer la carte que le système prend (system_default) est le même appareil : le délai reste
// (B23-06).
void switch_device_delay(PlaySettings& d, const std::string& before, std::string_view system_default) {
  if (d.audio_device == before) return;
  const bool same_card = !system_default.empty() && ((before.empty() && d.audio_device == system_default) ||
                                                     (d.audio_device.empty() && before == system_default));
  if (same_card) return;
  d.audio_device_delays_ms[before] = d.audio_delay_ms;
  const auto known = d.audio_device_delays_ms.find(d.audio_device);
  d.audio_delay_ms = known != d.audio_device_delays_ms.end() ? known->second : 0;
  d.audio_device_delays_ms.erase(d.audio_device);
}

// L'appareil de sortie : celui du système, suivi quand il change, ou un autre de la liste.
void device_section(PlaySettings& d, const SettingsChoices& choices) {
  const std::string device_before = d.audio_device;
  // Appareil : celui du système (suivi quand il change) ou un autre de la liste.
  section(i18n::tr("Device"));
  const SettingsChoices::AudioOut* chosen = nullptr;
  const SettingsChoices::AudioOut* system_default = nullptr;
  for (const SettingsChoices::AudioOut& out : choices.audio_outputs) {
    if (out.is_default) system_default = &out;
    if (!d.audio_device.empty() && out.name == d.audio_device) chosen = &out;
  }
  const SettingsChoices::AudioOut* active = chosen != nullptr ? chosen : system_default;
  // Un appareil enregistré qui a disparu reste le réglage : la case n'est pas cochée à sa place (B23-07).
  bool follow = d.audio_device.empty();
  // La case et la liste côte à côte ; la liste se grise quand le système choisit.
  const bool grid = panel::begin_grid("##appareil");
  if (grid) panel::cell();
  if (ImGui::Checkbox(i18n::tr("Follow the system default device"), &follow)) {
    if (follow) {
      d.audio_device.clear();
    } else if (active != nullptr) {
      d.audio_device = active->name;
    }
  }
  if (grid) panel::cell();
  ImGui::BeginDisabled(follow);
  const std::string missing = !d.audio_device.empty() && chosen == nullptr
                                  ? i18n::tr_format("{} (not connected)", d.audio_device)
                                  : std::string();
  if (ImGui::BeginCombo(panel::left(i18n::tr("Output device")).c_str(),
                        chosen != nullptr  ? chosen->name.c_str()
                        : !missing.empty() ? missing.c_str()
                                           : (system_default != nullptr ? system_default->name.c_str()
                                                                        : i18n::tr("None")))) {
    for (const SettingsChoices::AudioOut& out : choices.audio_outputs) {
      // Deux appareils de même nom ont le même libellé : l'identifiant ImGui vient de la place (C23-07).
      ImGui::PushID(static_cast<int>(&out - choices.audio_outputs.data()));
      if (ImGui::Selectable(out.name.c_str(), chosen == &out)) d.audio_device = out.name;
      ImGui::PopID();
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Output device"), i18n::tr("Recommended: follow the system."), panel::Apply::Reload);
  if (grid) panel::end_grid();
  switch_device_delay(d, device_before,
                      system_default != nullptr ? system_default->name : std::string_view{});
  if (active != nullptr && active->channels > 0)
    note(i18n::tr_format("{} channels on this device.", active->channels));
}

bool sound_tab(PlaySettings& d, const SettingsChoices& choices) {
  device_section(d, choices);
  panel::advanced_header();
  section(i18n::tr("Sync"));
  if (!panel::begin_grid("##synchro")) return false;
  panel::cell();
  // Sans AlwaysClamp : Ctrl + clic saisit un délai au-delà du rail, jusqu'à ±5 000 ms.
  const std::string delay_id = panel::left(i18n::tr("Audio delay (ms)"));
  panel::slider_int(delay_id.c_str(), &d.audio_delay_ms, limits::kAudioDelaySlider.lo,
                    limits::kAudioDelaySlider.hi, std::format("{} ms", d.audio_delay_ms),
                    ImGuiSliderFlags_None);
  const bool held = ImGui::IsItemActive();
  static_assert(limits::kAudioDelayMs.hi == 5000, "le texte de l'infobulle dit ±5,000 ms");
  panel::tip(
      i18n::tr("Audio delay (ms)"),
      i18n::tr("Positive delays the sound. Kept for this output device. Ctrl+click to type a value (up "
               "to ±5,000 ms). Recommended: 0."));
  panel::cell();
  if (ImGui::Button(i18n::tr("Reset to 0"))) d.audio_delay_ms = 0;
  d.audio_delay_ms = std::clamp(d.audio_delay_ms, limits::kAudioDelayMs.lo, limits::kAudioDelayMs.hi);
  panel::cell();
  ImGui::Checkbox(i18n::tr("Sync playback to the display"), &d.display_sync);
  panel::tip(i18n::tr("Sync playback to the display"),
             i18n::tr("On a fixed-rate display, every frame lasts the same time: no regular judder. Playback "
                      "speed changes by at most 0.125 %, and sound follows inaudibly. Recommended: on."));
  panel::cell();
  const std::string latency_id = panel::left(i18n::tr("Display latency (ms)"));
  panel::slider_int(latency_id.c_str(), &d.video_latency_ms, limits::kVideoLatencyMs.lo,
                    limits::kVideoLatencyMs.hi,
                    d.video_latency_ms == 0 ? std::string(i18n::tr("estimated"))
                                            : std::format("{} ms", d.video_latency_ms));
  panel::tip(i18n::tr("Display latency (ms)"),
             i18n::tr("The time between choosing a frame and seeing it. Set it if the picture seems early or "
                      "late. Recommended: estimated (0)."));
  d.video_latency_ms = std::clamp(d.video_latency_ms, limits::kVideoLatencyMs.lo, limits::kVideoLatencyMs.hi);
  panel::end_grid();
  return held;
}

}  // namespace ajp::app::play
