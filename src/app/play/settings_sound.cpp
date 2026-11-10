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

// Vrai pendant qu'on tient le curseur du délai audio.
// Un autre appareil choisi : le délai du précédent est gardé pour lui, celui du nouveau repris (P2.7).
void switch_device_delay(PlaySettings& d, const std::string& before) {
  if (d.audio_device == before) return;
  d.audio_device_delays_ms[before] = d.audio_delay_ms;
  const auto known = d.audio_device_delays_ms.find(d.audio_device);
  d.audio_delay_ms = known != d.audio_device_delays_ms.end() ? known->second : 0;
  d.audio_device_delays_ms.erase(d.audio_device);
}

bool sound_tab(PlaySettings& d, const SettingsChoices& choices) {
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
  bool follow = d.audio_device.empty() || chosen == nullptr;
  if (ImGui::Checkbox(i18n::tr("Follow the system default device"), &follow)) {
    if (follow) {
      d.audio_device.clear();
    } else if (active != nullptr) {
      d.audio_device = active->name;
    }
  }
  ImGui::BeginDisabled(follow);
  if (ImGui::BeginCombo(i18n::tr("Output device"),
                        chosen != nullptr ? chosen->name.c_str()
                                          : (system_default != nullptr ? system_default->name.c_str()
                                                                       : i18n::tr("None")))) {
    for (const SettingsChoices::AudioOut& out : choices.audio_outputs) {
      if (ImGui::Selectable(out.name.c_str(), chosen == &out)) d.audio_device = out.name;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  switch_device_delay(d, device_before);
  if (active != nullptr && active->channels > 0)
    note(i18n::tr_format("{} channels on this device.", active->channels));
  panel::tip(i18n::tr("Changing the device reopens the video at the same spot."));

  if (!panel::advanced()) return false;
  section(i18n::tr("Sync"));
  ImGui::SliderInt(i18n::tr("Audio delay (ms)"), &d.audio_delay_ms, limits::kAudioDelaySlider.lo,
                   limits::kAudioDelaySlider.hi);
  const bool held = ImGui::IsItemActive();
  static_assert(limits::kAudioDelayMs.hi == 5000, "le texte de l'infobulle dit ±5,000 ms");
  panel::tip(i18n::tr("Positive delays the sound. Ctrl+click the slider to type a value (up to ±5,000 ms)."));
  if (ImGui::Button(i18n::tr("Reset to 0"))) d.audio_delay_ms = 0;
  d.audio_delay_ms = std::clamp(d.audio_delay_ms, limits::kAudioDelayMs.lo, limits::kAudioDelayMs.hi);
  ImGui::Checkbox(i18n::tr("Sync playback to the display"), &d.display_sync);
  panel::tip(
      i18n::tr("On a fixed-rate display, playback speeds up or slows down by at most 0.125% so each frame "
               "fits whole refreshes: no more doubled frames. Sound follows, inaudibly."));
  ImGui::SliderInt(i18n::tr("Display latency (ms)"), &d.video_latency_ms, limits::kVideoLatencyMs.lo,
                   limits::kVideoLatencyMs.hi, d.video_latency_ms == 0 ? i18n::tr("estimated") : "%d ms",
                   ImGuiSliderFlags_AlwaysClamp);
  note(i18n::tr(
      "The time between choosing a frame and seeing it. Set it here if the picture seems early or late."));
  d.video_latency_ms = std::clamp(d.video_latency_ms, limits::kVideoLatencyMs.lo, limits::kVideoLatencyMs.hi);
  return held;
}

}  // namespace ajp::app::play
