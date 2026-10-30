// L'onglet Audio du panneau des réglages : appareil, écoute, profils, Son anime, délai.
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
bool sound_tab(PlaySettings& d, const SettingsChoices& choices) {
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
  if (active != nullptr && active->channels > 0)
    note(i18n::tr_format("{} channels on this device.", active->channels));
  panel::tip(i18n::tr("Changing the device reopens the video at the same spot."));

  // Écoute : le rendu pour le casque, les enceintes ou le home cinéma. Aucune détection : c'est votre choix.
  section(i18n::tr("Listening"));
  const bool multichannel = active != nullptr && active->channels > 2;
  const std::string device_name = active != nullptr ? active->name : std::string();
  bool own = false;
  for (const auto& [name, key] : d.device_listening) own = own || name == device_name;
  std::string* key = &d.listening;
  if (own) {
    for (auto& [name, value] : d.device_listening)
      if (name == device_name) key = &value;
  }
  struct Choice {
    const char* key;
    const char* label;
    const char* detail;
  };
  const std::array<Choice, 3> kListening{{
      {"headphones", i18n::tr("Headphones"), i18n::tr("Crossfeed for stereo, binaural for surround.")},
      {"speakers", i18n::tr("Speakers"),
       i18n::tr("Faithful sound; surround is mixed down to stereo with a lifted center.")},
      {"cinema", i18n::tr("Home theater"), i18n::tr("Multichannel, surround sent as is.")},
  }};
  for (const Choice& choice : kListening) {
    if (&choice != kListening.data()) ImGui::SameLine();  // trois choix courts : une seule ligne
    const bool cinema = std::string_view(choice.key) == "cinema";
    ImGui::BeginDisabled(cinema && !multichannel);
    if (ImGui::RadioButton(choice.label, *key == choice.key)) *key = choice.key;
    ImGui::EndDisabled();
    panel::tip(cinema && !multichannel ? i18n::tr("Greyed out: this device only has 2 channels.")
                                       : choice.detail);
  }
  const int listen = listening_index(*key).value_or(1);
  // Style de son : UN choix qui règle ensemble le profil d'écoute (équilibre, voix, graves, enceintes
  // virtuelles) et le « son anime » (traitement des pistes stéréo : spatialisation, voix, sonie,
  // compresseur), les deux listes de réglages ci-dessous (avancé) s'y rattachent. Propre à l'appareil s'il a
  // son profil propre.
  if (listen != 2) {
    struct Style {
      const char* label;
      const char* detail;
      const char* profile;
      const char* anime;
    };
    static const std::array<Style, 6> kStyles{{
        {"Natural", "The track as mixed, with a light touch.", "natural", "natural"},
        {"Anime", "Crisper voices, wider stereo, controlled loudness: tireless dialogue.", "anime", "anime"},
        {"Strong anime", "Anime, pushed further: voices +3 dB, more bass, firmer compression.", "voices",
         "strong"},
        {"Cinema", "Large virtual room and strong effects, for action.", "cinema", "natural"},
        {"Bass boost", "Deep bass and a little more treble, for action and music.", "bass", "natural"},
        {"Night", "Quiet scenes stay audible, explosions stay down.", "night", "natural"},
    }};
    std::string* profile = &d.audio_profile;
    std::string* level = &d.anime_sound;
    if (own) {
      auto found_p =
          std::ranges::find(d.device_profile, device_name, &std::pair<std::string, std::string>::first);
      if (found_p == d.device_profile.end())
        found_p = d.device_profile.insert(found_p, {device_name, d.audio_profile});
      profile = &found_p->second;
      auto found_a =
          std::ranges::find(d.device_anime, device_name, &std::pair<std::string, std::string>::first);
      if (found_a == d.device_anime.end())
        found_a = d.device_anime.insert(found_a, {device_name, d.anime_sound});
      level = &found_a->second;
    }
    const Style* current = nullptr;
    for (const Style& style : kStyles)
      if (*profile == style.profile && *level == style.anime) current = &style;
    section(i18n::tr("Sound style"));
    if (ImGui::BeginCombo(i18n::tr("Style"),
                          current != nullptr ? i18n::tr(current->label) : i18n::tr("Custom"))) {
      for (const Style& style : kStyles) {
        if (ImGui::Selectable(i18n::tr(style.label), &style == current)) {
          *profile = style.profile;
          *level = style.anime;
          if (const audio::MixPreset* preset = audio::find_preset(style.profile))
            d.audio_custom = preset->tuning;
          if (const audio::AnimeLevel* anime = audio::find_anime_level(style.anime))
            if (anime->tuning.enabled) d.anime_custom = anime->tuning;
        }
        panel::tip(i18n::tr(style.detail));
      }
      ImGui::EndCombo();
    }
    if (current != nullptr) panel::tip(i18n::tr(current->detail));
    note(
        i18n::tr("One choice sets the mix profile and the Anime sound together. Fine control is in Advanced "
                 "settings."));
  }
  if (panel::advanced() && listen != 2) {
    // Le profil de virtualisation : un choix nommé ; en avancé, chaque réglage à la main (Personnalisé). Un
    // appareil qui a son profil propre garde aussi son choix de profil (device_profile) ; sinon c'est le
    // profil global.
    std::string* profile = &d.audio_profile;
    if (own) {
      auto found =
          std::ranges::find(d.device_profile, device_name, &std::pair<std::string, std::string>::first);
      if (found == d.device_profile.end())
        found = d.device_profile.insert(found, {device_name, d.audio_profile});
      profile = &found->second;
    }
    const audio::MixPreset* shown = audio::find_preset(*profile);
    const bool custom = *profile == "custom";
    if (shown == nullptr || (listen == 0 && !shown->headphones) || (listen == 1 && !shown->speakers))
      shown = audio::find_preset("natural");
    if (ImGui::BeginCombo(i18n::tr("Profile"),
                          custom ? i18n::tr("Custom") : std::string(i18n::tr_view(shown->label)).c_str())) {
      for (const audio::MixPreset& preset : audio::mix_presets()) {
        if ((listen == 0 && !preset.headphones) || (listen == 1 && !preset.speakers)) continue;
        if (ImGui::Selectable(std::string(i18n::tr_view(preset.label)).c_str(),
                              !custom && &preset == shown)) {
          *profile = std::string(preset.key);
          d.audio_custom = preset.tuning;  // « Personnalisé » part de ce profil
        }
        panel::tip(std::string(i18n::tr_view(preset.detail)).c_str());
      }
      if (panel::advanced() && ImGui::Selectable(i18n::tr("Custom"), custom)) *profile = "custom";
      ImGui::EndCombo();
    }
    if (!custom) panel::tip(std::string(i18n::tr_view(shown->detail)).c_str());
    if (!custom) {
      panel::note(listen == 0
                      ? i18n::tr("The profile is mostly heard in surround; in stereo it sets the crossfeed.")
                      : i18n::tr("In stereo the profile works through the equalizer and compression; center, "
                                 "rear and bass exist in 5.1 and 7.1 only."));
    }
    if (panel::advanced()) {
      audio::MixTuning t = custom ? d.audio_custom : tuning_of(*profile, d.audio_custom, listen);
      bool moved = false;
      const auto slider = [&](const char* label, float& value, float lo, float hi, const char* format) {
        moved |= ImGui::SliderFloat(label, &value, lo, hi, format, ImGuiSliderFlags_AlwaysClamp);
      };
      if (listen == 0) {
        slider(i18n::tr("Crossfeed"), t.crossfeed_gain, 0.0F, 0.7F, "%.2f");
        slider(i18n::tr("Crossfeed filter (Hz)"), t.crossfeed_cutoff_hz, 300.0F, 1500.0F, "%.0f");
        slider(i18n::tr("Crossfeed delay (ms)"), t.crossfeed_delay_ms, 0.0F, 0.6F, "%.2f");
        slider(i18n::tr("Head size (m)"), t.head_radius_m, 0.07F, 0.11F, "%.3f");
        slider(i18n::tr("Head shadow"), t.head_shadow, 0.0F, 0.7F, "%.2f");
      } else {
        slider(i18n::tr("Center level (voices)"), t.center_lift, 1.0F, 2.0F, "%.2f");
        slider(i18n::tr("Stereo mix gain"), t.downmix_scale, 0.4F, 0.9F, "%.2f");
      }
      slider(i18n::tr("Rear level"), t.surround_gain, 0.5F, 1.6F, "%.2f");
      slider(i18n::tr("Bass level (LFE)"), t.lfe_gain, 0.0F, 0.6F, "%.2f");
      slider(i18n::tr("Speaker distance"), t.distance, 0.0F, 1.0F, "%.2f");
      slider(i18n::tr("Room (reverb)"), t.room, 0.0F, 1.0F, "%.2f");
      slider(i18n::tr("Equalizer: bass (dB)"), t.eq_bass_db, -12.0F, 12.0F, "%.1f");
      slider(i18n::tr("Equalizer: voices (dB)"), t.eq_voice_db, -12.0F, 12.0F, "%.1f");
      slider(i18n::tr("Equalizer: treble (dB)"), t.eq_treble_db, -12.0F, 12.0F, "%.1f");
      slider(i18n::tr("Night compression"), t.night, 0.0F, 1.0F, "%.2f");
      if (listen == 0) slider(i18n::tr("Virtual speakers"), t.virtual_speakers, 0.0F, 1.0F, "%.2f");
      if (moved) {
        d.audio_custom = audio::clamped(t);
        *profile = "custom";
      }
      if (custom && ImGui::Button(i18n::tr("Back to Natural"))) {
        *profile = "natural";
        d.audio_custom = audio::MixTuning{};
      }
    }
  }
  if (panel::advanced() && listen != 2) {
    // « Son anime » (docs/audio-anime.md) : le traitement dédié aux pistes stéréo sur une sortie stéréo.
    // Propre à l'appareil comme le profil de virtualisation.
    section(i18n::tr("Anime sound"));
    std::string* level = &d.anime_sound;
    if (own) {
      auto found =
          std::ranges::find(d.device_anime, device_name, &std::pair<std::string, std::string>::first);
      if (found == d.device_anime.end()) found = d.device_anime.insert(found, {device_name, d.anime_sound});
      level = &found->second;
    }
    const bool custom = *level == "custom";
    const audio::AnimeLevel* shown = audio::find_anime_level(*level);
    if (shown == nullptr) shown = audio::find_anime_level("natural");
    if (ImGui::BeginCombo(i18n::tr("Level"),
                          custom ? i18n::tr("Custom") : std::string(i18n::tr_view(shown->label)).c_str())) {
      for (const audio::AnimeLevel& choice : audio::anime_levels()) {
        if (ImGui::Selectable(std::string(i18n::tr_view(choice.label)).c_str(),
                              !custom && &choice == shown)) {
          *level = std::string(choice.key);
          if (choice.tuning.enabled) d.anime_custom = choice.tuning;  // « Personnalisé » part de ce niveau
        }
        panel::tip(std::string(i18n::tr_view(choice.detail)).c_str());
      }
      if (panel::advanced() && ImGui::Selectable(i18n::tr("Custom"), custom)) {
        *level = "custom";
        if (!d.anime_custom.enabled) d.anime_custom = audio::find_anime_level("anime")->tuning;
      }
      ImGui::EndCombo();
    }
    if (!custom) panel::tip(std::string(i18n::tr_view(shown->detail)).c_str());
    // Comparer : tant que le bouton est tenu, le niveau passe à Naturel ; relâché, il revient.
    static std::string held_level;
    ImGui::Button(i18n::tr("Compare (hold: Natural)"));
    if (ImGui::IsItemActive()) {
      if (held_level.empty() && *level != "natural") {
        held_level = *level;
        *level = "natural";
      }
    } else if (!held_level.empty()) {
      *level = held_level;
      held_level.clear();
    }
    panel::tip(i18n::tr("Hold to hear the track without Anime sound. The volume differs a little."));
    const bool applies = active == nullptr || active->channels <= 2;
    if (applies) {
      note(i18n::tr("Applies to a stereo track on a stereo output only."));
    } else {
      note(i18n::tr("This device has more than 2 channels: Anime sound needs a stereo output."));
    }
    if (panel::advanced()) {
      audio::AnimeTuning t = custom ? d.anime_custom : anime_tuning_of(*level, d.anime_custom);
      if (!t.enabled && !custom) t = d.anime_custom;  // Naturel : les curseurs montrent le dernier niveau
      bool moved = false;
      const auto slider = [&](const char* label, float& value, float lo, float hi, const char* format) {
        moved |= ImGui::SliderFloat(label, &value, lo, hi, format, ImGuiSliderFlags_AlwaysClamp);
      };
      slider(i18n::tr("Spatialization"), t.width, 0.0F, 1.0F, "%.2f");
      slider(i18n::tr("Decorrelated ambience"), t.ambience, 0.0F, 1.0F, "%.2f");
      slider(i18n::tr("Voices: center boost (dB)"), t.voice_db, 0.0F, 4.0F, "%.1f");
      slider(i18n::tr("Voices: 3 kHz presence (dB)"), t.presence_db, 0.0F, 4.0F, "%.1f");
      slider(i18n::tr("Voices forward"), t.voice_forward, 0.0F, 1.0F, "%.2f");
      slider(i18n::tr("Bass boost"), t.bass, 0.0F, 1.0F, "%.2f");
      moved |= ImGui::Checkbox(i18n::tr("Loudness normalization (EBU R128)"), &t.loudness);
      ImGui::BeginDisabled(!t.loudness);
      slider(i18n::tr("Target loudness (LUFS)"), t.target_lufs, -23.0F, -16.0F, "%.0f");
      ImGui::EndDisabled();
      slider(i18n::tr("Soft compressor"), t.compress, 0.0F, 1.0F, "%.2f");
      slider(i18n::tr("Limiter ceiling (dBTP)"), t.ceiling_db, -3.0F, -0.5F, "%.1f");
      if (moved) {
        t.enabled = true;
        d.anime_custom = audio::clamped(t);
        d.anime_custom.enabled = true;
        *level = "custom";
      }
    }
  }
  if (panel::advanced() && listen == 0) {
    // Réponses impulsionnelles à la HeSuVi : remplacent le modèle paramétrique au casque. Propres à
    // l'appareil si celui-ci a son profil propre.
    std::string* path = &d.hrir_path;
    if (own) {
      auto found = std::ranges::find(d.device_hrir, device_name, &std::pair<std::string, std::string>::first);
      if (found == d.device_hrir.end()) found = d.device_hrir.insert(found, {device_name, d.hrir_path});
      path = &found->second;
    }
    static std::array<char, 512> field{};
    static std::string shown;
    if (!ImGui::IsAnyItemActive() && shown != *path) {  // suit l'appareil et les retours en arrière
      std::snprintf(field.data(), field.size(), "%s", path->c_str());
      shown = *path;
    }
    if (ImGui::InputText(i18n::tr("HeSuVi response (.wav)"), field.data(), field.size())) {
      *path = field.data();
      shown = *path;
    }
    ImGui::SameLine();
    if (ImGui::Button(i18n::tr("None##hrir"))) {
      path->clear();
      field[0] = '\0';
      shown.clear();
    }
    panel::tip(
        i18n::tr("A WAV impulse response (HeSuVi 14 ch, 7.1 16 ch or 2 ch HRIR) replaces the headphone "
                 "model. Empty: the default. Adds about 11 ms of latency."));
    note(i18n::tr("No response is bundled: each has its own licence, read the one of the file you pick."));
  }
  if (panel::advanced()) {
    ImGui::Checkbox(i18n::tr("Original (no processing)"), &d.audio_faithful);
    panel::tip(i18n::tr("Turns all processing off: the raw track, simply mixed for the device."));
  }
  if (panel::advanced() && !device_name.empty() &&
      ImGui::Checkbox(i18n::tr("Own profile for this device"), &own)) {
    if (own) {
      d.device_listening.emplace_back(device_name, d.listening);
    } else {
      std::erase_if(d.device_listening, [&](const auto& entry) { return entry.first == device_name; });
      std::erase_if(d.device_profile, [&](const auto& entry) { return entry.first == device_name; });
      std::erase_if(d.device_anime, [&](const auto& entry) { return entry.first == device_name; });
      std::erase_if(d.device_hrir, [&](const auto& entry) { return entry.first == device_name; });
    }
  }
  if (panel::advanced()) panel::tip(i18n::tr("Another device uses the global profile."));

  if (!panel::advanced()) return false;
  section(i18n::tr("Sync"));
  ImGui::SliderInt(i18n::tr("Audio delay (ms)"), &d.audio_delay_ms, -1000, 1000);
  const bool held = ImGui::IsItemActive();
  panel::tip(
      i18n::tr("Positive delays the sound. Ctrl+click the slider to type a value (up to ±10,000 ms)."));
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
