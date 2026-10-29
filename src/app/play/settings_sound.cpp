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
  section("Appareil");
  const SettingsChoices::AudioOut* chosen = nullptr;
  const SettingsChoices::AudioOut* system_default = nullptr;
  for (const SettingsChoices::AudioOut& out : choices.audio_outputs) {
    if (out.is_default) system_default = &out;
    if (!d.audio_device.empty() && out.name == d.audio_device) chosen = &out;
  }
  const SettingsChoices::AudioOut* active = chosen != nullptr ? chosen : system_default;
  bool follow = d.audio_device.empty() || chosen == nullptr;
  if (ImGui::Checkbox("Suivre l'appareil par défaut du système", &follow)) {
    if (follow) {
      d.audio_device.clear();
    } else if (active != nullptr) {
      d.audio_device = active->name;
    }
  }
  ImGui::BeginDisabled(follow);
  if (ImGui::BeginCombo("Appareil de sortie",
                        chosen != nullptr
                            ? chosen->name.c_str()
                            : (system_default != nullptr ? system_default->name.c_str() : "Aucun"))) {
    for (const SettingsChoices::AudioOut& out : choices.audio_outputs) {
      if (ImGui::Selectable(out.name.c_str(), chosen == &out)) d.audio_device = out.name;
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  if (active != nullptr && active->channels > 0)
    note(std::format("{} canaux sur cet appareil.", active->channels));
  panel::tip(
      "Changer d'appareil rouvre la vidéo au même endroit. Le format du son se détecte seul, piste par "
      "piste.");

  // Écoute : le rendu pour le casque, les enceintes ou le home cinéma. Aucune détection : c'est votre choix.
  section("Écoute");
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
  constexpr std::array<Choice, 3> kListening{{
      {"headphones", "Casque", "Crossfeed en stéréo, binaural en surround."},
      {"speakers", "Enceintes", "Rendu fidèle ; le surround est mixé en stéréo, centre rehaussé."},
      {"cinema", "Home cinéma", "Multicanal, surround envoyé tel quel."},
  }};
  for (const Choice& choice : kListening) {
    if (&choice != kListening.data()) ImGui::SameLine();  // trois choix courts : une seule ligne
    const bool cinema = std::string_view(choice.key) == "cinema";
    ImGui::BeginDisabled(cinema && !multichannel);
    if (ImGui::RadioButton(choice.label, *key == choice.key)) *key = choice.key;
    ImGui::EndDisabled();
    panel::tip(cinema && !multichannel ? "Grisé : cet appareil n'a que 2 canaux." : choice.detail);
  }
  const int listen = listening_index(*key).value_or(1);
  if (listen != 2) {
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
    if (ImGui::BeginCombo("Profil", custom ? "Personnalisé" : std::string(shown->label).c_str())) {
      for (const audio::MixPreset& preset : audio::mix_presets()) {
        if ((listen == 0 && !preset.headphones) || (listen == 1 && !preset.speakers)) continue;
        if (ImGui::Selectable(std::string(preset.label).c_str(), !custom && &preset == shown)) {
          *profile = std::string(preset.key);
          d.audio_custom = preset.tuning;  // « Personnalisé » part de ce profil
        }
        panel::tip(std::string(preset.detail).c_str());
      }
      if (panel::advanced() && ImGui::Selectable("Personnalisé", custom)) *profile = "custom";
      ImGui::EndCombo();
    }
    if (!custom) panel::tip(std::string(shown->detail).c_str());
    if (!custom) {
      panel::note(listen == 0
                      ? "Le profil s'entend surtout en surround ; en stéréo, il règle le crossfeed."
                      : "En stéréo, le profil agit par l'égaliseur et la compression ; centre, arrière et "
                        "basses n'existent qu'en 5.1 et 7.1.");
    }
    if (panel::advanced()) {
      audio::MixTuning t = custom ? d.audio_custom : tuning_of(*profile, d.audio_custom, listen);
      bool moved = false;
      const auto slider = [&](const char* label, float& value, float lo, float hi, const char* format) {
        moved |= ImGui::SliderFloat(label, &value, lo, hi, format, ImGuiSliderFlags_AlwaysClamp);
      };
      if (listen == 0) {
        slider("Crossfeed (stéréo)", t.crossfeed_gain, 0.0F, 0.7F, "%.2f");
        slider("Crossfeed : filtre (Hz)", t.crossfeed_cutoff_hz, 300.0F, 1500.0F, "%.0f");
        slider("Crossfeed : retard (ms)", t.crossfeed_delay_ms, 0.0F, 0.6F, "%.2f");
        slider("Taille de la tête (m)", t.head_radius_m, 0.07F, 0.11F, "%.3f");
        slider("Ombre de la tête", t.head_shadow, 0.0F, 0.7F, "%.2f");
      } else {
        slider("Niveau du centre (voix)", t.center_lift, 1.0F, 2.0F, "%.2f");
        slider("Gain du mixage stéréo", t.downmix_scale, 0.4F, 0.9F, "%.2f");
      }
      slider("Niveau des arrières", t.surround_gain, 0.5F, 1.6F, "%.2f");
      slider("Niveau des basses (LFE)", t.lfe_gain, 0.0F, 0.6F, "%.2f");
      slider("Distance des enceintes", t.distance, 0.0F, 1.0F, "%.2f");
      slider("Pièce (réverbération)", t.room, 0.0F, 1.0F, "%.2f");
      slider("Égaliseur : graves (dB)", t.eq_bass_db, -8.0F, 8.0F, "%.1f");
      slider("Égaliseur : voix (dB)", t.eq_voice_db, -8.0F, 8.0F, "%.1f");
      slider("Égaliseur : aigus (dB)", t.eq_treble_db, -8.0F, 8.0F, "%.1f");
      slider("Nuit : compression", t.night, 0.0F, 1.0F, "%.2f");
      if (moved) {
        d.audio_custom = audio::clamped(t);
        *profile = "custom";
      }
      if (custom && ImGui::Button("Revenir à Naturel")) {
        *profile = "natural";
        d.audio_custom = audio::MixTuning{};
      }
    }
  }
  if (listen != 2) {
    // « Son anime » (docs/audio-anime.md) : le traitement dédié aux pistes stéréo sur une sortie stéréo.
    // Propre à l'appareil comme le profil de virtualisation.
    section("Son anime");
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
    if (ImGui::BeginCombo("Niveau", custom ? "Personnalisé" : std::string(shown->label).c_str())) {
      for (const audio::AnimeLevel& choice : audio::anime_levels()) {
        if (ImGui::Selectable(std::string(choice.label).c_str(), !custom && &choice == shown)) {
          *level = std::string(choice.key);
          if (choice.tuning.enabled) d.anime_custom = choice.tuning;  // « Personnalisé » part de ce niveau
        }
        panel::tip(std::string(choice.detail).c_str());
      }
      if (panel::advanced() && ImGui::Selectable("Personnalisé", custom)) {
        *level = "custom";
        if (!d.anime_custom.enabled) d.anime_custom = audio::find_anime_level("anime")->tuning;
      }
      ImGui::EndCombo();
    }
    if (!custom) panel::tip(std::string(shown->detail).c_str());
    // Comparer : tant que le bouton est tenu, le niveau passe à Naturel ; relâché, il revient.
    static std::string held_level;
    ImGui::Button("Comparer (maintenir : Naturel)");
    if (ImGui::IsItemActive()) {
      if (held_level.empty() && *level != "natural") {
        held_level = *level;
        *level = "natural";
      }
    } else if (!held_level.empty()) {
      *level = held_level;
      held_level.clear();
    }
    panel::tip(
        "Tenez le bouton pour entendre la piste sans le Son anime, relâchez pour le retrouver. Le volume "
        "diffère un peu : la normalisation de sonie rapproche aussi les scènes fortes et faibles.");
    const bool applies = active == nullptr || active->channels <= 2;
    note(applies
             ? "Actif pour une piste stéréo sur une sortie stéréo. Une piste mono, 5.1 ou 7.1, le mode Home "
               "cinéma et le mode Fidèle gardent leur chemin habituel (aucune dégradation)."
             : "Cet appareil a plus de 2 canaux : le Son anime ne s'applique qu'à une sortie stéréo.");
    if (panel::advanced()) {
      audio::AnimeTuning t = custom ? d.anime_custom : anime_tuning_of(*level, d.anime_custom);
      if (!t.enabled && !custom) t = d.anime_custom;  // Naturel : les curseurs montrent le dernier niveau
      bool moved = false;
      const auto slider = [&](const char* label, float& value, float lo, float hi, const char* format) {
        moved |= ImGui::SliderFloat(label, &value, lo, hi, format, ImGuiSliderFlags_AlwaysClamp);
      };
      slider("Spatialisation", t.width, 0.0F, 1.0F, "%.2f");
      slider("Ambiance décorrélée", t.ambience, 0.0F, 1.0F, "%.2f");
      slider("Voix : centre porté (dB)", t.voice_db, 0.0F, 4.0F, "%.1f");
      slider("Voix : présence 3 kHz (dB)", t.presence_db, 0.0F, 4.0F, "%.1f");
      slider("Voix en avant", t.voice_forward, 0.0F, 1.0F, "%.2f");
      slider("Renfort des graves", t.bass, 0.0F, 1.0F, "%.2f");
      moved |= ImGui::Checkbox("Normalisation de sonie (EBU R128)", &t.loudness);
      ImGui::BeginDisabled(!t.loudness);
      slider("Sonie visée (LUFS)", t.target_lufs, -23.0F, -16.0F, "%.0f");
      ImGui::EndDisabled();
      slider("Compresseur doux", t.compress, 0.0F, 1.0F, "%.2f");
      slider("Plafond du limiteur (dBTP)", t.ceiling_db, -3.0F, -0.5F, "%.1f");
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
    if (ImGui::InputText("Réponse HeSuVi (.wav)", field.data(), field.size())) {
      *path = field.data();
      shown = *path;
    }
    ImGui::SameLine();
    if (ImGui::Button("Aucune##hrir")) {
      path->clear();
      field[0] = '\0';
      shown.clear();
    }
    panel::tip(
        "Un fichier WAV de réponses (14 canaux HeSuVi, 16 canaux 7.1 ou 2 canaux HRIR) remplace le modèle de "
        "tête du casque, pour le stéréo comme pour le 5.1 et le 7.1. Vide : le modèle d'origine. Latence "
        "ajoutée : environ 11 ms.");
    note(
        "Le lecteur ne livre aucune réponse : elles ont chacune leur licence, lisez celle du fichier que "
        "vous "
        "choisissez.");
  }
  if (panel::advanced()) {
    ImGui::Checkbox("Fidèle (aucun traitement)", &d.audio_faithful);
    panel::tip(
        "Coupe tout traitement, pour comparer : le son brut de la piste, mixé simplement pour l'appareil.");
  }
  if (panel::advanced() && !device_name.empty() && ImGui::Checkbox("Profil propre à cet appareil", &own)) {
    if (own) {
      d.device_listening.emplace_back(device_name, d.listening);
    } else {
      std::erase_if(d.device_listening, [&](const auto& entry) { return entry.first == device_name; });
      std::erase_if(d.device_profile, [&](const auto& entry) { return entry.first == device_name; });
      std::erase_if(d.device_anime, [&](const auto& entry) { return entry.first == device_name; });
      std::erase_if(d.device_hrir, [&](const auto& entry) { return entry.first == device_name; });
    }
  }
  if (panel::advanced()) panel::tip("Un autre appareil reprend le profil global, sans rien demander.");

  if (!panel::advanced()) return false;
  section("Synchronisation");
  ImGui::SliderInt("Délai audio (ms)", &d.audio_delay_ms, -1000, 1000);
  const bool held = ImGui::IsItemActive();
  panel::tip(
      "Positif : le son est retardé. Ctrl + clic sur le curseur pour saisir une valeur (jusqu'à ±10 000 "
      "ms).");
  if (ImGui::Button("Remettre à 0")) d.audio_delay_ms = 0;
  d.audio_delay_ms = std::clamp(d.audio_delay_ms, limits::kAudioDelayMs.lo, limits::kAudioDelayMs.hi);
  ImGui::Checkbox("Caler la lecture sur l'écran", &d.display_sync);
  panel::tip(
      "Sur un écran à fréquence fixe, la lecture accélère ou ralentit d'au plus 0,125 % pour que chaque "
      "image tienne un nombre entier de rafraîchissements : plus aucune image doublée par dérive (une toutes "
      "les 17 s à 60 Hz sinon). Le son suit, sans changement audible. Sans calage possible (VRR, cadence "
      "trop loin), le son reste maître.");
  ImGui::SliderInt("Latence de l'écran (ms)", &d.video_latency_ms, limits::kVideoLatencyMs.lo,
                   limits::kVideoLatencyMs.hi, d.video_latency_ms == 0 ? "estimée" : "%d ms",
                   ImGuiSliderFlags_AlwaysClamp);
  note(
      "Le temps entre le choix d'une image et son apparition : estimé d'après le rendu et la file de "
      "l'écran, ou imposé ici si l'image paraît en avance ou en retard sur le son.");
  d.video_latency_ms = std::clamp(d.video_latency_ms, limits::kVideoLatencyMs.lo, limits::kVideoLatencyMs.hi);
  return held;
}

}  // namespace ajp::app::play
