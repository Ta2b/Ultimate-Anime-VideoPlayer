#include "app/play/settings_panel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>

#include "app/play/auto_config.h"
#include "app/play/panel_style.h"
#include "app/play/settings_chain_tabs.h"
#include "app/play/settings_engines.h"
#include "app/play/settings_export.h"
#include "app/play/text_fit.h"
#include "app/play/tracks.h"

namespace ajp::app::play {
namespace {

using panel::kWarn;
using panel::note;
using panel::section;

// Le moteur du mode HDR : RTX HDR (TrueHDR, NVIDIA), HDR adaptatif (tous GPU : libplacebo, puis
// l'aspect TrueHDR émulé qui suit la scène), ou SDR.
constexpr std::array<const char*, 3> kRoutes{"RTX HDR (NVIDIA)", "HDR adaptatif (tous GPU)", "SDR"};

void hdr_tab(PlaySettings& d, const SettingsChoices& choices) {
  // Un écran SDR n'a pas de HDR à convertir : les options restent visibles mais grisées.
  ImGui::BeginDisabled(!choices.hdr_output);
  section("Sortie");
  const int route = d.true_hdr ? 0 : (d.hdr_native ? 1 : 2);
  if (ImGui::BeginCombo("Moteur HDR", kRoutes[static_cast<std::size_t>(route)])) {
    for (int i = 0; i < static_cast<int>(kRoutes.size()); ++i) {
      if (i == 0 && !choices.nvidia) continue;  // TrueHDR ne tourne que sur un GPU NVIDIA
      ImGui::BeginDisabled(i == 0 && !choices.true_hdr_available);
      if (ImGui::Selectable(kRoutes[static_cast<std::size_t>(i)], i == route)) {
        d.true_hdr = i == 0;
        d.hdr_native = i == 1;
      }
      ImGui::EndDisabled();
    }
    ImGui::EndCombo();
  }
  if (choices.nvidia && !choices.true_hdr_backend)
    note(
        "RTX HDR demande le moteur TensorRT (écran des moteurs) : sous Vulkan, le HDR adaptatif le "
        "remplace.");
  else if (choices.nvidia && !choices.true_hdr_available)
    note("RTX HDR : le SDK Maxine VideoFX n'est pas installé (écran des moteurs).");
  if (!choices.hdr_output)
    note(
        "Écran SDR : le HDR est indisponible ; l'image reste en SDR et le grade (onglet Couleurs) imite le "
        "rendu HDR.");
  section(choices.nvidia ? "Réglages du mode HDR (RTX HDR et HDR adaptatif)" : "Réglages du HDR adaptatif");
  bool automatic = d.true_hdr_peak == 0;
  if (ImGui::Checkbox(std::format("Pic automatique ({} nits)", choices.auto_peak).c_str(), &automatic))
    d.true_hdr_peak =
        automatic ? 0 : std::clamp(choices.auto_peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi);
  ImGui::BeginDisabled(automatic);
  int peak = automatic ? choices.auto_peak : d.true_hdr_peak;
  if (ImGui::SliderInt("Pic (nits)", &peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi, "%d",
                       ImGuiSliderFlags_AlwaysClamp) &&
      !automatic)
    d.true_hdr_peak = peak;
  ImGui::EndDisabled();
  if (panel::advanced())
    ImGui::SliderInt("Gris moyen", &d.true_hdr_gray, limits::kTrueHdrGray.lo, limits::kTrueHdrGray.hi, "%d",
                     ImGuiSliderFlags_AlwaysClamp);
  if (choices.nvidia && panel::advanced()) {
    ImGui::BeginDisabled(!d.true_hdr);
    ImGui::Checkbox("Débandage (RTX HDR)", &d.true_hdr_debanding);
    ImGui::EndDisabled();
  }
  if (panel::advanced() && ImGui::Button("Valeurs par défaut")) {
    const PlaySettings defaults;
    d.true_hdr_peak = defaults.true_hdr_peak;
    d.true_hdr_gray = defaults.true_hdr_gray;
    d.true_hdr_debanding = defaults.true_hdr_debanding;
  }
  note(std::format(
      "Réglages appliqués en direct{}. Contraste et saturation sont dans l'onglet Grade, communs au HDR et "
      "au SDR. Gris moyen : luminosité des tons moyens. Le HDR adaptatif suit la scène : moins d'expansion "
      "sur les scènes claires et les grands aplats blancs, ombres plus douces la nuit, éclats isolés vers le "
      "pic.",
      choices.nvidia ? ", sur RTX HDR comme sur le HDR adaptatif" : ""));
  ImGui::EndDisabled();
}

// Un curseur sur un réglage du grade (double) ; true s'il a bougé. La valeur revient arrondie au
// centième et bornée : le float du curseur à 0,8 vaut 0,800000012 en double, au-dessus de la borne
// de check_grade_settings, et une valeur saisie au clavier (Ctrl + clic) peut sortir de la plage ;
// l'une comme l'autre faisait refuser tout le grade.
bool grade_slider(const char* label, double& value, core::Range<double> range) {
  auto shown = static_cast<float>(value);
  if (!ImGui::SliderFloat(label, &shown, static_cast<float>(range.lo), static_cast<float>(range.hi), "%.2f",
                          ImGuiSliderFlags_AlwaysClamp))
    return false;
  value = std::clamp(std::round(static_cast<double>(shown) * 100.0) / 100.0, range.lo, range.hi);
  return true;
}

// Contraste et saturation : ceux de TrueHDR, que le HDR adaptatif émule, et que la sortie SDR
// gradée reprend autour du gris moyen (render::sdr_params) ; un seul réglage pour les trois.
void look_section(PlaySettings& d) {
  section("Contraste et saturation (HDR et SDR)");
  ImGui::SliderInt("Contraste", &d.true_hdr_contrast, limits::kTrueHdrContrast.lo,
                   limits::kTrueHdrContrast.hi, "%d", ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderInt("Saturation", &d.true_hdr_saturation, limits::kTrueHdrSaturation.lo,
                   limits::kTrueHdrSaturation.hi, "%d", ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::Button("Contraste et saturation par défaut")) {
    const PlaySettings defaults;
    d.true_hdr_contrast = defaults.true_hdr_contrast;
    d.true_hdr_saturation = defaults.true_hdr_saturation;
  }
  note(
      "En direct, sur RTX HDR, le HDR adaptatif et la sortie SDR : en SDR, la même courbe que TrueHDR "
      "autour du gris moyen (onglet HDR), sans les éclats au-dessus du blanc.");
  if (!d.grade && !d.true_hdr) note("Hors RTX HDR, ils demandent « Grader la sortie » ci-dessous.");
}

void grade_tab(PlaySettings& d, const SettingsChoices& choices) {
  // Le grade s'applique sur toutes les routes, le mode SDR compris : il imite le rendu HDR (contraste,
  // couleurs riches, hautes lumières) pour ceux qui n'ont pas d'écran HDR. L'onglet reste actif.
  look_section(d);
  section("Grade");
  ImGui::Checkbox("Grader la sortie (HDR10 et SDR)", &d.grade);
  ImGui::BeginDisabled(!d.grade);
  if (ImGui::BeginCombo("Préréglage", std::string(render::preset_label(d.grade_preset)).c_str())) {
    for (int i = 0; i < render::kGradePresetCount; ++i) {
      const auto preset = static_cast<render::GradePreset>(i);
      if (ImGui::Selectable(std::string(render::preset_label(preset)).c_str(), preset == d.grade_preset)) {
        d.grade_preset = preset;
        d.grade_values = with_preset(preset, d.grade_values);
      }
    }
    ImGui::EndCombo();
  }
  render::GradeSettings& g = d.grade_values;
  using namespace render::grade_limits;
  bool moved = false;
  if (panel::advanced()) {
    moved |= grade_slider("Éclats", g.highlights_rtx, kHighlights);
    moved |= grade_slider("Aplats lumineux", g.area_rtx, kAreaRtx);
    moved |= grade_slider("Couleurs", g.vibrance_rtx, kVibrance);
    moved |= grade_slider("Couleurs étendues (large gamut)", g.expand, kExpand);
    moved |= grade_slider("Noirs profonds", g.shadows, kShadows);
    moved |= grade_slider("Garde-fou couleurs saturées", g.guard, kGuard);
    if (moved) d.grade_preset = render::GradePreset::Custom;
    ImGui::Checkbox("Gamut Display P3 de l'écran", &g.gamut_p3);
  }
  ImGui::EndDisabled();
  if (d.true_hdr) {
    note("Route RTX HDR : TrueHDR, puis le grade sur la sortie PQ.");
  } else if (d.hdr_native) {
    note("HDR adaptatif : libplacebo en PQ, aspect TrueHDR émulé selon la scène, puis le grade.");
  } else {
    note(choices.hdr_output
             ? "Mode SDR : le grade imite le rendu HDR sur l'image SDR, dans le conteneur HDR10 de l'écran."
             : "Mode SDR : le grade imite le rendu HDR dans la gamme SDR (contraste, couleurs riches, hautes "
               "lumières), "
               "pour profiter du lecteur sans écran HDR. Le préréglage Éclatant l'accentue.");
  }
}

// Composition finale (Image IA > Agrandissement, réglages avancés) : le filtre vers la fenêtre et le
// débandage.
void composition_section(PlaySettings& d) {
  section("Composition finale");
  if (ImGui::BeginCombo("Filtre vers la fenêtre", std::string(label(d.final_scaler)).c_str())) {
    for (int i = 0; i < kFinalScalerCount; ++i) {
      const auto scaler = static_cast<FinalScaler>(i);
      if (ImGui::Selectable(std::string(label(scaler)).c_str(), scaler == d.final_scaler))
        d.final_scaler = scaler;
    }
    ImGui::EndCombo();
  }
  ImGui::Checkbox("Débander les aplats et dégradés", &d.deband);
  note(
      "Le filtre étire la sortie de la chaîne (par exemple 1440p) vers la fenêtre ou l'écran 4K ; EWA "
      "Lanczos est le plus fin, plus lourd pour le GPU. Le débandage adoucit les marches des aplats et "
      "des dégradés, courants en anime, sans toucher aux contours. Aucun effet sur l'export.");
}

// Noir absolu et soins OLED (HDR > Conversion, réglages avancés).
void oled_sections(PlaySettings& d, const SettingsChoices& choices) {
  section("Noir absolu OLED");
  if (ImGui::BeginCombo("Noir absolu", std::string(label(d.oled_black)).c_str())) {
    for (int i = 0; i < kOledBlackCount; ++i) {
      const auto mode = static_cast<OledBlack>(i);
      if (ImGui::Selectable(std::string(label(mode)).c_str(), mode == d.oled_black)) d.oled_black = mode;
    }
    ImGui::EndCombo();
  }
  ImGui::BeginDisabled(d.oled_black == OledBlack::Off);
  grade_slider("Seuil (pas de code)", d.oled_black_steps, render::grade_limits::kBlackSteps);
  ImGui::EndDisabled();
  note(("Écran : " + choices.display_text).c_str());
  section("Soins OLED");
  ImGui::Checkbox("Effacer la barre en pause (anti-marquage)", &d.oled_care);
  ImGui::Checkbox("Cadence fixe (VRR sans scintillement)", &d.steady_cadence);
  note(
      "Anti-marquage : sur un écran OLED, la barre s'efface aussi en pause après 20 s sans activité. "
      "Cadence fixe : l'image est représentée à chaque rafraîchissement de l'écran, pour que le VRR reste à "
      "fréquence constante ; une fréquence qui varie fait scintiller les scènes sombres d'un OLED. Coûte une "
      "composition par rafraîchissement.");
  if (!choices.black_status.empty()) note(choices.black_status.c_str());
  note(
      "Les quasi-noirs deviennent un vrai noir (0) ; le pied rejoint l'image au seuil, compté en pas de code "
      "au-dessus du noir de la source. Calé en direct sur une rampe de 32 marches passée par RTX HDR et le "
      "grade ; même gain sur R, V et B, et le maximum local 8x8 évite le scintillement.");
  ImGui::Spacing();
  note(
      "Éclatant : éclats et aplats lumineux, couleurs et noirs poussés, teinte gardée (même gain sur R, "
      "V et B) ; le garde-fou adoucit les couleurs saturées près du pic au lieu de les écrêter. "
      "Naturel : le grade du lecteur actuel. Tout s'applique en direct.");
}

std::string language_label(std::string_view tag) {
  std::string name = language_name(tag);
  return name.empty() ? std::string(tag) : name;  // une balise écrite à la main dans play.toml
}

// Une langue au choix : `none_label` pour la valeur vide (audio : « Selon le fichier », sous-titres :
// « Aucun »), sinon une des langues du menu.
void language_combo(const char* label, const char* none_label, std::string& tag) {
  const std::string current = tag.empty() ? none_label : language_label(tag);
  if (!ImGui::BeginCombo(label, current.c_str())) return;
  if (ImGui::Selectable(none_label, tag.empty())) tag.clear();
  for (const Language& language : menu_languages()) {
    if (ImGui::Selectable(std::string(language.name).c_str(), same_language(tag, language.code)))
      tag = std::string(language.code);
  }
  ImGui::EndCombo();
}

// Deux préférences par ordre : l'audio et les sous-titres voulus ensemble ; la 2e sert quand la 1re
// n'est pas entière dans le fichier.
void languages_tab(PlaySettings& d) {
  d.language_presets.resize(limits::kLanguagePresets);
  for (std::size_t rank = 0; rank < d.language_presets.size(); ++rank) {
    ImGui::PushID(static_cast<int>(rank));
    section(rank == 0 ? "Préférence 1" : "Préférence 2, si la 1re n'est pas disponible");
    language_combo("Langue de l'audio", "Selon le fichier", d.language_presets[rank].audio);
    language_combo("Langue des sous-titres", "Aucun", d.language_presets[rank].subtitles);
    ImGui::PopID();
  }
  note(
      "À l'ouverture d'un fichier : la première préférence dont l'audio ET les sous-titres existent est "
      "prise telle quelle ; « Aucun » ne montre aucun sous-titre. Sinon, la deuxième ; sinon la meilleure "
      "langue de chaque liste, puis les pistes par défaut du fichier. Une piste sans langue est reconnue "
      "par son titre (VF, VOSTFR, Japanese…). Vaut au prochain fichier ; les pistes en cours se changent "
      "dans la barre.");
}

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
    if (panel::advanced()) {
      audio::MixTuning t = custom ? d.audio_custom : tuning_of(*profile, d.audio_custom, listen);
      bool moved = false;
      const auto slider = [&](const char* label, float& value, float lo, float hi, const char* format) {
        moved |= ImGui::SliderFloat(label, &value, lo, hi, format, ImGuiSliderFlags_AlwaysClamp);
      };
      if (listen == 0) {
        slider("Crossfeed (stéréo)", t.crossfeed_gain, 0.0F, 0.7F, "%.2f");
        slider("Crossfeed : adoucissement (Hz)", t.crossfeed_cutoff_hz, 300.0F, 1500.0F, "%.0f");
        slider("Crossfeed : retard (ms)", t.crossfeed_delay_ms, 0.0F, 0.6F, "%.2f");
        slider("Taille de la tête (m)", t.head_radius_m, 0.07F, 0.11F, "%.3f");
        slider("Ombre de la tête", t.head_shadow, 0.0F, 0.7F, "%.2f");
      } else {
        slider("Niveau du centre (voix)", t.center_lift, 1.0F, 2.0F, "%.2f");
        slider("Gain du mixage stéréo", t.downmix_scale, 0.4F, 0.9F, "%.2f");
      }
      slider("Niveau des arrières et côtés", t.surround_gain, 0.5F, 1.6F, "%.2f");
      slider("Niveau des basses (LFE)", t.lfe_gain, 0.0F, 0.6F, "%.2f");
      slider("Distance des enceintes virtuelles", t.distance, 0.0F, 1.0F, "%.2f");
      slider("Pièce (réverbération légère)", t.room, 0.0F, 1.0F, "%.2f");
      slider("Égaliseur : graves (dB)", t.eq_bass_db, -8.0F, 8.0F, "%.1f");
      slider("Égaliseur : voix (dB)", t.eq_voice_db, -8.0F, 8.0F, "%.1f");
      slider("Égaliseur : aigus (dB)", t.eq_treble_db, -8.0F, 8.0F, "%.1f");
      slider("Nuit : compression de la dynamique", t.night, 0.0F, 1.0F, "%.2f");
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
    if (ImGui::InputText("Réponse impulsionnelle (HeSuVi, .wav)", field.data(), field.size())) {
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
      std::erase_if(d.device_hrir, [&](const auto& entry) { return entry.first == device_name; });
    }
  }
  if (panel::advanced()) panel::tip("Un autre appareil reprend le profil global, sans rien demander.");

  section("Lecture");
  ImGui::Checkbox("Lire l'épisode suivant à la fin", &d.autoplay_next);
  panel::tip(
      "À la fin d'une vidéo ouverte par un fichier, lit la suivante du même dossier (ordre naturel des "
      "noms : ep2 avant ep10). N : épisode suivant, Maj + N : précédent, O : passer le générique "
      "(chapitre Opening ou Ending, sinon 85 s).");
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

void subtitles_tab(PlaySettings& d) {
  section("Taille");
  int percent = static_cast<int>(std::lround(d.subtitle_scale * 100.0F));
  if (ImGui::SliderInt("Taille (%)", &percent, static_cast<int>(limits::kSubtitleScale.lo * 100.0F),
                       static_cast<int>(limits::kSubtitleScale.hi * 100.0F), "%d",
                       ImGuiSliderFlags_AlwaysClamp)) {
    d.subtitle_scale = static_cast<float>(percent) / 100.0F;
  }
  if (ImGui::Button("Taille d'origine")) d.subtitle_scale = 1.0F;
  note(
      "Agrandit ou réduit toutes les polices, panneaux compris (comme la taille de mpv). Changement en "
      "direct ; la piste se choisit dans la barre (j).");
}

void stats_tab(PlaySettings& d) {
  section("Statistiques à l'écran");
  ImGui::Checkbox("Toutes les statistiques (Tab)", &d.stats.all);
  ImGui::Spacing();
  ImGui::BeginDisabled(d.stats.all);
  ImGui::Checkbox("Chaîne (modèles, tailles, moteur)", &d.stats.chain);
  ImGui::Checkbox("Images (temps par image, sautées)", &d.stats.frames);
  ImGui::Checkbox("RIFE (interpolées, tenues, coupes)", &d.stats.rife);
  ImGui::Checkbox("Pistes", &d.stats.tracks);
  ImGui::Checkbox("Sortie HDR", &d.stats.output);
  ImGui::Checkbox("Position exacte", &d.stats.position);
  ImGui::Checkbox("Mémoire vidéo", &d.stats.memory);
  ImGui::EndDisabled();
}

// Général > Configuration : le préréglage, la priorité, la machine, « Évaluer ma machine ».
bool config_page(PlaySettings& d, const SettingsChoices& choices, BenchNote& bench,
                 std::vector<Input>& actions) {
  section("Configuration automatique");
  if (ImGui::BeginCombo("Préréglage", std::string(label(d.preset)).c_str())) {
    for (int i = 0; i < kPresetCount; ++i) {
      const auto preset = static_cast<Preset>(i);
      if (ImGui::Selectable(std::string(label(preset)).c_str(), preset == d.preset)) {
        d.preset = preset;
        // La chaîne du préréglage s'écrit tout de suite (Personnalisé : rien ne bouge).
        d = apply_preset(
            d,
            Machine{.tier = static_cast<Tier>(choices.machine_tier),
                    .display_height = choices.display_height,
                    .display_hz = choices.display_hz},
            core::SourceInfo{
                .width = choices.source_width, .height = choices.source_height, .fps = choices.source_fps});
      }
    }
    ImGui::EndCombo();
  }
  panel::tip(
      "Auto choisit la chaîne selon votre carte, votre écran et la vidéo. Régler un réglage de l'image ou de "
      "la fluidité à la main passe en Personnalisé, sans rien perdre.");
  if (ImGui::BeginCombo("Priorité", std::string(label(d.priority)).c_str())) {
    for (int i = 0; i < kPriorityCount; ++i) {
      const auto priority = static_cast<Priority>(i);
      if (ImGui::Selectable(std::string(label(priority)).c_str(), priority == d.priority))
        d.priority = priority;
    }
    ImGui::EndCombo();
  }
  panel::tip(
      "Ce que l'automatique préfère quand la machine ne tient pas tout : plus d'images (Fluidité) ou une "
      "image plus fine (Image).");

  section("Cette machine");
  ImGui::TextWrapped("%s", choices.gpu_name.empty() ? "Carte graphique inconnue" : choices.gpu_name.c_str());
  note(std::format(
      "Niveau : {}{}", tier_label(static_cast<Tier>(choices.machine_tier)),
      choices.display_hz > 0.0 ? std::format(" · écran {:.0f} Hz", choices.display_hz) : std::string()));
  note(choices.bench_text.empty()
           ? std::string("Jamais évaluée : le niveau est une estimation d'après la carte.")
           : choices.bench_text);
  const bool started = bench_page(choices.place, choices.source_file, bench);

  section("Assistant");
  if (ImGui::Button("Relancer l'assistant")) actions.emplace_back(OpenEngineScreen{.wizard = true});
  panel::tip(
      "Les huit écrans de bienvenue : moteurs, image, HDR, audio, langues, mises à jour. Rien ne change "
      "avant « Terminer ».");
  return started;
}

// Général > Réinitialiser.
void reset_page(std::vector<Input>& actions, const ProfilesView& view) {
  section("Profil d'origine");
  note(
      "Les profils (Ctrl+1 à Ctrl+9) restent dans user.toml ; cette page remet le profil actif à ses valeurs "
      "d'origine.");
  if (ImGui::Button("Revenir au profil d'origine"))
    actions.emplace_back(EditProfile{.edit = core::edit::ResetProfile{.slot = view.active}});
}

struct PageInfo {
  const char* category;
  const char* name;
  bool advanced;  // seulement avec la case « Réglages avancés »
};
// L'arbre : l'ordre est celui de l'écran. Les catégories se déduisent du changement de `category`.
constexpr std::array<PageInfo, 14> kPages{{
    {"Général", "Configuration", false},
    {"Général", "Lecteur par défaut", false},  // masquée quand « Ne plus me le demander » est coché
    {"Général", "Mises à jour", false},
    {"Général", "Langues", false},
    {"Général", "Sous-titres", false},
    {"Général", "Statistiques", false},
    {"Général", "Réinitialiser", false},
    {"Image IA", "Agrandissement", false},
    {"Image IA", "Fluidité", false},
    {"HDR et couleurs", "Conversion HDR", false},
    {"HDR et couleurs", "Réglages", false},
    {"Audio", "Sortie", false},
    {"Export et moteurs", "Fichier", false},
    {"Export et moteurs", "Moteurs installés", false},
}};

}  // namespace

void language_choice(const char* label, const char* none_label, std::string& tag) {
  language_combo(label, none_label, tag);
}

SettingsPanel::SettingsPanel() : export_(std::make_unique<ExportDraft>()) {}
SettingsPanel::~SettingsPanel() = default;

SettingsPanel::Result SettingsPanel::draw(const PlaySettings& running, const SettingsChoices& choices,
                                          float scale, std::vector<Input>& actions) {
  Result result;
  if (!prompt_done_ && choices.offer_default_player && !running.default_player_dismissed &&
      default_player_prompt(choices.place, prompt_note_, prompt_dismissed_, scale)) {
    prompt_done_ = true;
    if (prompt_dismissed_) {
      PlaySettings quiet = running;
      quiet.default_player_dismissed = true;
      result.live = quiet;
    }
  }
  if (!open_) return result;
  // Réglages changés ailleurs (Tab, volume) : le panneau les reprend, sans perdre ce qui attend.
  if (running != seen_) {
    draft_ = live_part(draft_, running, choices.rife_loaded);
    seen_ = running;
  }
  const ImGuiIO& io = ImGui::GetIO();
  const float pad = 18.0F * scale;
  const float width = std::min(760.0F * scale, io.DisplaySize.x - 2.0F * pad);
  const float bottom = 120.0F * scale;  // la barre de lecture reste visible dessous
  ImGui::SetNextWindowPos({io.DisplaySize.x - pad, pad + 40.0F * scale}, ImGuiCond_Always, {1.0F, 0.0F});
  ImGui::SetNextWindowSize({width, std::max(240.0F * scale, io.DisplaySize.y - pad - 40.0F * scale - bottom)},
                           ImGuiCond_Always);
  bool keep = true;
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
                                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
  if (ImGui::Begin("Paramètres", &keep, kFlags)) {
    ImGui::PushItemWidth(-width * 0.42F);
    panel::advanced() = advanced_;
    bool delay_held = false;
    const PlaySettings before = draft_;
    // Sous les deux colonnes : la case « Réglages avancés » (et le bandeau « Appliquer » quand il est là).
    const float footer = (pending_shown_ ? 84.0F * scale : 0.0F) + ImGui::GetFrameHeightWithSpacing() +
                         2.0F * ImGui::GetStyle().ItemSpacing.y;
    const float tree_width = std::min(200.0F * scale, width * 0.3F);
    if ((!advanced_ && kPages[static_cast<std::size_t>(page_)].advanced) ||
        (page_ == 1 && draft_.default_player_dismissed && !advanced_))
      page_ = 0;
    // Une page masquée : réglage avancé sans la case, ou « Lecteur par défaut » une fois refusé.
    const auto visible = [&](std::size_t i) {
      if (kPages[i].advanced && !advanced_) return false;
      return !(i == 1 && draft_.default_player_dismissed && !advanced_);
    };
    // Barre de défilement fine, collée au bord extérieur des deux zones.
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8.0F * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 4.0F * scale);
    ImGui::BeginChild("##arbre", ImVec2(tree_width, -footer), ImGuiChildFlags_Borders);
    // La molette sur l'arbre passe d'une page à la suivante ou à la précédente.
    if (ImGui::IsWindowHovered() && io.MouseWheel != 0.0F) {
      const int step = io.MouseWheel < 0.0F ? 1 : -1;
      for (int next = page_ + step; next >= 0 && next < static_cast<int>(kPages.size()); next += step) {
        if (visible(static_cast<std::size_t>(next))) {
          page_ = next;
          break;
        }
      }
    }
    const char* category = "";
    bool category_open = false;
    for (std::size_t i = 0; i < kPages.size(); ++i) {
      const PageInfo& page = kPages[i];
      if (!visible(i)) continue;
      if (std::string_view(page.category) != category) {
        if (category_open) ImGui::TreePop();
        category = page.category;
        ImGui::SetNextItemOpen(true, ImGuiCond_Once);
        category_open = ImGui::TreeNodeEx(category, ImGuiTreeNodeFlags_SpanAvailWidth);
      }
      if (category_open && ImGui::Selectable(page.name, static_cast<int>(i) == page_))
        page_ = static_cast<int>(i);
    }
    if (category_open) ImGui::TreePop();
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##page", ImVec2(0, -footer));
    panel::begin_page();
    {
      const std::string_view name = kPages[static_cast<std::size_t>(page_)].name;
      if (name == "Configuration") {
        if (config_page(draft_, choices, bench_, actions)) actions.emplace_back(Command{cmd::Pause{}});
      } else if (name == "Lecteur par défaut") {
        default_player_page(choices.place, system_, draft_.default_player_dismissed,
                            choices.default_player_is_ajp);
      } else if (name == "Mises à jour") {
        updates_page(draft_, choices.place, updates_);
      } else if (name == "Agrandissement") {
        image_tab(draft_, choices);
        if (advanced_) composition_section(draft_);
      } else if (name == "Fluidité") {
        fluidity_tab(draft_, choices);
      } else if (name == "Conversion HDR") {
        hdr_tab(draft_, choices);
        if (advanced_) oled_sections(draft_, choices);
      } else if (name == "Réglages") {
        grade_tab(draft_, choices);
      } else if (name == "Sortie") {
        delay_held = sound_tab(draft_, choices);
      } else if (name == "Langues") {
        languages_tab(draft_);
      } else if (name == "Sous-titres") {
        subtitles_tab(draft_);
      } else if (name == "Fichier") {
        export_tab(*export_, running, choices, actions);
      } else if (name == "Moteurs installés") {
        engines_tab(choices, engine_draft_, actions, draft_.engines_dir);
      } else if (name == "Statistiques") {
        stats_tab(draft_);
      } else if (name == "Réinitialiser") {
        reset_page(actions, choices.profiles);
      }
    }
    panel::end_page();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::Checkbox("Réglages avancés", &advanced_);
    // Un réglage de la chaîne fait à la main : le préréglage devient Personnalisé (les valeurs restent).
    if (draft_.preset != Preset::Custom && draft_.preset == before.preset &&
        preset_chain_differs(before, draft_))
      draft_.preset = Preset::Custom;
    ImGui::PopItemWidth();

    // La part directe s'applique tout de suite ; le reste attend « Appliquer ».
    PlaySettings live = live_part(running, draft_, choices.rife_loaded);
    // Chaque délai recale le son (il reprend les blocs en file) : un glissement hacherait le son à
    // chaque image, le délai s'applique donc au relâchement (saisie au clavier comprise).
    if (delay_held) live.audio_delay_ms = running.audio_delay_ms;
    if (live != running) result.live = live;
    const Change pending = change_between(live, draft_, choices.rife_loaded);
    pending_shown_ = pending.cost == ChangeCost::Reopen;
    if (pending.cost == ChangeCost::Reopen) {
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
      ImGui::TextWrapped("À appliquer : %s. Fermer le panneau abandonne ces changements.",
                         pending.reason.c_str());
      ImGui::PopStyleColor();
      if (ImGui::Button("Appliquer (rouvre la vidéo)")) result.reopen = draft_;
      ImGui::SameLine();
      if (ImGui::Button("Abandonner ces changements")) draft_ = live;
    }
  }
  ImGui::End();
  if (!keep) open_ = false;
  return result;
}

}  // namespace ajp::app::play
