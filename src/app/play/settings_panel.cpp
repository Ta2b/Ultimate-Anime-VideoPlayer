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

#include "app/play/panel_style.h"
#include "app/play/settings_chain_tabs.h"
#include "app/play/settings_engines.h"
#include "app/play/settings_export.h"
#include "app/play/text_fit.h"
#include "app/play/tracks.h"

namespace ajp::app::play {
namespace {

using panel::kNote;
using panel::kWarn;
using panel::note;
using panel::section;

// Le moteur du mode HDR : RTX HDR (TrueHDR, NVIDIA), HDR adaptatif (tous GPU : libplacebo, puis
// l'aspect TrueHDR émulé qui suit la scène), ou SDR.
constexpr std::array<const char*, 3> kRoutes{"RTX HDR (NVIDIA)", "HDR adaptatif (tous GPU)", "SDR"};

void hdr_tab(PlaySettings& d, const SettingsChoices& choices) {
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
  if (!choices.hdr_output) note("L'écran ne reçoit pas de HDR10 : la sortie reste SDR.");
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
  ImGui::SliderInt("Gris moyen", &d.true_hdr_gray, limits::kTrueHdrGray.lo, limits::kTrueHdrGray.hi, "%d",
                   ImGuiSliderFlags_AlwaysClamp);
  if (choices.nvidia) {
    ImGui::BeginDisabled(!d.true_hdr);
    ImGui::Checkbox("Débandage (RTX HDR)", &d.true_hdr_debanding);
    ImGui::EndDisabled();
  }
  if (ImGui::Button("Valeurs par défaut")) {
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
  moved |= grade_slider("Éclats", g.highlights_rtx, kHighlights);
  moved |= grade_slider("Aplats lumineux", g.area_rtx, kAreaRtx);
  moved |= grade_slider("Couleurs", g.vibrance_rtx, kVibrance);
  moved |= grade_slider("Couleurs étendues (large gamut)", g.expand, kExpand);
  moved |= grade_slider("Noirs profonds", g.shadows, kShadows);
  moved |= grade_slider("Garde-fou couleurs saturées", g.guard, kGuard);
  if (moved) d.grade_preset = render::GradePreset::Custom;
  ImGui::Checkbox("Gamut Display P3 de l'écran", &g.gamut_p3);
  ImGui::EndDisabled();
  if (!choices.hdr_output) {
    note(
        "Sortie SDR : couleurs, noirs, garde-fou, contraste et saturation du mode HDR s'appliquent ; "
        "éclats, aplats lumineux et couleurs étendues demandent une sortie HDR10.");
  } else if (d.true_hdr) {
    note("Route RTX HDR : TrueHDR, puis le grade sur la sortie PQ.");
  } else if (d.hdr_native) {
    note("HDR adaptatif : libplacebo en PQ, aspect TrueHDR émulé selon la scène, puis le grade.");
  } else {
    note("Sortie HDR10 sans moteur HDR : pas de grade (moteur à choisir dans l'onglet HDR).");
  }
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
bool sound_tab(PlaySettings& d) {
  section("Synchronisation");
  ImGui::SliderInt("Délai audio (ms)", &d.audio_delay_ms, -1000, 1000);
  const bool held = ImGui::IsItemActive();
  if (ImGui::Button("Remettre à 0")) d.audio_delay_ms = 0;
  note(
      "Positif : le son est retardé. Ctrl + clic sur le curseur pour saisir une valeur (jusqu'à ±10 000 "
      "ms).");
  d.audio_delay_ms = std::clamp(d.audio_delay_ms, limits::kAudioDelayMs.lo, limits::kAudioDelayMs.hi);
  ImGui::Checkbox("Lire l'épisode suivant à la fin", &d.autoplay_next);
  note(
      "À la fin d'une vidéo ouverte par un fichier, lit la suivante du même dossier (ordre naturel des "
      "noms : ep2 avant ep10). N : épisode suivant, Maj + N : précédent, O : passer le générique "
      "(chapitre Opening ou Ending, sinon 85 s).");
  ImGui::Checkbox("Caler la lecture sur l'écran", &d.display_sync);
  note(
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

void profiles_tab(const ProfilesView& view, std::array<char, 64>& rename, std::vector<Input>& actions) {
  section("Profils (Ctrl+1 à Ctrl+9)");
  for (int slot = 1; slot <= 9; ++slot) {
    const std::optional<ProfileRow>& row = view.rows[static_cast<std::size_t>(slot - 1)];
    if (!row) continue;
    ImGui::PushID(slot);
    const bool active = slot == view.active;
    if (ImGui::Selectable(std::format("{}  {}{}", slot, row->name, active ? "  ·  actif" : "").c_str(),
                          active))
      actions.emplace_back(Command{cmd::SelectProfile{.slot = slot}});
    ImGui::PushStyleColor(ImGuiCol_Text, kNote);
    ImGui::TextUnformatted(("     " + row->summary).c_str());
    ImGui::PopStyleColor();
    ImGui::PopID();
  }
  note("Changer d'avis dans les autres onglets modifie le profil actif ; il est gardé dans user.toml.");

  section("Profil actif");
  ImGui::InputText("Nom", rename.data(), rename.size());
  ImGui::SameLine();
  if (ImGui::Button("Renommer") && rename[0] != '\0')
    actions.emplace_back(
        EditProfile{.edit = core::edit::ProfileName{.slot = view.active, .name = rename.data()}});
  bool start = view.start == view.active;
  if (ImGui::Checkbox("Profil de démarrage", &start) && start)
    actions.emplace_back(EditProfile{.edit = core::edit::StartProfile{.slot = view.active}});
  if (ImGui::Button("Revenir au profil d'origine"))
    actions.emplace_back(EditProfile{.edit = core::edit::ResetProfile{.slot = view.active}});
}

}  // namespace

SettingsPanel::SettingsPanel() : export_(std::make_unique<ExportDraft>()) {}
SettingsPanel::~SettingsPanel() = default;

SettingsPanel::Result SettingsPanel::draw(const PlaySettings& running, const SettingsChoices& choices,
                                          float scale, std::vector<Input>& actions) {
  Result result;
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
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
  if (ImGui::Begin("Paramètres", &keep, kFlags)) {
    ImGui::PushItemWidth(-width * 0.42F);
    bool delay_held = false;
    if (ImGui::BeginTabBar("##sections")) {
      if (ImGui::BeginTabItem("Profils")) {
        // Le champ reçoit le nom du profil actif quand le profil ou son nom change (autre profil,
        // retour au profil d'origine), pas quand il est vide : on peut l'effacer puis taper.
        const std::optional<ProfileRow>& current =
            choices.profiles.rows[static_cast<std::size_t>(choices.profiles.active - 1)];
        const std::string name = current ? current->name : std::string();
        if (choices.profiles.active != shown_profile_ || name != shown_name_) {
          copy_text(rename_, name);
          shown_profile_ = choices.profiles.active;
          shown_name_ = name;
        }
        profiles_tab(choices.profiles, rename_, actions);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Image")) {
        image_tab(draft_, choices);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Fluidité")) {
        fluidity_tab(draft_, choices);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("HDR")) {
        hdr_tab(draft_, choices);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Grade")) {
        grade_tab(draft_, choices);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Langues")) {
        languages_tab(draft_);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Sous-titres")) {
        subtitles_tab(draft_);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Son")) {
        delay_held = sound_tab(draft_);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Stats")) {
        stats_tab(draft_);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Moteurs")) {
        engines_tab(choices, engine_draft_, actions);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Export")) {
        export_tab(*export_, running, choices, actions);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Système")) {
        system_tab(choices.place, system_);
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
    ImGui::PopItemWidth();

    // La part directe s'applique tout de suite ; le reste attend « Appliquer ».
    PlaySettings live = live_part(running, draft_, choices.rife_loaded);
    // Chaque délai recale le son (il reprend les blocs en file) : un glissement hacherait le son à
    // chaque image, le délai s'applique donc au relâchement (saisie au clavier comprise).
    if (delay_held) live.audio_delay_ms = running.audio_delay_ms;
    if (live != running) result.live = live;
    const Change pending = change_between(live, draft_, choices.rife_loaded);
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
