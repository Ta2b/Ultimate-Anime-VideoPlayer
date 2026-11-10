#include "app/play/settings_panel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <imgui.h>

#include "app/play/auto_config.h"
#include "app/play/i18n.h"
#include "app/play/look_catalog.h"
#include "app/play/panel_style.h"
#include "app/play/settings_bench_results.h"
#include "app/play/settings_chain_tabs.h"
#include "app/play/settings_engines.h"
#include "app/play/settings_export.h"
#include "app/play/settings_sound.h"
#include "app/play/text_fit.h"
#include "app/play/tracks.h"
#include "core/engines.h"
#include "render/cube_lut.h"

namespace ajp::app::play {
namespace {

using panel::kWarn;
using panel::note;
using panel::section;

void hdr_tab(PlaySettings& d, const SettingsChoices& choices) {
  // Pas de choix : le HDR suit le système (host_facts.h, system_hdr). La page dit ce qui tourne ; le pic de
  // l'écran, en avancé, est grisé tant que l'écran ne reçoit pas de HDR.
  section(i18n::tr("Output"));
  note(choices.hdr_output
           ? i18n::tr("HDR: on. The player follows your system's HDR setting.")
           : i18n::tr("SDR: HDR is off in your system, or the display is too dim for it. The Color page's "
                      "grade still applies."));
  // Tri de l'ADR 0012 : la méthode HDR anime règle seule la luminosité ; reste le pic de l'écran, en avancé.
  if (panel::advanced()) {
    panel::end_group();  // l'encadré Sortie se ferme avant la zone grisée
    ImGui::BeginDisabled(!choices.hdr_output);
    section(i18n::tr("Display peak"));
    bool automatic = d.true_hdr_peak == 0;
    if (ImGui::Checkbox(i18n::tr_format("Auto peak ({} nits)", choices.auto_peak).c_str(), &automatic))
      d.true_hdr_peak =
          automatic ? 0 : std::clamp(choices.auto_peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi);
    ImGui::BeginDisabled(automatic);
    int peak = automatic ? choices.auto_peak : d.true_hdr_peak;
    if (ImGui::SliderInt(i18n::tr("Peak (nits)"), &peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi,
                         "%d", ImGuiSliderFlags_AlwaysClamp) &&
        !automatic)
      d.true_hdr_peak = peak;
    ImGui::EndDisabled();
#if defined(__linux__)
    // Sous KDE, le pic est en nits du contenu : KWin les affiche × luminosité SDR / 203 (system_peak.cpp).
    // Windows affiche le PQ d'une application tel quel : contenu = écran, rien à dire.
    note(
        i18n::tr("In content nits: KDE shows them brighter (× its SDR brightness / 203), so a 1000-nit "
                 "screen reads lower here."));
#endif
    panel::end_group();  // le dernier encadré s'ouvre dans la zone grisée : il se ferme avant d'en sortir
    ImGui::EndDisabled();
  }
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

void grade_tab(PlaySettings& d, const SettingsChoices& /*choices*/, std::vector<Input>& /*actions*/) {
  // Tri de l'ADR 0012 : la méthode HDR anime (guidée, par plan) fait le rendu. Restent les noirs profonds,
  // qui suivent l'épisode (le curseur est leur maximum, dans un épisode très clair : A38-16), et le look : la
  // LUT de base et « skin fix » fixes, le contraste et les éclats de -100 à 100 (0 : le réglage de
  // référence).
  section(i18n::tr("Color grading"));
  render::GradeSettings& g = d.grade_values;
  ImGui::SliderInt(i18n::tr("Saturation"), &g.anime_saturation, 0, 200, "%d %%",
                   ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderInt(i18n::tr("Contrast"), &g.anime_contrast, 0, 200, "%d %%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderInt(i18n::tr("Highlights"), &g.anime_highlights, 0, 200, "%d %%",
                   ImGuiSliderFlags_AlwaysClamp);
  if (panel::advanced()) {
    ImGui::SliderInt(i18n::tr("LUT contrast intensity"), &d.look_contrast, -100, 100, "%d",
                     ImGuiSliderFlags_AlwaysClamp);
    ImGui::SliderInt(i18n::tr("LUT highlight intensity"), &d.look_highlight, -100, 100, "%d",
                     ImGuiSliderFlags_AlwaysClamp);
    grade_slider(i18n::tr("Deep blacks"), g.shadows, render::grade_limits::kShadows);
    panel::tip(i18n::tr("The maximum, reached in a very bright episode; it fades out in a very dark one."));
  }
  if (ImGui::Button(i18n::tr("Defaults"))) {
    const PlaySettings defaults;
    g.anime_saturation = defaults.grade_values.anime_saturation;
    g.anime_contrast = defaults.grade_values.anime_contrast;
    g.anime_highlights = defaults.grade_values.anime_highlights;
    d.look_contrast = defaults.look_contrast;
    d.look_highlight = defaults.look_highlight;
    g.shadows = defaults.grade_values.shadows;
  }
}

// Vidéo > Affichage : la composition finale, le filtre vers la fenêtre et le débandage.
void composition_section(PlaySettings& d) {
  section(i18n::tr("Final scaling"));
  if (ImGui::BeginCombo(i18n::tr("Scaler"), std::string(label(d.final_scaler)).c_str())) {
    for (int i = 0; i < kFinalScalerCount; ++i) {
      const auto scaler = static_cast<FinalScaler>(i);
      if (ImGui::Selectable(std::string(label(scaler)).c_str(), scaler == d.final_scaler))
        d.final_scaler = scaler;
    }
    ImGui::EndCombo();
  }
  ImGui::Checkbox(i18n::tr("Debanding"), &d.deband);
  note(
      i18n::tr("The scaler stretches the output to the window; EWA Lanczos is sharpest but heaviest. No "
               "effect on export."));
}

// Noir absolu et soins OLED (Vidéo > Affichage, réglages avancés).
void oled_sections(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("OLED black"));
  if (ImGui::BeginCombo(i18n::tr("True black"), std::string(label(d.oled_black)).c_str())) {
    for (int i = 0; i < kOledBlackCount; ++i) {
      const auto mode = static_cast<OledBlack>(i);
      if (ImGui::Selectable(std::string(label(mode)).c_str(), mode == d.oled_black)) d.oled_black = mode;
    }
    ImGui::EndCombo();
  }
  ImGui::BeginDisabled(d.oled_black == OledBlack::Off);
  grade_slider(i18n::tr("Threshold (code steps)"), d.oled_black_steps, render::grade_limits::kBlackSteps);
  ImGui::EndDisabled();
  note((i18n::tr("Display: ") + choices.display_text).c_str());
  section(i18n::tr("OLED care"));
  ImGui::Checkbox(i18n::tr("Hide the bar when paused (burn-in)"), &d.oled_care);
  ImGui::Checkbox(i18n::tr("Steady frame rate (VRR flicker)"), &d.steady_cadence);
  note(i18n::tr("Steady frame rate keeps VRR at a constant refresh so dark scenes do not flicker on OLED."));
  if (!choices.black_status.empty()) note(choices.black_status.c_str());
  note(i18n::tr("Near-blacks become true black (0), up to the threshold. Live."));
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
    if (ImGui::Selectable(i18n::tr(language.name), same_language(tag, language.code)))
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
    section(rank == 0 ? i18n::tr("Preferred languages") : i18n::tr("Fallback"));
    language_combo(i18n::tr("Audio language"), i18n::tr("From the file"), d.language_presets[rank].audio);
    language_combo(i18n::tr("Subtitle language"), i18n::tr("None"), d.language_presets[rank].subtitles);
    panel::end_group();  // l'encadré de la section ouvre sa propre fenêtre fille : il se ferme avant le PopID
    ImGui::PopID();
  }
  note(i18n::tr(
      "The first preference whose tracks exist is used when a file opens. Applies to the next file."));
}

void subtitles_tab(PlaySettings& d) {
  section(i18n::tr("Subtitle size"));
  int percent = static_cast<int>(std::lround(d.subtitle_scale * 100.0F));
  if (ImGui::SliderInt(i18n::tr("Size"), &percent, static_cast<int>(limits::kSubtitleScale.lo * 100.0F),
                       static_cast<int>(limits::kSubtitleScale.hi * 100.0F), "%d %%",
                       ImGuiSliderFlags_AlwaysClamp)) {
    d.subtitle_scale = static_cast<float>(percent) / 100.0F;
  }
  if (ImGui::Button(i18n::tr("Reset size"))) d.subtitle_scale = 1.0F;
  panel::tip(i18n::tr("Scales all text, panels included. Live."));
  section(i18n::tr("Subtitle delay"));
  ImGui::SliderInt(i18n::tr("Delay (ms)"), &d.subtitle_delay_ms, -5000, 5000, "%d", ImGuiSliderFlags_None);
  d.subtitle_delay_ms =
      std::clamp(d.subtitle_delay_ms, limits::kSubtitleDelayMs.lo, limits::kSubtitleDelayMs.hi);
  if (ImGui::Button(i18n::tr("Reset delay"))) d.subtitle_delay_ms = 0;
  panel::tip(i18n::tr("Positive: subtitles later. Keys z and x: 100 ms. This session only."));
}

// L'épisode suivant : lu tout seul à la fin du fichier ouvert.
void episodes_section(PlaySettings& d) {
  section(i18n::tr("Episodes"));
  ImGui::Checkbox(i18n::tr("Autoplay next episode"), &d.autoplay_next);
  panel::tip(i18n::tr("Plays the next file in the folder. N: next, Shift+N: previous, O: skip credits."));
}

void stats_tab(PlaySettings& d) {
  section(i18n::tr("On-screen statistics"));
  ImGui::Checkbox(i18n::tr("All statistics (Tab)"), &d.stats.all);
  ImGui::Spacing();
  ImGui::BeginDisabled(d.stats.all);
  ImGui::Checkbox(i18n::tr("Pipeline"), &d.stats.chain);
  ImGui::Checkbox(i18n::tr("Frames"), &d.stats.frames);
  ImGui::Checkbox(i18n::tr("Interpolation"), &d.stats.rife);
  ImGui::Checkbox(i18n::tr("Tracks"), &d.stats.tracks);
  ImGui::Checkbox(i18n::tr("HDR output"), &d.stats.output);
  ImGui::Checkbox(i18n::tr("Exact position"), &d.stats.position);
  ImGui::Checkbox(i18n::tr("Video memory"), &d.stats.memory);
  ImGui::EndDisabled();
}

// Vidéo > Post-traitement : le niveau, d'Économe à Ultra, refait la chaîne (réduction, SPAN, interpolation)
// pour cette machine, cet écran et cette vidéo (auto_config.h : apply_preset). Auto prend le niveau de la
// machine ; un réglage fait à la main plus bas passe en Personnalisé, montré seulement quand il est en cours.
void level_section(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("Level"));
  const auto detected = static_cast<Tier>(std::clamp(choices.machine_tier, 0, kTierCount - 1));
  // Le préréglage qui porte le niveau de la machine : le nom qu'Auto affiche entre parenthèses.
  constexpr std::array<Preset, kTierCount> kByTier{Preset::Economy, Preset::Fast, Preset::Normal,
                                                   Preset::Quality, Preset::Ultra};
  const auto text = [&](Preset preset) {
    if (preset != Preset::Auto) return std::string(label(preset));
    if (choices.calibrated) return std::string(i18n::tr("Auto (calibrated)"));
    return i18n::tr_format("Auto ({})", label(kByTier[static_cast<std::size_t>(detected)]));
  };
  if (ImGui::BeginCombo(i18n::tr("Processing level"), text(d.preset).c_str())) {
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
                    .calibrated = choices.calibrated},
            core::SourceInfo{
                .width = choices.source_width, .height = choices.source_height, .fps = choices.source_fps});
      }
    }
    ImGui::EndCombo();
  }
  panel::tip(
      i18n::tr("From Economy (light on the graphics card) to Ultra. Auto picks the level of this machine. "
               "Changing a setting below switches to Custom."));
}

// Vidéo > Post-traitement : la netteté après SPAN (render/sharpen.h), en direct.
void sharpness_section(PlaySettings& d) {
  section(i18n::tr("Sharpness"));
  ImGui::SliderInt(i18n::tr("Sharpness"), &d.sharpness, limits::kSharpness.lo, limits::kSharpness.hi, "%d %%",
                   ImGuiSliderFlags_AlwaysClamp);
  panel::tip(
      i18n::tr("Crisper line art after upscaling, without halos. 0: untouched. Live, no effect on export."));
}

// Général > À propos : ce que le lecteur sait de la machine, l'assistant à refaire.
void about_page(const SettingsChoices& choices, std::vector<Input>& actions) {
  section(i18n::tr("This machine"));
  ImGui::TextWrapped("%s",
                     choices.gpu_name.empty() ? i18n::tr("Unknown graphics card") : choices.gpu_name.c_str());
  if (choices.display_hz > 0.0) note(i18n::tr_format("Display {:.0f} Hz", choices.display_hz));

  section(i18n::tr("Setup wizard"));
  if (ImGui::Button(i18n::tr("Run the wizard again"))) actions.emplace_back(OpenEngineScreen{.wizard = true});
  panel::tip(i18n::tr("Nothing changes until \"Finish\"."));
  report_problem_section(choices.place, choices.gpu_name);
}

// Général > À propos, en bas.
// Vrai quand « Tout réinitialiser » est pressé : le panneau rouvre alors la vidéo avec les réglages par
// défaut, sans attendre « Appliquer ».
bool reset_page(PlaySettings& draft) {
  bool reset_all = false;
  section(i18n::tr("All settings"));
  note(
      i18n::tr("Puts every setting back to its default value: chain, HDR, grade and colors, sound, playback. "
               "The engines folder is kept."));
  if (ImGui::Button(i18n::tr("Reset all settings"))) {
    PlaySettings fresh;
    fresh.engines_dir = draft.engines_dir;
    draft = fresh;
    reset_all = true;
  }
  return reset_all;
}

}  // namespace

// L'arbre : l'ordre est celui de l'écran. Les catégories se déduisent du changement de `category`. Trois
// catégories (ADR 0012) : Vidéo, Audio et lecture, Général.
constexpr std::array<SettingsPage, 10> kPages{{
    {Page::PostProcessing, "Video", "Post-processing", false},  // niveau, agrandissement, interpolation
    {Page::Colors, "Video", "Colors", false},    // sortie HDR ou SDR (elle suit le système), grade anime
    {Page::Display, "Video", "Display", false},  // filtre final, débandage ; OLED en avancé
    {Page::Sound, "Audio and playback", "Sound", false},
    {Page::Playback, "Audio and playback", "Playback",
     false},                                               // langues préférées, sous-titres, épisode suivant
    {Page::Application, "General", "Application", false},  // langue, mises à jour, lecteur par défaut
    {Page::Engines, "General", "Engines", false},          // le moteur et l'état de ses modules
    {Page::Export, "General", "Export", false},
    {Page::Statistics, "General", "Statistics", false},
    {Page::About, "General", "About", false},  // la machine, l'assistant, tout réinitialiser
}};

std::span<const SettingsPage> settings_pages() {
  return kPages;
}

void language_choice(const char* label, const char* none_label, std::string& tag) {
  language_combo(label, none_label, tag);
}

SettingsPanel::SettingsPanel() : export_(std::make_unique<ExportDraft>()) {}
bool SettingsPanel::busy() const {
  return open_ && (bench_running(bench_) || prepare_.pending != nullptr || (machine_ && machine_->running()));
}

SettingsPanel::~SettingsPanel() {
  if (prepare_.stop) prepare_.stop->request_stop();  // la tâche dépend du lecteur qui ouvre ce panneau
  bench_.stop();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
SettingsPanel::Result SettingsPanel::draw(const PlaySettings& running, const SettingsChoices& choices,
                                          float scale, std::vector<Input>& actions) {
  Result result;
  if (!open_) return result;
  // Réglages changés ailleurs (Tab, volume) : le panneau les reprend, sans perdre ce qui attend.
  if (running != seen_) {
    draft_ = follow_running(draft_, seen_, running, choices.rife_loaded);
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
  if (ImGui::Begin((std::string(i18n::tr("Settings")) + "###settings").c_str(), &keep, kFlags)) {
    ImGui::PushItemWidth(-width * 0.42F);
    panel::advanced() = advanced_;
    bool delay_held = false;
    const PlaySettings before = draft_;
    // Sous les deux colonnes : la case « Réglages avancés » (et le bandeau « Appliquer » quand il est là).
    const float footer = (pending_shown_ ? 84.0F * scale : 0.0F) + ImGui::GetFrameHeightWithSpacing() +
                         2.0F * ImGui::GetStyle().ItemSpacing.y;
    const float tree_width = std::min(200.0F * scale, width * 0.3F);
    if (!advanced_ && kPages[static_cast<std::size_t>(page_)].advanced) page_ = 0;
    // Une page masquée : un réglage avancé sans la case « Réglages avancés ».
    const auto visible = [&](std::size_t i) { return advanced_ || !kPages[i].advanced; };
    // Barre de défilement fine, collée au bord extérieur des deux zones.
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, 8.0F * scale);
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, 4.0F * scale);
    // Arbre collé à gauche : peu de marge intérieure.
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4.0F * scale, 4.0F * scale));
    ImGui::BeginChild("##arbre", ImVec2(tree_width, -footer), ImGuiChildFlags_Borders);
    ImGui::PopStyleVar();
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
    // L'arbre, dessiné à la main : une bande par catégorie (chevron, titre en couleur d'accent), les pages en
    // retrait dessous, la page choisie marquée d'une barre d'accent et d'un fond.
    {
      ImDrawList* draw = ImGui::GetWindowDrawList();
      const ImGuiStyle& style = ImGui::GetStyle();
      const float row_height = ImGui::GetFontSize() + 12.0F * scale;
      const float head_height = ImGui::GetFontSize() + 14.0F * scale;
      const float indent = 26.0F * scale;
      const ImU32 accent = ImGui::ColorConvertFloat4ToU32(panel::kAccent);
      ImVec4 soft = panel::kAccent;
      soft.w = 0.20F;
      ImVec4 band = style.Colors[ImGuiCol_Header];
      band.w = 0.14F;
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {style.ItemSpacing.x, 2.0F * scale});
      std::string_view category;
      const std::string_view page_category = kPages[static_cast<std::size_t>(page_)].category;
      std::string_view shown_category;
      if (shown_page_ >= 0) shown_category = kPages[static_cast<std::size_t>(shown_page_)].category;
      int category_index = -1;
      bool category_open = false;
      for (std::size_t i = 0; i < kPages.size(); ++i) {
        const SettingsPage& page = kPages[i];
        if (!visible(i)) continue;
        if (std::string_view(page.category) != category) {
          category = page.category;
          ++category_index;
          const auto slot = static_cast<std::size_t>(category_index);
          // Quand la page change de catégorie (clic, molette), la catégorie de la page s'ouvre et les autres
          // se replient : l'arbre tient dans une fenêtre basse. Ensuite chaque catégorie garde l'état choisi
          // à la main.
          if (page_ != shown_page_ && shown_category != page_category)
            category_open_[slot] = category == page_category;
          if (category_index > 0) ImGui::Dummy({0.0F, 6.0F * scale});
          const ImVec2 at = ImGui::GetCursorScreenPos();
          const float row_width = ImGui::GetContentRegionAvail().x;
          ImGui::PushID(category_index);
          if (ImGui::InvisibleButton("##categorie", {row_width, head_height}))
            category_open_[slot] = !category_open_[slot];
          ImGui::PopID();
          const bool hovered = ImGui::IsItemHovered();
          ImVec4 fill = band;
          if (hovered) fill.w = 0.30F;
          draw->AddRectFilled(at, {at.x + row_width, at.y + head_height},
                              ImGui::ColorConvertFloat4ToU32(fill), 5.0F * scale);
          // Le chevron : un triangle plein, à droite quand c'est replié, vers le bas quand c'est ouvert.
          const float cx = at.x + 12.0F * scale;
          const float cy = at.y + head_height * 0.5F;
          const float r = 4.0F * scale;
          if (category_open_[slot])
            draw->AddTriangleFilled({cx - r, cy - r * 0.6F}, {cx + r, cy - r * 0.6F}, {cx, cy + r * 0.9F},
                                    accent);
          else
            draw->AddTriangleFilled({cx - r * 0.6F, cy - r}, {cx - r * 0.6F, cy + r}, {cx + r * 0.9F, cy},
                                    accent);
          draw->AddText({at.x + 26.0F * scale, at.y + (head_height - ImGui::GetFontSize()) * 0.5F}, accent,
                        i18n::tr(page.category));
          category_open = category_open_[slot];
        }
        if (!category_open) continue;
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float row_width = ImGui::GetContentRegionAvail().x;
        ImGui::PushID(static_cast<int>(i));
        const bool picked = ImGui::InvisibleButton("##page", {row_width, row_height});
        ImGui::PopID();
        if (picked) page_ = static_cast<int>(i);
        const bool selected = static_cast<int>(i) == page_;
        if (selected) {
          draw->AddRectFilled({at.x + indent - 8.0F * scale, at.y}, {at.x + row_width, at.y + row_height},
                              ImGui::ColorConvertFloat4ToU32(soft), 5.0F * scale);
          draw->AddRectFilled({at.x + indent - 8.0F * scale, at.y + 4.0F * scale},
                              {at.x + indent - 5.0F * scale, at.y + row_height - 4.0F * scale}, accent,
                              2.0F * scale);
        } else if (ImGui::IsItemHovered()) {
          draw->AddRectFilled({at.x + indent - 8.0F * scale, at.y}, {at.x + row_width, at.y + row_height},
                              ImGui::ColorConvertFloat4ToU32(band), 5.0F * scale);
        }
        draw->AddText({at.x + indent, at.y + (row_height - ImGui::GetFontSize()) * 0.5F},
                      ImGui::GetColorU32(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled),
                      i18n::tr(page.name));
      }
      ImGui::PopStyleVar();
    }
    shown_page_ = page_;
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##page", ImVec2(0, -footer));
    panel::begin_page();
    {
      switch (kPages[static_cast<std::size_t>(page_)].id) {
        case Page::PostProcessing: {
          level_section(draft_, choices);
          image_tab(draft_, choices);
          sharpness_section(draft_);
          fluidity_tab(draft_, choices);
          if (advanced_) {
            span_prepare_section(choices.span_prepare, choices.span_job ? *choices.span_job : prepare_);
            // Une ligne du bench appliquée l'est tout de suite, sans « Appliquer » : c'est ce qu'on a
            // demandé.
            if (!machine_) machine_ = std::make_unique<MachineNote>();
            if (bench_results_section(draft_, choices, *machine_, actions) && draft_ != running)
              result.reopen = draft_;
          }
          break;
        }
        case Page::Colors: {
          hdr_tab(draft_, choices);
          grade_tab(draft_, choices, actions);
          break;
        }
        case Page::Display: {
          composition_section(draft_);
          if (advanced_) oled_sections(draft_, choices);
          break;
        }
        case Page::Sound: {
          delay_held = sound_tab(draft_, choices);
          break;
        }
        case Page::Playback: {
          languages_tab(draft_);
          subtitles_tab(draft_);
          episodes_section(draft_);
          break;
        }
        case Page::Application: {
          language_section(draft_);
          gpu_section(draft_, choices.gpus);
          updates_page(draft_, choices.place, updates_);
          default_player_page(choices.place, system_, choices.default_player_is_ajp);
          break;
        }
        case Page::Engines: {
          engines_tab(choices, engine_draft_, actions, draft_.engines_dir, engines_folder_);
          break;
        }
        case Page::Export: {
          export_tab(*export_, running, choices, actions);
          break;
        }
        case Page::Statistics: {
          stats_tab(draft_);
          break;
        }
        case Page::About: {
          about_page(choices, actions);
          if (reset_page(draft_)) result.reopen = draft_;
          break;
        }
      }
    }
    panel::end_page();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::Checkbox(i18n::tr("Advanced settings"), &advanced_);
    // Annuler et rétablir : un état n'est retenu que lorsque plus aucun curseur n'est tenu (un glissement
    // compte pour un seul changement).
    if (!ImGui::IsAnyItemActive() && draft_ != stable_) {
      undo_.push_back(stable_);
      if (undo_.size() > 100) undo_.erase(undo_.begin());
      redo_.clear();
      stable_ = draft_;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(undo_.empty());
    if (ImGui::Button(i18n::tr("Undo"))) {
      redo_.push_back(draft_);
      draft_ = undo_.back();
      undo_.pop_back();
      stable_ = draft_;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(redo_.empty());
    if (ImGui::Button(i18n::tr("Redo"))) {
      undo_.push_back(draft_);
      draft_ = redo_.back();
      redo_.pop_back();
      stable_ = draft_;
    }
    ImGui::EndDisabled();
    // Un réglage de la chaîne fait à la main : le préréglage devient Personnalisé (les valeurs restent).
    if (draft_.preset != Preset::Custom && draft_.preset == before.preset &&
        preset_chain_differs(before, draft_, choices.calibrated.has_value()))
      draft_.preset = Preset::Custom;
    ImGui::PopItemWidth();

    // La part directe s'applique tout de suite ; le reste attend « Appliquer ».
    PlaySettings live = live_part(running, draft_, choices.rife_loaded, choices.chain());
    // Chaque délai recale le son (il reprend les blocs en file) : un glissement hacherait le son à
    // chaque image, le délai s'applique donc au relâchement (saisie au clavier comprise).
    if (delay_held) live.audio_delay_ms = running.audio_delay_ms;
    if (live != running) result.live = live;
    const Change pending = change_between(live, draft_, choices.rife_loaded, choices.chain());
    pending_shown_ = pending.cost == ChangeCost::Reopen;
    if (pending.cost == ChangeCost::Reopen) {
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
      ImGui::TextWrapped(
          "%s",
          i18n::tr_format("To apply: {}. Closing the panel discards these changes.", pending.reason).c_str());
      ImGui::PopStyleColor();
      if (ImGui::Button(i18n::tr("Apply (reopens the video)"))) result.reopen = draft_;
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Discard changes"))) draft_ = live;
    }
  }
  ImGui::End();
  if (!keep) open_ = false;
  return result;
}

}  // namespace ajp::app::play
