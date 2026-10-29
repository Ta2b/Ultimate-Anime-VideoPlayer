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
#include "app/play/settings_sound.h"
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
        "Écran SDR : le HDR est indisponible ; l'image reste en SDR et le grade (page Couleurs) imite le "
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
  note(
      "Appliqués en direct. Contraste et saturation sont dans la page Couleurs. Gris moyen : luminosité des "
      "tons moyens. Le HDR adaptatif suit la scène : moins d'expansion sur les scènes claires, ombres plus "
      "douces la nuit.");
  panel::end_group();  // le dernier encadré s'ouvre dans la zone grisée : il se ferme avant d'en sortir
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
                   limits::kTrueHdrContrast.hi, "%d %%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderInt("Saturation", &d.true_hdr_saturation, limits::kTrueHdrSaturation.lo,
                   limits::kTrueHdrSaturation.hi, "%d %%", ImGuiSliderFlags_AlwaysClamp);
  if (ImGui::Button("Contraste et saturation par défaut")) {
    const PlaySettings defaults;
    d.true_hdr_contrast = defaults.true_hdr_contrast;
    d.true_hdr_saturation = defaults.true_hdr_saturation;
  }
  note(
      "En direct, pour RTX HDR, le HDR adaptatif et la sortie SDR (autour du gris moyen de la page "
      "Conversion HDR, sans les éclats au-dessus du blanc).");
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
  panel::tip(
      "Éclatant : éclats et aplats lumineux, couleurs et noirs poussés, teinte gardée ; le garde-fou adoucit "
      "les couleurs saturées près du pic au lieu de les écrêter. Naturel : le grade d'origine du lecteur. "
      "Tout s'applique en direct.");
  render::GradeSettings& g = d.grade_values;
  using namespace render::grade_limits;
  bool moved = false;
  if (panel::advanced()) {
    moved |= grade_slider("Éclats", g.highlights_rtx, kHighlights);
    moved |= grade_slider("Aplats lumineux", g.area_rtx, kAreaRtx);
    moved |= grade_slider("Couleurs", g.vibrance_rtx, kVibrance);
    moved |= grade_slider("Couleurs étendues", g.expand, kExpand);
    moved |= grade_slider("Noirs profonds", g.shadows, kShadows);
    moved |= grade_slider("Garde-fou des couleurs", g.guard, kGuard);
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

// Noir absolu et soins OLED (HDR et couleurs > Conversion HDR, réglages avancés).
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
      "Anti-marquage : la barre s'efface aussi en pause après 20 s sans activité. Cadence fixe : l'image est "
      "représentée à chaque rafraîchissement, pour que le VRR garde une fréquence constante (une fréquence "
      "qui varie fait scintiller les scènes sombres d'un OLED) ; coûte une composition par "
      "rafraîchissement.");
  if (!choices.black_status.empty()) note(choices.black_status.c_str());
  note(
      "Les quasi-noirs deviennent un vrai noir (0) ; le pied rejoint l'image au seuil, compté en pas de code "
      "au-dessus du noir de la source. Calé en direct sur une rampe de 32 marches.");
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
    section(rank == 0 ? "Langues préférées" : "Sinon, si la première n'est pas disponible");
    language_combo("Langue de l'audio", "Selon le fichier", d.language_presets[rank].audio);
    language_combo("Langue des sous-titres", "Aucun", d.language_presets[rank].subtitles);
    panel::end_group();  // l'encadré de la section ouvre sa propre fenêtre fille : il se ferme avant le PopID
    ImGui::PopID();
  }
  note(
      "À l'ouverture d'un fichier, la première préférence dont l'audio et les sous-titres existent est prise "
      "telle quelle (« Aucun » : pas de sous-titres) ; sinon la deuxième ; sinon la meilleure langue de "
      "chaque liste. Vaut au prochain fichier ; les pistes en cours se changent dans la barre.");
}

void subtitles_tab(PlaySettings& d) {
  section("Taille des sous-titres");
  int percent = static_cast<int>(std::lround(d.subtitle_scale * 100.0F));
  if (ImGui::SliderInt("Taille", &percent, static_cast<int>(limits::kSubtitleScale.lo * 100.0F),
                       static_cast<int>(limits::kSubtitleScale.hi * 100.0F), "%d %%",
                       ImGuiSliderFlags_AlwaysClamp)) {
    d.subtitle_scale = static_cast<float>(percent) / 100.0F;
  }
  if (ImGui::Button("Taille d'origine")) d.subtitle_scale = 1.0F;
  panel::tip(
      "Agrandit ou réduit toutes les polices, panneaux compris (comme la taille de mpv). Changement en "
      "direct ; la piste se choisit dans la barre (j).");
}

// L'épisode suivant : lu tout seul à la fin du fichier ouvert.
void episodes_section(PlaySettings& d) {
  section("Épisodes");
  ImGui::Checkbox("Lire l'épisode suivant à la fin", &d.autoplay_next);
  panel::tip(
      "À la fin d'une vidéo ouverte par un fichier, lit la suivante du même dossier (ordre naturel des "
      "noms : ep2 avant ep10). N : épisode suivant, Maj + N : précédent, O : passer le générique "
      "(chapitre Opening ou Ending, sinon 85 s).");
}

void stats_tab(PlaySettings& d) {
  section("Statistiques à l'écran");
  ImGui::Checkbox("Toutes les statistiques (Tab)", &d.stats.all);
  ImGui::Spacing();
  ImGui::BeginDisabled(d.stats.all);
  ImGui::Checkbox("Chaîne (modèles, tailles, moteur)", &d.stats.chain);
  ImGui::Checkbox("Images (temps par image, sautées)", &d.stats.frames);
  ImGui::Checkbox("Interpolation (créées, tenues, coupes)", &d.stats.rife);
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
      "Les écrans de bienvenue (ceux sans objet sont sautés) : moteurs, image, HDR, audio, langues, mises à "
      "jour. Rien ne change "
      "avant « Terminer ».");
  return started;
}

// Général > Réinitialiser.
void reset_page(std::vector<Input>& actions, const ProfilesView& view) {
  section("Profil d'origine");
  note(
      "Remet le profil actif (Ctrl+1 à Ctrl+9) à ses valeurs d'origine ; les autres profils ne bougent pas.");
  if (ImGui::Button("Revenir au profil d'origine"))
    actions.emplace_back(EditProfile{.edit = core::edit::ResetProfile{.slot = view.active}});
}

}  // namespace

// L'arbre : l'ordre est celui de l'écran. Les catégories se déduisent du changement de `category`.
constexpr std::array<SettingsPage, 12> kPages{{
    {"Général", "Configuration", false},
    {"Général", "Lecture", false},      // langues préférées, sous-titres, épisode suivant
    {"Général", "Application", false},  // lecteur par défaut et mises à jour
    {"Général", "Statistiques", false},
    {"Général", "Réinitialiser", false},
    {"Image IA", "Agrandissement", false},
    {"Image IA", "Fluidité", false},
    {"HDR et couleurs", "Conversion HDR", false},
    {"HDR et couleurs", "Couleurs", false},
    {"Audio", "Sortie", false},
    {"Export et moteurs", "Export", false},
    {"Export et moteurs", "Moteurs", false},
}};

std::span<const SettingsPage> settings_pages() {
  return kPages;
}

void language_choice(const char* label, const char* none_label, std::string& tag) {
  language_combo(label, none_label, tag);
}

SettingsPanel::SettingsPanel() : export_(std::make_unique<ExportDraft>()) {}
SettingsPanel::~SettingsPanel() {
  if (prepare_.stop) prepare_.stop->request_stop();  // la tâche dépend du lecteur qui ouvre ce panneau
}

SettingsPanel::Result SettingsPanel::draw(const PlaySettings& running, const SettingsChoices& choices,
                                          float scale, std::vector<Input>& actions) {
  Result result;
  // Le panneau ouvert passe d'abord : la petite fenêtre masquerait ses réglages et en prendrait la molette.
  if (!open_ && !prompt_done_ && choices.offer_default_player && !running.default_player_dismissed &&
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
                        page.category);
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
                      ImGui::GetColorU32(selected ? ImGuiCol_Text : ImGuiCol_TextDisabled), page.name);
      }
      ImGui::PopStyleVar();
    }
    shown_page_ = page_;
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##page", ImVec2(0, -footer));
    panel::begin_page();
    {
      const std::string_view name = kPages[static_cast<std::size_t>(page_)].name;
      if (name == "Configuration") {
        if (config_page(draft_, choices, bench_, actions)) actions.emplace_back(Command{cmd::Pause{}});
        if (bench_.finished) {
          bench_.finished = false;
          prepare_after_bench(draft_, choices.span_prepare, choices.place.data, prepare_);
        }
      } else if (name == "Application") {
        default_player_page(choices.place, system_, draft_.default_player_dismissed,
                            choices.default_player_is_ajp);
        updates_page(draft_, choices.place, updates_);
      } else if (name == "Agrandissement") {
        image_tab(draft_, choices);
        if (advanced_) {
          span_prepare_section(draft_, choices.span_prepare, prepare_);
          composition_section(draft_);
        }
      } else if (name == "Fluidité") {
        fluidity_tab(draft_, choices);
      } else if (name == "Conversion HDR") {
        hdr_tab(draft_, choices);
        if (advanced_) oled_sections(draft_, choices);
      } else if (name == "Couleurs") {
        grade_tab(draft_, choices);
      } else if (name == "Sortie") {
        delay_held = sound_tab(draft_, choices);
      } else if (name == "Lecture") {
        languages_tab(draft_);
        subtitles_tab(draft_);
        episodes_section(draft_);
      } else if (name == "Export") {
        export_tab(*export_, running, choices, actions);
      } else if (name == "Moteurs") {
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
