#include "app/play/settings_panel.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <format>
#include <string_view>

#include <imgui.h>

namespace ajp::app::play {
namespace {

constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};
constexpr ImVec4 kNote{0.66F, 0.68F, 0.74F, 1.0F};
constexpr ImVec4 kWarn{1.0F, 0.78F, 0.42F, 1.0F};

struct HeightChoice {
  int lines;
  const char* label;
};
constexpr std::array<HeightChoice, 5> kHeights{{
    {0, "Source (sans réduction : 4K)"},
    {900, "900 lignes"},
    {720, "720 lignes (recommandé)"},
    {540, "540 lignes"},
    {480, "480 lignes"},
}};

void note(const char* text) {
  ImGui::PushStyleColor(ImGuiCol_Text, kNote);
  ImGui::TextWrapped("%s", text);
  ImGui::PopStyleColor();
}

void section(const char* title) {
  ImGui::Spacing();
  ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
  ImGui::TextUnformatted(title);
  ImGui::PopStyleColor();
  ImGui::Separator();
}

void image_tab(PlaySettings& d, const SettingsChoices& choices) {
  section("Agrandissement");
  ImGui::Checkbox("SPAN ×2 (AnimeJaNai)", &d.upscale);
  ImGui::BeginDisabled(!d.upscale);
  const std::string current = d.model.empty() ? choices.default_model : d.model;
  if (ImGui::BeginCombo("Modèle", model_label(current).c_str())) {
    for (const std::string& model : choices.models) {
      const bool chosen = model == current;
      if (ImGui::Selectable(model_label(model).c_str(), chosen))
        d.model = model == choices.default_model ? std::string() : model;
      if (chosen) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  note("Performance : le plus rapide. Balanced : plus fin, plus lent. Sharp1 : contours plus nets.");

  section("Réduction avant l'agrandissement");
  const auto height = std::ranges::find(kHeights, d.work_height, &HeightChoice::lines);
  const std::string height_label =
      height != kHeights.end() ? std::string(height->label) : std::format("{} lignes", d.work_height);
  if (ImGui::BeginCombo("Hauteur de travail", height_label.c_str())) {
    for (const HeightChoice& choice : kHeights) {
      if (ImGui::Selectable(choice.label, choice.lines == d.work_height)) d.work_height = choice.lines;
    }
    ImGui::EndCombo();
  }
  const bool reduces = d.work_height > 0 && d.work_height < choices.source_height;
  if (reduces) {
    if (ImGui::BeginCombo("Algorithme", std::string(label(d.downscale)).c_str())) {
      for (int i = 0; i < kDownscaleCount; ++i) {
        const auto filter = static_cast<Downscale>(i);
        if (ImGui::Selectable(std::string(label(filter)).c_str(), filter == d.downscale))
          d.downscale = filter;
      }
      ImGui::BeginDisabled();
      ImGui::Selectable("SSimDownscaler (à venir)", false);
      ImGui::EndDisabled();
      ImGui::EndCombo();
    }
    note(
        "Area : moyenne exacte. Spline36 : celui d'AnimeJaNai, net. Catmull-Rom et Mitchell gardent du "
        "détail ; Bilinear et Bicubic adoucissent. Changement en direct.");
  } else {
    note("Pas de réduction : la source entre telle quelle dans SPAN.");
  }
}

void fluidity_tab(PlaySettings& d, const SettingsChoices& choices) {
  const bool four_k =
      (d.work_height == 0 || d.work_height >= choices.source_height) && choices.source_height >= 1080;
  section("Interpolation RIFE 4.26");
  ImGui::BeginDisabled(!choices.rife_available);
  constexpr std::array<std::pair<int, const char*>, 4> kFactors{
      {{0, "Désactivée"}, {2, "×2"}, {3, "×3 (recommandé)"}, {4, "×4"}}};
  for (const auto& [factor, text] : kFactors) {
    if (ImGui::RadioButton(text, d.rife_factor == factor || (factor == 0 && d.rife_factor == 1)))
      d.rife_factor = factor;
  }
  ImGui::EndDisabled();
  if (!choices.rife_available) note("Modèle RIFE introuvable (rife_v4.26.onnx).");
  if (!choices.rife_models.empty()) {
    ImGui::BeginDisabled(d.motion_engine != MotionEngine::Rife);
    const std::string current = d.rife_model.empty() ? choices.default_rife_model : d.rife_model;
    if (ImGui::BeginCombo("Modèle RIFE", rife_label(current).c_str())) {
      for (const std::string& model : choices.rife_models) {
        const bool chosen = model == current;
        if (ImGui::Selectable(rife_label(model).c_str(), chosen))
          d.rife_model = model == choices.default_rife_model ? std::string() : model;
        if (chosen) ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    if (current.find("heavy") != std::string::npos) {
      note(
          "RIFE heavy : plus fin sur les mouvements difficiles, mais environ 4 fois plus lent que RIFE 4.26 "
          "(34 ms par passe à 720p sur une 4070 Ti). Même en ×2, la lecture prend du retard dans les scènes "
          "qui bougent à chaque image : la chaîne laisse alors ces paires sans interpolation.");
    }
  }
  section("Moteur du mouvement");
  for (int i = 0; i < kMotionEngineCount; ++i) {
    const auto engine = static_cast<MotionEngine>(i);
    if (ImGui::RadioButton(std::string(label(engine)).c_str(), d.motion_engine == engine))
      d.motion_engine = engine;
  }
  if (d.motion_engine == MotionEngine::OpticalFlow) {
    note(
        "Expérimental. L'accélérateur de flux optique des RTX (celui de Smooth Motion et de DLSS Frame "
        "Generation) estime le mouvement à la place de RIFE, bien moins cher que ses 7 ms par paire à 720p, "
        "pour un mouvement moins fin. Utile en 4K ×3 ou ×4. Sans lui (autre carte, pilote trop ancien), "
        "RIFE reprend.");
  }
  if (four_k && d.rife_factor >= 3) {
    note(
        "En 4K sans réduction, ×2 tient sur cette carte ; ×3 et ×4 prennent du retard dans les scènes qui "
        "bougent (SPAN 1080p → 2160p coûte 11 ms par image).");
  } else if (four_k) {
    note("En 4K, RIFE estime le mouvement en 720p et déforme les images 1080p : ×2 tient sur cette carte.");
  }
  note("Les coupes de scène et les dessins tenus ne sont pas interpolés.");
  section("Smooth Motion (pilote NVIDIA)");
  ImGui::Checkbox("Doubler les images après RIFE", &d.smooth_motion);
  note(
      "Expérimental. La couche du pilote NVIDIA (RTX 40 et 50) génère une image entre deux images "
      "présentées : RIFE ×3 puis Smooth Motion donnent environ 144 images par seconde. Vaut au prochain "
      "lancement. Sans RIFE, la lecture peut caler : gardez RIFE à ×2 ou plus.");
  if (d.smooth_motion && d.rife_factor < 2)
    note("RIFE est désactivé : Smooth Motion risque de caler la lecture.");
}

// La route HDR : RTX HDR (TrueHDR), HDR natif (libplacebo, aspect TrueHDR émulé), ou SDR.
constexpr std::array<const char*, 3> kRoutes{"RTX HDR (TrueHDR)", "HDR natif (libplacebo)", "SDR"};

void hdr_tab(PlaySettings& d, const SettingsChoices& choices) {
  section("Sortie");
  const int route = d.true_hdr ? 0 : (d.hdr_native ? 1 : 2);
  if (ImGui::BeginCombo("Route", kRoutes[static_cast<std::size_t>(route)])) {
    for (int i = 0; i < static_cast<int>(kRoutes.size()); ++i) {
      ImGui::BeginDisabled(i == 0 && !choices.true_hdr_available);
      if (ImGui::Selectable(kRoutes[static_cast<std::size_t>(i)], i == route)) {
        d.true_hdr = i == 0;
        d.hdr_native = i == 1;
      }
      ImGui::EndDisabled();
    }
    ImGui::EndCombo();
  }
  if (!choices.true_hdr_available) note("SDK Maxine VideoFX introuvable (/usr/local/VideoFX).");
  if (!choices.hdr_output) note("L'écran ne reçoit pas de HDR10 : la sortie reste SDR.");
  section("RTX HDR (Maxine TrueHDR) et son émulation");
  bool automatic = d.true_hdr_peak == 0;
  if (ImGui::Checkbox(std::format("Pic automatique ({} nits)", choices.auto_peak).c_str(), &automatic))
    d.true_hdr_peak =
        automatic ? 0 : std::clamp(choices.auto_peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi);
  ImGui::BeginDisabled(automatic);
  int peak = automatic ? choices.auto_peak : d.true_hdr_peak;
  if (ImGui::SliderInt("Pic (nits)", &peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi) && !automatic)
    d.true_hdr_peak = peak;
  ImGui::EndDisabled();
  ImGui::SliderInt("Contraste", &d.true_hdr_contrast, limits::kTrueHdrContrast.lo,
                   limits::kTrueHdrContrast.hi);
  ImGui::SliderInt("Saturation", &d.true_hdr_saturation, limits::kTrueHdrSaturation.lo,
                   limits::kTrueHdrSaturation.hi);
  ImGui::SliderInt("Gris moyen", &d.true_hdr_gray, limits::kTrueHdrGray.lo, limits::kTrueHdrGray.hi);
  ImGui::BeginDisabled(!d.true_hdr);
  ImGui::Checkbox("Débandage", &d.true_hdr_debanding);
  ImGui::EndDisabled();
  if (ImGui::Button("Valeurs par défaut")) {
    const PlaySettings defaults;
    d.true_hdr_peak = defaults.true_hdr_peak;
    d.true_hdr_contrast = defaults.true_hdr_contrast;
    d.true_hdr_saturation = defaults.true_hdr_saturation;
    d.true_hdr_gray = defaults.true_hdr_gray;
    d.true_hdr_debanding = defaults.true_hdr_debanding;
  }
  note(
      "Réglages appliqués en direct, sur RTX HDR comme sur son émulation (HDR natif). Gris moyen : "
      "luminosité des tons moyens ; le débandage lisse les dégradés (le couper allège un peu RTX HDR).");
}

// Un curseur sur un réglage du grade (double) ; true s'il a bougé.
bool grade_slider(const char* label, double& value, core::Range<double> range) {
  auto shown = static_cast<float>(value);
  if (!ImGui::SliderFloat(label, &shown, static_cast<float>(range.lo), static_cast<float>(range.hi), "%.2f"))
    return false;
  value = static_cast<double>(shown);
  return true;
}

void grade_tab(PlaySettings& d, const SettingsChoices& choices) {
  section("Grade HDR");
  ImGui::Checkbox("Grader la sortie HDR", &d.grade);
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
    note("L'écran ne reçoit pas de HDR10 : pas de grade.");
  } else if (d.true_hdr) {
    note("Route RTX HDR : TrueHDR, puis le grade sur la sortie PQ.");
  } else if (d.hdr_native) {
    note("Route native : libplacebo en PQ, aspect TrueHDR émulé, puis le grade.");
  } else {
    note("Sortie SDR : pas de grade (route à choisir dans l'onglet RTX HDR).");
  }
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

void sound_tab(PlaySettings& d) {
  section("Synchronisation");
  ImGui::SliderInt("Délai audio (ms)", &d.audio_delay_ms, -1000, 1000);
  if (ImGui::Button("Remettre à 0")) d.audio_delay_ms = 0;
  note("Positif : le son est retardé. Ctrl + clic sur le curseur pour saisir une valeur (±5000 ms).");
  d.audio_delay_ms = std::clamp(d.audio_delay_ms, limits::kAudioDelayMs.lo, limits::kAudioDelayMs.hi);
}

void subtitles_tab(PlaySettings& d) {
  section("Taille");
  int percent = static_cast<int>(std::lround(d.subtitle_scale * 100.0F));
  if (ImGui::SliderInt("Taille (%)", &percent, static_cast<int>(limits::kSubtitleScale.lo * 100.0F),
                       static_cast<int>(limits::kSubtitleScale.hi * 100.0F))) {
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
  const std::optional<ProfileRow>& current = view.rows[static_cast<std::size_t>(view.active - 1)];
  if (current && rename[0] == '\0') {
    const std::size_t n = std::min(current->name.size(), rename.size() - 1);
    std::copy_n(current->name.data(), n, rename.data());
    rename[n] = '\0';
  }
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

void engines_tab(const SettingsChoices& choices) {
  section("Moteurs choisis");
  ImGui::TextWrapped("%s", choices.chosen_engines.c_str());
  note("Choisis à l'écran de bienvenue ; ajp-anime --first-run le remontre.");
  section("Moteurs TensorRT");
  if (choices.engines.empty()) note("Aucun moteur construit.");
  for (const EngineBadge& engine : choices.engines) {
    ImGui::TextUnformatted(engine.name.c_str());
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, engine.loaded ? kAccent : kNote);
    ImGui::TextUnformatted(engine.loaded ? "· chargé" : "· en cache");
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, kNote);
    ImGui::TextUnformatted(engine.detail.c_str());
    ImGui::PopStyleColor();
    ImGui::Spacing();
  }
  note("Un moteur est construit une fois par modèle et par taille, puis relu du cache.");
}

}  // namespace

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
    if (ImGui::BeginTabBar("##sections")) {
      if (ImGui::BeginTabItem("Profils")) {
        if (choices.profiles.active != shown_profile_) {  // nom à renommer : celui du profil actif
          rename_[0] = '\0';
          shown_profile_ = choices.profiles.active;
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
      if (ImGui::BeginTabItem("RTX HDR")) {
        hdr_tab(draft_, choices);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Grade")) {
        grade_tab(draft_, choices);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Sous-titres")) {
        subtitles_tab(draft_);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Son")) {
        sound_tab(draft_);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Stats")) {
        stats_tab(draft_);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Moteurs")) {
        engines_tab(choices);
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
    ImGui::PopItemWidth();

    // La part directe s'applique tout de suite ; le reste attend « Appliquer ».
    const PlaySettings live = live_part(running, draft_, choices.rife_loaded);
    if (live != running) result.live = live;
    const Change pending = change_between(live, draft_, choices.rife_loaded);
    if (pending.cost == ChangeCost::Reopen) {
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
      ImGui::TextWrapped("À appliquer : %s.", pending.reason.c_str());
      ImGui::PopStyleColor();
      if (ImGui::Button("Appliquer")) result.reopen = draft_;
      ImGui::SameLine();
      if (ImGui::Button("Annuler")) draft_ = live;
    }
  }
  ImGui::End();
  if (!keep) open_ = false;
  return result;
}

}  // namespace ajp::app::play
