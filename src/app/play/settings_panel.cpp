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

bool is_noise(std::string_view token) {
  // b5f48, SPANF3, unshuffle, fp16, 1x3xHxW, dyn-HW, op21, dynamo : la fabrique du modèle.
  if (token.starts_with("SPAN") || token == "unshuffle" || token == "fp16" || token == "dynamo" ||
      token.starts_with("dyn-") || token.starts_with("1x3x"))
    return true;
  if (token.size() >= 3 && token.starts_with("op") && std::isdigit(static_cast<unsigned char>(token[2])))
    return true;
  if (token.size() >= 4 && token[0] == 'b' && std::isdigit(static_cast<unsigned char>(token[1])) &&
      token.find('f') != std::string_view::npos)
    return true;
  return false;
}

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
        "Area : moyenne exacte, le profil du lecteur actuel. Catmull-Rom et Mitchell gardent plus de "
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
  if (four_k && d.rife_factor >= 3) {
    note(
        "En 4K sans réduction, ×2 tient sur cette carte ; ×3 et ×4 prennent du retard dans les scènes qui "
        "bougent (SPAN 1080p → 2160p coûte 11 ms par image).");
  } else if (four_k) {
    note("En 4K, RIFE estime le mouvement en 720p et déforme les images 1080p : ×2 tient sur cette carte.");
  }
  note("Les coupes de scène et les dessins tenus ne sont pas interpolés.");
}

void hdr_tab(PlaySettings& d, const SettingsChoices& choices) {
  section("RTX HDR (Maxine TrueHDR)");
  ImGui::BeginDisabled(!choices.true_hdr_available);
  ImGui::Checkbox("Activer RTX HDR", &d.true_hdr);
  ImGui::EndDisabled();
  if (!choices.true_hdr_available) note("SDK Maxine VideoFX introuvable (/usr/local/VideoFX).");
  ImGui::Spacing();
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
  ImGui::Checkbox("Débandage", &d.true_hdr_debanding);
  if (ImGui::Button("Valeurs par défaut")) {
    const PlaySettings defaults;
    d.true_hdr_peak = defaults.true_hdr_peak;
    d.true_hdr_contrast = defaults.true_hdr_contrast;
    d.true_hdr_saturation = defaults.true_hdr_saturation;
    d.true_hdr_gray = defaults.true_hdr_gray;
    d.true_hdr_debanding = defaults.true_hdr_debanding;
  }
  note(
      "Réglages appliqués en direct. Gris moyen : luminosité des tons moyens ; le débandage lisse les "
      "dégradés (le couper allège un peu RTX HDR).");
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

void engines_tab(const SettingsChoices& choices) {
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

std::string model_label(const std::string& file) {
  std::string_view name = file;
  if (name.ends_with(".onnx")) name.remove_suffix(5);
  if (name.starts_with("2x_")) name.remove_prefix(3);
  if (name.starts_with("AnimeJaNai_")) name.remove_prefix(11);
  std::string out;
  while (!name.empty()) {
    const std::size_t cut = name.find('_');
    const std::string_view token = name.substr(0, cut);
    if (!is_noise(token)) {
      std::string word(token);
      // « V3.1Sharp1 » : « V3.1 Sharp1 ».
      if (const std::size_t sharp = word.find("Sharp"); sharp != std::string::npos && sharp > 0)
        word.insert(sharp, " ");
      out += out.empty() ? word : " " + word;
    }
    if (cut == std::string_view::npos) break;
    name.remove_prefix(cut + 1);
  }
  return out.empty() ? file : out;
}

SettingsPanel::Result SettingsPanel::draw(const PlaySettings& running, const SettingsChoices& choices,
                                          float scale) {
  Result result;
  if (!open_) return result;
  // Réglages changés ailleurs (Tab, volume) : le panneau les reprend, sans perdre ce qui attend.
  if (running != seen_) {
    draft_ = live_part(draft_, running, choices.rife_loaded);
    seen_ = running;
  }
  const ImGuiIO& io = ImGui::GetIO();
  const float pad = 18.0F * scale;
  const float width = std::min(520.0F * scale, io.DisplaySize.x - 2.0F * pad);
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
      if (ImGui::BeginTabItem("Sous-titres")) {
        subtitles_tab(draft_);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Son")) {
        sound_tab(draft_);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Statistiques")) {
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
