#include "app/play/settings_color_tabs.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <string>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/panel_style.h"
#include "render/grade_params.h"

namespace kirari::app::play {

using panel::note;
using panel::section;
using render::grade_limits::kAnimePercent;

void output_section(PlaySettings& d, const SettingsChoices& choices) {
  section(i18n::tr("Output"));
  if (ImGui::BeginCombo(panel::left(i18n::tr("HDR")).c_str(), std::string(label(d.hdr_mode)).c_str())) {
    for (int i = 0; i < kHdrModeCount; ++i) {
      const auto mode = static_cast<HdrMode>(i);
      if (ImGui::Selectable(std::string(label(mode)).c_str(), mode == d.hdr_mode)) d.hdr_mode = mode;
    }
    ImGui::EndCombo();
  }
  panel::tip(
      i18n::tr("HDR"),
      i18n::tr("Auto: HDR when the system is in HDR and the display is bright enough (350 nits). On: HDR "
               "whenever the system is in HDR. Off: SDR. HDR needs HDR turned on in the system. "
               "Recommended: Auto."));
  note(choices.hdr_output ? i18n::tr("Now: HDR10.") : i18n::tr("Now: SDR."));
}

// Un curseur en % du grade anime (100 % : la méthode telle que réglée), qui efface le panneau tant qu'il est
// tenu.
void percent_slider(const char* label, int& value, const char* body) {
  const std::string id = panel::left(label);
  panel::slider_int(id.c_str(), &value, kAnimePercent.lo, kAnimePercent.hi, std::format("{} %", value));
  panel::live_preview();
  panel::tip(label, body);
}

void colors_section(PlaySettings& d, bool source_hdr) {
  // Tri de l'ADR 0012 : la méthode HDR anime (guidée, par plan) fait le rendu, en HDR comme en SDR. Restent
  // ses trois dosages, les noirs profonds (le curseur est leur maximum, dans un épisode très clair : A38-16)
  // et l'intensité des LUT de contraste et d'éclats (0 : la pile de référence).
  section(i18n::tr("Colors"));
  // Une source déjà HDR n'est pas gradée (source_fit.h) : ses couleurs restent celles du disque.
  ImGui::BeginDisabled(source_hdr);
  render::GradeSettings& g = d.grade_values;
  // Deux curseurs par ligne, comme chez Dolphin.
  if (!panel::begin_grid("##couleurs")) {
    ImGui::EndDisabled();
    return;
  }
  panel::cell();
  percent_slider(i18n::tr("Saturation"), g.anime_saturation,
                 i18n::tr("Color intensity. Recommended: 100 %."));
  panel::cell();
  percent_slider(i18n::tr("Contrast"), g.anime_contrast,
                 i18n::tr("Gap between dark and bright areas. Recommended: 100 %."));
  panel::cell();
  percent_slider(i18n::tr("Highlights"), g.anime_highlights,
                 i18n::tr("Brightness of lights and glints. Recommended: 100 %."));
  panel::cell();
  grade_slider(i18n::tr("Deep blacks"), g.shadows, render::grade_limits::kShadows);
  panel::live_preview();
  panel::tip(i18n::tr("Deep blacks"),
             i18n::tr("Makes dark scenes deeper and richer, less so in episodes that are dark throughout. "
                      "Lower it if shadows lose detail. Recommended: 0.50."));
  panel::cell();
  const std::string contrast = panel::left(i18n::tr("LUT contrast"));
  panel::slider_int(contrast.c_str(), &d.look_contrast, -100, 100, std::to_string(d.look_contrast));
  panel::live_preview();
  panel::tip(
      i18n::tr("LUT contrast"),
      i18n::tr(
          "More or less contrast than the reference look. Higher makes colors punchier; lower is softer and "
          "closer to the source. 0: the reference look; −100: removed; 100: doubled. Recommended: 0."));
  panel::cell();
  const std::string highlight = panel::left(i18n::tr("LUT highlights"));
  panel::slider_int(highlight.c_str(), &d.look_highlight, -100, 100, std::to_string(d.look_highlight));
  panel::live_preview();
  panel::tip(
      i18n::tr("LUT highlights"),
      i18n::tr("How much lights, sparks and skies shine. Higher makes them brighter; lower stays closer to "
               "the source. 0: the reference look; −100: removed; 100: doubled. Recommended: 0."));
  panel::end_grid();
  if (ImGui::Button(i18n::tr("Defaults##couleurs"))) {
    const PlaySettings defaults;
    g.anime_saturation = defaults.grade_values.anime_saturation;
    g.anime_contrast = defaults.grade_values.anime_contrast;
    g.anime_highlights = defaults.grade_values.anime_highlights;
    g.shadows = defaults.grade_values.shadows;
    d.look_contrast = defaults.look_contrast;
    d.look_highlight = defaults.look_highlight;
  }
  ImGui::EndDisabled();
  if (source_hdr)
    note(i18n::tr("Greyed out: the video is already HDR, it is shown without the anime grade."));
}

// Un curseur sur un réglage du grade (double) ; true s'il a bougé. La valeur revient arrondie au
// centième et bornée : le float du curseur à 0,8 vaut 0,800000012 en double, au-dessus de la borne
// de check_grade_settings, et une valeur saisie au clavier (Ctrl + clic) peut sortir de la plage ;
// l'une comme l'autre faisait refuser tout le grade.
bool grade_slider(const char* label, double& value, core::Range<double> range) {
  auto shown = static_cast<float>(value);
  const std::string id = panel::left(label);
  if (!panel::slider_float(id.c_str(), &shown, static_cast<float>(range.lo), static_cast<float>(range.hi),
                           std::format("{:.2f}", value)))
    return false;
  value = std::clamp(std::round(static_cast<double>(shown) * 100.0) / 100.0, range.lo, range.hi);
  return true;
}

// Vidéo > HDR : le pic de l'écran, le blanc des surfaces, le limiteur de flashs ; en avancé, ce qui touche au
// calcul du grade. Grisé en sortie SDR, sauf le grade, qui sert aussi en SDR.
void hdr_page(PlaySettings& d, const SettingsChoices& choices) {
  if (!choices.hdr_output)
    panel::wrapped(i18n::tr("The output is SDR: these settings apply when it is HDR (Video > General)."));
  ImGui::BeginDisabled(!choices.hdr_output);
  section(i18n::tr("Brightness"));
  // Deux réglages par ligne, comme chez Dolphin. Le blanc des surfaces et le limiteur de flashs sont ceux
  // du grade anime : grisés sur une source déjà HDR, qu'il ne retouche pas.
  if (panel::begin_grid("##luminosite")) {
    panel::cell();
    bool automatic = d.true_hdr_peak == 0;
    if (ImGui::Checkbox(i18n::tr_format("Display peak: auto ({} nits)", choices.auto_peak).c_str(),
                        &automatic))
      d.true_hdr_peak =
          automatic ? 0 : std::clamp(choices.auto_peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi);
    panel::tip(i18n::tr("Display peak"),
               i18n::tr("How bright the brightest lights get. Read from the display; set it by hand if "
                        "highlights look clipped or dull. Recommended: auto."));
    panel::cell();
    ImGui::BeginDisabled(automatic);
    int peak = automatic ? choices.auto_peak : d.true_hdr_peak;
    const std::string peak_id = panel::left(i18n::tr("Peak (nits)"));
    if (panel::slider_int(peak_id.c_str(), &peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi,
                          std::to_string(peak)) &&
        !automatic)
      d.true_hdr_peak = peak;
    panel::live_preview();
    ImGui::EndDisabled();
#if defined(__linux__)
    // Sous KDE, le pic est en nits du contenu : KWin les affiche × luminosité SDR / 203 (system_peak.cpp).
    // Windows affiche le PQ d'une application tel quel : contenu = écran, rien à dire.
    panel::tip(i18n::tr("Peak (nits)"),
               i18n::tr("In content nits: KDE shows them brighter (× its SDR brightness / 203)."));
#endif
    ImGui::BeginDisabled(choices.source_hdr);
    panel::cell();
    const std::string white_id = panel::left(i18n::tr("Surface white"));
    panel::slider_int(white_id.c_str(), &d.surface_white, kSurfaceWhite.lo, kSurfaceWhite.hi,
                      std::format("{} %", d.surface_white));
    panel::live_preview();
    panel::tip(i18n::tr("Surface white"),
               i18n::tr("How bright walls and skin are. Lower makes the picture calmer and lets lights stand "
                        "out more; they still reach the peak. Recommended: 100 %."));
    panel::cell();
    ImGui::Checkbox(i18n::tr("Limit rapid flashes"), &d.flash_guard);
    panel::tip(i18n::tr("Limit rapid flashes"),
               i18n::tr("Dims bursts of more than three flashes a second for one second, for photosensitive "
                        "viewers. Recommended: on."));
    ImGui::EndDisabled();
    panel::end_grid();
  }
  panel::end_group();  // l'encadré se ferme dans la zone grisée
  ImGui::EndDisabled();

  panel::advanced_header();
  section(i18n::tr("Grade"));
  // Rien du grade ne tourne sur une source déjà HDR. Le grade avant le warp et le look en une table n'ont
  // plus de case (novembre 2026) : toujours pris là où ils s'appliquent (source_grade.h, look_table.h).
  ImGui::BeginDisabled(choices.source_hdr);
  const bool grid = panel::begin_grid("##grade");
  if (grid) panel::cell();
  ImGui::Checkbox(i18n::tr("Skin from faces (test)"), &d.face_skin);
  panel::tip(
      i18n::tr("Skin from faces (test)"),
      i18n::tr(
          "Warm lamps in scenes with faces light up again, instead of being toned down like skin. A trial: "
          "some faces can be missed. Quality impact: minor gain, not yet judged. Recommended: off."),
      panel::Apply::Reload);
  if (grid) panel::end_grid();
  ImGui::EndDisabled();
  if (ImGui::Button(i18n::tr("Defaults##hdr"))) {
    const PlaySettings defaults;
    d.true_hdr_peak = defaults.true_hdr_peak;
    d.surface_white = defaults.surface_white;
    d.flash_guard = defaults.flash_guard;
    d.face_skin = defaults.face_skin;
  }
}

}  // namespace kirari::app::play
