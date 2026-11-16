#include "app/play/settings_color_tabs.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/panel_style.h"
#include "render/grade_params.h"

namespace ajp::app::play {

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
  ImGui::SliderInt(panel::left(label).c_str(), &value, kAnimePercent.lo, kAnimePercent.hi, "%d %%",
                   ImGuiSliderFlags_AlwaysClamp);
  panel::live_preview();
  panel::tip(label, body);
}

void colors_section(PlaySettings& d) {
  // Tri de l'ADR 0012 : la méthode HDR anime (guidée, par plan) fait le rendu, en HDR comme en SDR. Restent
  // ses trois dosages, les noirs profonds (le curseur est leur maximum, dans un épisode très clair : A38-16)
  // et l'intensité des LUT de contraste et d'éclats (0 : la pile de référence).
  section(i18n::tr("Colors"));
  render::GradeSettings& g = d.grade_values;
  percent_slider(i18n::tr("Saturation"), g.anime_saturation,
                 i18n::tr("Color intensity. Recommended: 100 %."));
  percent_slider(i18n::tr("Contrast"), g.anime_contrast,
                 i18n::tr("Gap between dark and bright areas. Recommended: 100 %."));
  percent_slider(i18n::tr("Highlights"), g.anime_highlights,
                 i18n::tr("Brightness of lights and glints. Recommended: 100 %."));
  grade_slider(i18n::tr("Deep blacks"), g.shadows, render::grade_limits::kShadows);
  panel::live_preview();
  panel::tip(i18n::tr("Deep blacks"),
             i18n::tr("Makes dark scenes deeper and richer, less so in episodes that are dark throughout. "
                      "Lower it if shadows lose detail. Recommended: 0.50."));
  ImGui::SliderInt(panel::left(i18n::tr("LUT contrast")).c_str(), &d.look_contrast, -100, 100, "%d",
                   ImGuiSliderFlags_AlwaysClamp);
  panel::live_preview();
  panel::tip(
      i18n::tr("LUT contrast"),
      i18n::tr(
          "More or less contrast than the reference look. Higher makes colors punchier; lower is softer and "
          "closer to the source. 0: the reference look; −100: removed; 100: doubled. Recommended: 0."));
  ImGui::SliderInt(panel::left(i18n::tr("LUT highlights")).c_str(), &d.look_highlight, -100, 100, "%d",
                   ImGuiSliderFlags_AlwaysClamp);
  panel::live_preview();
  panel::tip(
      i18n::tr("LUT highlights"),
      i18n::tr("How much lights, sparks and skies shine. Higher makes them brighter; lower stays closer to "
               "the source. 0: the reference look; −100: removed; 100: doubled. Recommended: 0."));
  if (ImGui::Button(i18n::tr("Defaults##couleurs"))) {
    const PlaySettings defaults;
    g.anime_saturation = defaults.grade_values.anime_saturation;
    g.anime_contrast = defaults.grade_values.anime_contrast;
    g.anime_highlights = defaults.grade_values.anime_highlights;
    g.shadows = defaults.grade_values.shadows;
    d.look_contrast = defaults.look_contrast;
    d.look_highlight = defaults.look_highlight;
  }
}

// Un curseur sur un réglage du grade (double) ; true s'il a bougé. La valeur revient arrondie au
// centième et bornée : le float du curseur à 0,8 vaut 0,800000012 en double, au-dessus de la borne
// de check_grade_settings, et une valeur saisie au clavier (Ctrl + clic) peut sortir de la plage ;
// l'une comme l'autre faisait refuser tout le grade.
bool grade_slider(const char* label, double& value, core::Range<double> range) {
  auto shown = static_cast<float>(value);
  if (!ImGui::SliderFloat(panel::left(label).c_str(), &shown, static_cast<float>(range.lo),
                          static_cast<float>(range.hi), "%.2f", ImGuiSliderFlags_AlwaysClamp))
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
  bool automatic = d.true_hdr_peak == 0;
  if (ImGui::Checkbox(i18n::tr_format("Display peak: auto ({} nits)", choices.auto_peak).c_str(), &automatic))
    d.true_hdr_peak =
        automatic ? 0 : std::clamp(choices.auto_peak, limits::kTrueHdrPeak.lo, limits::kTrueHdrPeak.hi);
  panel::tip(i18n::tr("Display peak"),
             i18n::tr("How bright the brightest lights get. Read from the display; set it by hand if "
                      "highlights look clipped or dull. Recommended: auto."));
  ImGui::BeginDisabled(automatic);
  int peak = automatic ? choices.auto_peak : d.true_hdr_peak;
  if (ImGui::SliderInt(panel::left(i18n::tr("Peak (nits)")).c_str(), &peak, limits::kTrueHdrPeak.lo,
                       limits::kTrueHdrPeak.hi, "%d", ImGuiSliderFlags_AlwaysClamp) &&
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
  ImGui::SliderInt(panel::left(i18n::tr("Surface white")).c_str(), &d.surface_white, kSurfaceWhite.lo,
                   kSurfaceWhite.hi, "%d %%", ImGuiSliderFlags_AlwaysClamp);
  panel::live_preview();
  panel::tip(i18n::tr("Surface white"),
             i18n::tr("How bright walls and skin are. Lower makes the picture calmer and lets lights stand "
                      "out more; they still reach the peak. Recommended: 100 %."));
  ImGui::Checkbox(i18n::tr("Limit rapid flashes"), &d.flash_guard);
  panel::tip(i18n::tr("Limit rapid flashes"),
             i18n::tr("Dims bursts of more than three flashes a second for one second, for photosensitive "
                      "viewers. Recommended: on."));
  panel::end_group();  // l'encadré se ferme dans la zone grisée
  ImGui::EndDisabled();

  panel::advanced_header();
  section(i18n::tr("Grade"));
  ImGui::BeginDisabled(!choices.hdr_output);
  ImGui::Checkbox(i18n::tr("Grade before interpolation"), &d.grade_before_warp);
  ImGui::EndDisabled();
  panel::tip(i18n::tr("Grade before interpolation"),
             i18n::tr("Grades each source frame once instead of every displayed frame: frees the card at "
                      "high multipliers. Moving edges can show slight color differences. Speed: up to 8× "
                      "less grading at ×8. Quality impact: minor, on moving edges. Recommended: off."));
  ImGui::Checkbox(i18n::tr("LUTs in one table"), &d.look_table);
  panel::tip(i18n::tr("LUTs in one table"),
             i18n::tr("Makes the HDR grade about 4 times lighter for the card, with no visible difference. "
                      "Turn it off only to compare. Speed: grading about 4× faster. Quality impact: "
                      "imperceptible. Recommended: on."));
  ImGui::Checkbox(i18n::tr("Skin from faces (test)"), &d.face_skin);
  panel::tip(
      i18n::tr("Skin from faces (test)"),
      i18n::tr(
          "Warm lamps in scenes with faces light up again, instead of being toned down like skin. A trial: "
          "some faces can be missed. Quality impact: minor gain, not yet judged. Recommended: off."),
      panel::Apply::Reload);
  if (ImGui::Button(i18n::tr("Defaults##hdr"))) {
    const PlaySettings defaults;
    d.true_hdr_peak = defaults.true_hdr_peak;
    d.surface_white = defaults.surface_white;
    d.flash_guard = defaults.flash_guard;
    d.grade_before_warp = defaults.grade_before_warp;
    d.look_table = defaults.look_table;
    d.face_skin = defaults.face_skin;
  }
}

}  // namespace ajp::app::play
