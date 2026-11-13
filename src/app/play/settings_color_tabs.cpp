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

void grade_tab(PlaySettings& d) {
  // Tri de l'ADR 0012 : la méthode HDR anime (guidée, par plan) fait le rendu. Restent les noirs profonds,
  // qui suivent l'épisode (le curseur est leur maximum, dans un épisode très clair : A38-16), et le look : la
  // LUT de base et « skin fix » fixes, le contraste et les éclats de -100 à 100 (0 : le réglage de
  // référence).
  section(i18n::tr("Color grading"));
  render::GradeSettings& g = d.grade_values;
  ImGui::SliderInt(i18n::tr("Saturation"), &g.anime_saturation, kAnimePercent.lo, kAnimePercent.hi, "%d %%",
                   ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderInt(i18n::tr("Contrast"), &g.anime_contrast, kAnimePercent.lo, kAnimePercent.hi, "%d %%",
                   ImGuiSliderFlags_AlwaysClamp);
  ImGui::SliderInt(i18n::tr("Highlights"), &g.anime_highlights, kAnimePercent.lo, kAnimePercent.hi, "%d %%",
                   ImGuiSliderFlags_AlwaysClamp);
  if (panel::advanced()) {
    ImGui::SliderInt(i18n::tr("LUT contrast intensity"), &d.look_contrast, -100, 100, "%d",
                     ImGuiSliderFlags_AlwaysClamp);
    ImGui::SliderInt(i18n::tr("LUT highlight intensity"), &d.look_highlight, -100, 100, "%d",
                     ImGuiSliderFlags_AlwaysClamp);
    ImGui::SliderInt(i18n::tr("Surface white"), &d.surface_white, kSurfaceWhite.lo, kSurfaceWhite.hi, "%d %%",
                     ImGuiSliderFlags_AlwaysClamp);
    panel::tip(
        i18n::tr("HDR: the white of walls and skin, in % of where the LUTs put it (about 70 % of the screen "
                 "peak). Lower leaves more room to the lights, which still reach the peak."));
    grade_slider(i18n::tr("Deep blacks"), g.shadows, render::grade_limits::kShadows);
    panel::tip(i18n::tr("The maximum, reached in a very bright episode; it fades out in a very dark one."));
    ImGui::Checkbox(i18n::tr("Skin from faces (test)"), &d.face_skin);
    panel::tip(
        i18n::tr("Faces are found ahead of playback and the grade only treats their skin tones as skin: warm "
                 "lamps of another tone light up again. A shot without a found face keeps the wide range. "
                 "Takes effect with the next video."));
  }
  ImGui::Checkbox(i18n::tr("Limit rapid flashes"), &d.flash_guard);
  panel::tip(
      i18n::tr("HDR only: a burst of rapid flashes (more than three a second) is dimmed for a second, for "
               "photosensitive viewers. Off: flashes keep their full light."));
  ImGui::Checkbox(i18n::tr("LUTs in one table (faster, test)"), &d.look_table);
  panel::tip(
      i18n::tr("The LUT stack is computed once into one table on the GPU, then read once per pixel instead "
               "of LUT by LUT. Faster, but not exact: a few PQ codes off at worst on saturated colors."));
  if (ImGui::Button(i18n::tr("Defaults"))) {
    const PlaySettings defaults;
    g.anime_saturation = defaults.grade_values.anime_saturation;
    g.anime_contrast = defaults.grade_values.anime_contrast;
    g.anime_highlights = defaults.grade_values.anime_highlights;
    d.look_contrast = defaults.look_contrast;
    d.look_highlight = defaults.look_highlight;
    d.flash_guard = defaults.flash_guard;
    d.look_table = defaults.look_table;
    d.face_skin = defaults.face_skin;
    d.surface_white = defaults.surface_white;
    g.shadows = defaults.grade_values.shadows;
  }
}

}  // namespace ajp::app::play
