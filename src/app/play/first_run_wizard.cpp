// L'assistant du premier lancement à l'écran : un écran par étape (wizard.h), la barre de progression et
// les boutons Retour, Passer, Suivant ; Prêt finit par « Calibrer maintenant » ou « Régler moi-même ».
#include <algorithm>
#include <string>

#include <imgui.h>

#include "app/play/first_run.h"
#include "app/play/first_run_draw.h"
#include "app/play/i18n.h"
#include "app/play/machine_calibration.h"
#include "app/play/panel_style.h"
#include "app/play/settings_panel.h"
#include "app/play/wizard.h"

namespace ajp::app::play {
namespace {

void radio_note(const char* label, const char* detail, bool selected, bool& pick, bool enabled = true) {
  ImGui::BeginDisabled(!enabled);
  if (ImGui::RadioButton(label, selected)) pick = true;
  ImGui::EndDisabled();
  if (!enabled)
    panel::tip(detail);
  else
    note(detail);
}

// Le contenu d'un écran de l'assistant.
void draw_wizard_step(FirstRunView& view, Wizard& wizard) {
  switch (current(wizard)) {
    case WizardStep::Welcome:
      section(i18n::tr("Welcome"));
      panel::wrapped(view.gpu.c_str());
      note(view.tier_line);
      if (!view.display_line.empty()) note(view.display_line);
      note(i18n::tr(
          "The player picks the settings for this machine; everything is in Settings (p) afterwards."));
      {
        // La langue de l'interface (P3.18), détectée au lancement, confirmée ici ; prise en direct.
        const bool french = i18n::language() == i18n::Language::French;
        if (ImGui::BeginCombo(i18n::tr("Interface language"), french ? "Français" : "English")) {  // i18n-ok
          if (ImGui::Selectable("English", !french)) wizard.interface_language = "en";             // i18n-ok
          if (ImGui::Selectable("Français", french)) wizard.interface_language = "fr";             // i18n-ok
          ImGui::EndCombo();
        }
        if (!wizard.interface_language.empty()) {
          const i18n::Language wanted = i18n::language_from_code(wizard.interface_language);
          if (wanted != i18n::language()) {
            i18n::set_language(wanted);
            if (view.retranslate) view.retranslate(view);
          }
        }
      }
      break;
    case WizardStep::Engines:
      section(i18n::tr("Engines"));
      for (EngineRow& row : view.engines) engine_row(row);
      note(
          i18n::tr("Vulkan works on every GPU. Checked engines install at the end of the assistant, verified "
                   "by hash; "
                   "Vulkan takes over if one fails."));
      break;
    case WizardStep::Languages:
      section(i18n::tr("Languages"));
      language_choice(i18n::tr("Audio language"), i18n::tr("From the file"), wizard.audio_language);
      language_choice(i18n::tr("Subtitle language"), i18n::tr("None"), wizard.subtitle_language);
      break;
    case WizardStep::Updates: {
      section(i18n::tr("Updates"));
      bool pick = false;
      radio_note(i18n::tr("Notify me"),
                 i18n::tr("A notification with the link to the new version; you install it yourself."),
                 wizard.update_mode == UpdateMode::Notify, pick);
      if (pick) wizard.update_mode = UpdateMode::Notify;
      pick = false;
      radio_note(i18n::tr("Never"), i18n::tr("No checks."), wizard.update_mode == UpdateMode::Never, pick);
      if (pick) wizard.update_mode = UpdateMode::Never;
      section(i18n::tr("Default player"));
      ImGui::Checkbox(i18n::tr("Make it the default video player"), &wizard.make_default);
      note(i18n::tr("AJP is added to \"Open with\" in any case."));
      break;
    }
    case WizardStep::Summary:
      section(i18n::tr("Ready"));
      for (const std::string& line : engines_summary(view.engines)) ImGui::BulletText("%s", line.c_str());
      for (const std::string& line : wizard_summary(wizard)) ImGui::BulletText("%s", line.c_str());
      section(i18n::tr("Machine calibration"));
      note(
          i18n::tr("\"Calibrate now\" tests several settings on a very demanding sequence: picture "
                   "resolution, SPAN upscaling and RIFE smoothness, and keeps the best combination this "
                   "machine sustains. Only a progress bar is shown; it takes a few minutes, longer the first "
                   "time (the engines are built once)."));
      // Calculée au moment de l'afficher, d'après les moteurs cochés maintenant : TensorRT choisi à l'écran
      // des moteurs construit ses moteurs pendant l'essai (A25-03).
      note(i18n::tr_format("Estimated for this machine: about {} minutes.",
                           calibration_minutes(std::ranges::any_of(view.engines,
                                                                   [](const EngineRow& row) {
                                                                     return row.offer.id ==
                                                                                core::EngineId::TensorRt &&
                                                                            row.chosen && row.usable &&
                                                                            engine_installs(row);
                                                                   }),
                                               view.laptop)));
      note(
          i18n::tr("\"Set it up myself\": the Auto preset estimated for this card, changeable in Video > "
                   "Post-processing."));
      note(i18n::tr("Everything can be changed later in Settings (p). To redo this: ajp --first-run."));
      break;
  }
}

// Entrée sans rien de choisi au clavier vaut « Suivant » (A25-14) ; un choix en surbrillance ou une liste
// ouverte la gardent pour eux. Pas sur Prêt : la calibration ne part pas d'une touche tapée trop vite.
bool enter_goes_next() {
  const ImGuiIO& io = ImGui::GetIO();
  return (io.ConfigFlags & ImGuiConfigFlags_NavEnableKeyboard) != 0 && !io.NavVisible &&
         !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId) &&
         (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false));
}

}  // namespace

void draw_wizard(FirstRunView& view, float scale) {
  const ImGuiIO& io = ImGui::GetIO();
  const float pad = 24.0F * scale;
  const ImVec2 size{std::min(860.0F * scale, io.DisplaySize.x - 2.0F * pad),
                    std::min(640.0F * scale, io.DisplaySize.y - 2.0F * pad)};
  ImGui::SetNextWindowPos({io.DisplaySize.x * 0.5F, io.DisplaySize.y * 0.5F}, ImGuiCond_Always, {0.5F, 0.5F});
  ImGui::SetNextWindowSize(size, ImGuiCond_Always);
  constexpr ImGuiWindowFlags kFlags = ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                                      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
  Wizard& wizard = view.wizard;
  if (ImGui::Begin(i18n::tr("Welcome"), nullptr, kFlags)) {
    note(progress_text(wizard));
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, kAccent);  // la couleur d'accent, pas le jaune par défaut
    ImGui::ProgressBar(static_cast<float>(wizard.at + 1) / static_cast<float>(wizard.steps.size()),
                       {-1.0F, 6.0F * scale}, "");
    ImGui::PopStyleColor();
    const float footer = ImGui::GetFrameHeight() * 1.6F + ImGui::GetStyle().ItemSpacing.y * 3.0F;
    // NavFlattened : au clavier, Tab passe des choix de l'écran aux boutons sans s'arrêter sur le cadre.
    ImGui::BeginChild("##ecran", {0.0F, -footer}, ImGuiChildFlags_NavFlattened);
    draw_wizard_step(view, wizard);
    ImGui::EndChild();
    ImGui::Separator();
    const float tall = ImGui::GetFrameHeight() * 1.4F;
    ImGui::BeginDisabled(is_first(wizard));
    if (ImGui::Button(i18n::tr("Back"), {0.0F, tall})) previous(wizard);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (!is_last(wizard)) {
      if (ImGui::Button(i18n::tr("Skip"), {0.0F, tall})) skip(wizard);
      ImGui::SameLine();
    }
    ImGui::PushStyleColor(ImGuiCol_Button, kAccent);
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4{0.05F, 0.07F, 0.12F, 1.0F});
    if (is_last(wizard)) {
      if (ImGui::Button(i18n::tr("Calibrate now"), {0.0F, tall})) {
        wizard.calibrate = true;
        view.done = true;
      }
    } else if (ImGui::Button(i18n::tr("Next"), {0.0F, tall}) || enter_goes_next()) {
      next(wizard);
    }
    ImGui::SetItemDefaultFocus();  // au clavier, Entrée part du bouton principal
    ImGui::PopStyleColor(2);
    // Le bouton secondaire de Prêt : sans calibration, le niveau Auto estimé pour cette carte (P3.1).
    if (is_last(wizard)) {
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Set it up myself"), {0.0F, tall})) {
        set_up_myself(wizard);
        view.done = true;
      }
    }
    if (view.during_playback) {
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Cancel"), {0.0F, tall})) view.cancelled = true;
    }
  }
  ImGui::End();
}

}  // namespace ajp::app::play
