#include "app/play/settings_engines.h"

#include <array>
#include <cstdio>
#include <format>
#include <string>

#include <imgui.h>

#include "app/play/first_run.h"
#include "app/play/panel_style.h"

namespace ajp::app::play {

using panel::kAccent;
using panel::kNote;
using panel::kWarn;
using panel::note;
using panel::section;

void engines_tab(const SettingsChoices& choices, std::optional<core::EngineId>& draft,
                 std::vector<Input>& actions, std::string& engines_dir) {
  section(i18n::tr("Active engine"));
  const std::optional<core::EngineId> in_use = choices.engine_in_use;
  if (in_use) ImGui::TextUnformatted(std::string(engine_label(*in_use)).c_str());
  if (!choices.engine_note.empty()) note(choices.engine_note);
  if (choices.switchable.size() < 2) {
    note(i18n::tr("Only one engine is installed: \"Install more engines\" lists those that suit this GPU."));
  } else {
    const core::EngineId shown = draft.value_or(in_use.value_or(core::EngineId::Vulkan));
    for (const core::EngineId id : choices.switchable) {
      if (ImGui::RadioButton(std::string(engine_label(id)).c_str(), shown == id)) draft = id;
    }
    if (draft && draft != in_use) {
      ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
      ImGui::TextWrapped(
          "%s", i18n::tr_format(i18n::tr("To apply: playback reopens at the same spot with {}; its engines "
                                         "are built the first time for each model and size."),
                                engine_label(*draft))
                    .c_str());
      ImGui::PopStyleColor();
      if (ImGui::Button(i18n::tr("Use this engine"))) {
        actions.emplace_back(UseEngine{.id = *draft});
        draft.reset();
      }
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Cancel##moteur"))) draft.reset();
    }
  }

  section(i18n::tr("Installed engines"));
  ImGui::TextWrapped("%s", choices.chosen_engines.c_str());
  if (ImGui::Button(i18n::tr("Install more engines…"))) actions.emplace_back(OpenEngineScreen{});
  note(i18n::tr("Reopens the engines screen; playback resumes at the same spot."));

  section(i18n::tr("Built engines"));
  if (choices.engines.empty()) note(i18n::tr("No engine built yet."));
  for (const EngineBadge& engine : choices.engines) {
    ImGui::TextWrapped("%s", engine.name.c_str());  // un nom long passe à la ligne, il ne déborde pas
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, engine.loaded ? kAccent : kNote);
    ImGui::TextUnformatted(engine.loaded ? i18n::tr("· loaded") : i18n::tr("· cached"));
    ImGui::PopStyleColor();
    ImGui::PushStyleColor(ImGuiCol_Text, kNote);
    ImGui::TextWrapped("%s", engine.detail.c_str());
    ImGui::PopStyleColor();
    ImGui::Spacing();
  }
  note(i18n::tr("An engine is built once per model and size, then read from the cache."));

  section(i18n::tr("Install folder"));
  // Un moteur téléchargé (TensorRT : 1,3 Go installés pour une RTX 4070 Ti) et ses téléchargements
  // s'installent ici ; l'espace libre se vérifie AVANT chaque téléchargement. Vide : le dossier des données
  // du lecteur.
  static std::array<char, 512> folder{};
  static bool folder_read = false;
  if (!folder_read) {
    std::snprintf(folder.data(), folder.size(), "%s", engines_dir.c_str());
    folder_read = true;
  }
  const ImGuiStyle& style = ImGui::GetStyle();
  ImGui::SetNextItemWidth(
      -(ImGui::CalcTextSize(i18n::tr("Default")).x + 2.0F * style.FramePadding.x + style.ItemSpacing.x));
  if (ImGui::InputTextWithHint("##moteurs", i18n::tr("Default folder"), folder.data(), folder.size()))
    engines_dir = folder.data();
  ImGui::SameLine();
  if (ImGui::Button(i18n::tr("Default##moteurs"))) {
    engines_dir.clear();
    folder[0] = '\0';
  }
  note(i18n::tr_format("Default: {}. Free here: {}. Applies at next launch; installed engines are not moved.",
                       choices.engines_dir_default, choices.engines_dir_free));
}

}  // namespace ajp::app::play
