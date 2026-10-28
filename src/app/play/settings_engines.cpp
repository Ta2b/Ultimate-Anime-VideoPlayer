#include "app/play/settings_engines.h"

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
                 std::vector<Input>& actions) {
  section("Moteur de SPAN et RIFE");
  const std::optional<core::EngineId> in_use = choices.engine_in_use;
  if (in_use) ImGui::TextUnformatted(std::format("En service : {}", engine_label(*in_use)).c_str());
  if (!choices.engine_note.empty()) note(choices.engine_note);
  if (choices.switchable.size() < 2) {
    note("Un seul moteur installé : « Installer d'autres moteurs » propose ceux qui conviennent à ce GPU.");
  } else {
    const core::EngineId shown = draft.value_or(in_use.value_or(core::EngineId::Vulkan));
    for (const core::EngineId id : choices.switchable) {
      if (ImGui::RadioButton(std::string(engine_label(id)).c_str(), shown == id)) draft = id;
    }
    if (draft && draft != in_use) {
      ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
      ImGui::TextWrapped(
          "À appliquer : la lecture se rouvre au même endroit avec %s ; ses moteurs se "
          "construisent la première fois pour chaque modèle et chaque taille.",
          std::string(engine_label(*draft)).c_str());
      ImGui::PopStyleColor();
      if (ImGui::Button("Utiliser ce moteur")) {
        actions.emplace_back(UseEngine{.id = *draft});
        draft.reset();
      }
      ImGui::SameLine();
      if (ImGui::Button("Annuler##moteur")) draft.reset();
    }
  }

  section("Moteurs installés");
  ImGui::TextWrapped("%s", choices.chosen_engines.c_str());
  if (ImGui::Button("Installer d'autres moteurs…")) actions.emplace_back(OpenEngineScreen{});
  note("Rouvre l'écran des moteurs sans relancer le lecteur ; la lecture reprend ensuite au même endroit.");

  section("Moteurs construits");
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

}  // namespace ajp::app::play
