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

  section("Dossier d'installation");
  // Un moteur téléchargé (TensorRT : 1,3 Go installés pour une RTX 4070 Ti) et ses téléchargements
  // s'installent ici ; l'espace libre se vérifie AVANT chaque téléchargement. Vide : le dossier des données
  // du lecteur.
  static std::array<char, 512> folder{};
  static bool folder_read = false;
  if (!folder_read) {
    std::snprintf(folder.data(), folder.size(), "%s", engines_dir.c_str());
    folder_read = true;
  }
  if (ImGui::InputText("Dossier##moteurs", folder.data(), folder.size())) engines_dir = folder.data();
  ImGui::SameLine();
  if (ImGui::Button("Par défaut##moteurs")) {
    engines_dir.clear();
    folder[0] = '\0';
  }
  note(
      std::format("Par défaut : {}. Libre à cet endroit : {}. Vaut au prochain lancement ; les moteurs déjà "
                  "installés ne sont pas déplacés.",
                  choices.engines_dir_default, choices.engines_dir_free));

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
