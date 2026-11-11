#include "app/play/settings_engines.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <system_error>

#include <imgui.h>

#include "app/play/first_run.h"
#include "app/play/panel_style.h"
#include "prepare/utf8.h"

namespace ajp::app::play {

using panel::kAccent;
using panel::kNote;
using panel::kWarn;
using panel::note;
using panel::section;

namespace {

// Le dossier accepte-t-il l'écriture ? Un fichier d'essai l'établit ; un dossier qui n'existe pas encore se
// juge sur son parent, où la création le posera.
bool writable_folder(const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::path where = std::filesystem::is_directory(path, error) ? path : path.parent_path();
  const std::filesystem::path probe = where / ".ajp-ecriture-essai";
  {
    std::ofstream out(probe, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << 'x';
    if (!out.flush()) return false;
  }
  std::filesystem::remove(probe, error);
  return true;
}

}  // namespace

void engines_tab(const SettingsChoices& choices, std::optional<core::EngineId>& draft,
                 std::vector<Input>& actions, std::string& engines_dir, FolderField& folder) {
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
      panel::wrapped(
          i18n::tr_format(i18n::tr("To apply: playback reopens at the same spot with {}; its engines "
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
  panel::wrapped(choices.chosen_engines.c_str());
  if (ImGui::Button(i18n::tr("Install more engines…"))) actions.emplace_back(OpenEngineScreen{});
  note(i18n::tr("Reopens the engines screen; playback resumes at the same spot."));

  section(i18n::tr("Built engines"));
  if (choices.engines.empty()) note(i18n::tr("No engine built yet."));
  for (const EngineBadge& engine : choices.engines) {
    // L'état d'abord, sur sa ligne ; le nom, long, passe à la ligne sans déborder ni cacher l'état (B23-11).
    ImGui::PushStyleColor(ImGuiCol_Text, engine.loaded ? kAccent : kNote);
    ImGui::TextUnformatted(engine.loaded ? i18n::tr("Loaded") : i18n::tr("Cached"));
    ImGui::PopStyleColor();
    panel::wrapped(engine.name.c_str());
    ImGui::PushStyleColor(ImGuiCol_Text, kNote);
    panel::wrapped(engine.detail.c_str());
    ImGui::PopStyleColor();
    ImGui::Spacing();
  }
  note(i18n::tr("An engine is built once per model and size, then read from the cache."));

  section(i18n::tr("Install folder"));
  // Un moteur téléchargé (TensorRT : 1,3 Go installés pour une RTX 4070 Ti) et ses téléchargements
  // s'installent ici ; l'espace libre se vérifie AVANT chaque téléchargement. Vide : le dossier des données
  // du lecteur.
  // Le champ reprend le réglage quand il a changé ailleurs (Annuler, Rétablir, Tout réinitialiser, A23-16).
  if (!folder.read || folder.committed != engines_dir) {
    std::snprintf(folder.text.data(), folder.text.size(), "%s", engines_dir.c_str());
    folder.committed = engines_dir;
    folder.problem.clear();
    folder.read = true;
  }
  const ImGuiStyle& style = ImGui::GetStyle();
  ImGui::SetNextItemWidth(
      -(ImGui::CalcTextSize(i18n::tr("Default")).x + 2.0F * style.FramePadding.x + style.ItemSpacing.x));
  // Le dossier est pris à la fin de la saisie (Entrée ou clic ailleurs), une fois vérifié : pas à chaque
  // frappe, où play.toml se réécrivait avec un chemin à moitié tapé (A22-09).
  ImGui::InputTextWithHint("##moteurs", i18n::tr("Default folder"), folder.text.data(), folder.text.size());
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    const std::string typed = folder.text.data();
    std::error_code error;
    const std::filesystem::path path = prepare::path_from_utf8(typed);
    // Un chemin relatif dépendrait du dossier de lancement ; le dossier doit exister (ou son parent, que la
    // création fera) et accepter l'écriture, sinon le téléchargement échouerait plus tard (B23-14).
    if (typed.empty()) {
      engines_dir.clear();
      folder.committed.clear();
      folder.problem.clear();
    } else if (!path.is_absolute()) {
      folder.problem = i18n::tr("Give a full path (starting at the root of a drive).");
    } else if (!std::filesystem::is_directory(path, error) &&
               !(path.has_parent_path() && std::filesystem::is_directory(path.parent_path(), error))) {
      folder.problem = i18n::tr("This folder does not exist.");
    } else if (!writable_folder(path)) {
      folder.problem = i18n::tr("This folder cannot be written to.");
    } else {
      engines_dir = typed;
      folder.committed = typed;
      folder.problem.clear();
    }
  }
  ImGui::SameLine();
  if (ImGui::Button(i18n::tr("Default##moteurs"))) {
    engines_dir.clear();
    folder.committed.clear();
    folder.problem.clear();
    folder.text[0] = '\0';
  }
  if (!folder.problem.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    panel::wrapped(folder.problem.c_str());
    ImGui::PopStyleColor();
  }
  note(i18n::tr_format("Default: {}. Free here: {}. Applies at next launch; installed engines are not moved.",
                       choices.engines_dir_default, choices.engines_dir_free));
}

void stages_note(const SettingsChoices& choices, std::vector<Input>& actions) {
  if (choices.stage_note.empty()) return;
  ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
  ImGui::TextWrapped("%s", choices.stage_note.c_str());
  ImGui::PopStyleColor();
  if (ImGui::Button(i18n::tr("Retry"))) actions.emplace_back(RetryStages{});
  note(i18n::tr("Playback reopens at the same spot and prepares these engines again."));
}

}  // namespace ajp::app::play
