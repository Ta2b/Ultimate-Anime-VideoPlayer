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

std::string clean_folder_text(std::string_view typed) {
  const auto blank = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  while (!typed.empty() && blank(typed.front())) typed.remove_prefix(1);
  while (!typed.empty() && blank(typed.back())) typed.remove_suffix(1);
  if (typed.size() >= 2 && typed.front() == '"' && typed.back() == '"') {
    typed.remove_prefix(1);
    typed.remove_suffix(1);
  }
  return std::string(typed);
}

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

// La carte de la lecture, sur un poste qui en a deux ou plus (portables).
void gpu_choice(std::string& gpu, const std::vector<std::string>& gpus) {
  if (gpus.size() < 2) return;
  const char* shown = gpu.empty() ? i18n::tr("Automatic (dedicated card first)") : gpu.c_str();
  if (ImGui::BeginCombo(panel::left(i18n::tr("Graphics card")).c_str(), shown)) {
    if (ImGui::Selectable(i18n::tr("Automatic (dedicated card first)"), gpu.empty())) gpu.clear();
    for (const std::string& name : gpus)
      if (ImGui::Selectable(name.c_str(), gpu == name)) gpu = name;
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("Graphics card"), i18n::tr("The card that plays the video. Recommended: Automatic."),
             panel::Apply::Restart);
}

void install_folder_row(const SettingsChoices& choices, std::string& engines_dir, FolderField& folder) {
  ImGui::TextUnformatted(i18n::tr("Install folder"));
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
    const std::string typed = clean_folder_text(folder.text.data());
    std::snprintf(folder.text.data(), folder.text.size(), "%s",
                  typed.c_str());  // le champ montre ce qui compte
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
  panel::tip(i18n::tr("Install folder"),
             i18n::tr("Where downloaded engines go. Installed engines are not moved."),
             panel::Apply::Restart);
  note(i18n::tr_format("Default: {}. Free: {}.", choices.engines_dir_default, choices.engines_dir_free));
}

}  // namespace

void engine_section(const SettingsChoices& choices, std::optional<core::EngineId>& draft,
                    std::vector<Input>& actions, std::string& gpu) {
  section(i18n::tr("Engine"));
  const std::optional<core::EngineId> in_use = choices.engine_in_use;
  const core::EngineId shown = draft.value_or(in_use.value_or(core::EngineId::Vulkan));
  ImGui::BeginDisabled(choices.switchable.size() < 2);
  if (ImGui::BeginCombo(panel::left(i18n::tr("Backend")).c_str(), std::string(engine_label(shown)).c_str())) {
    for (const core::EngineId id : choices.switchable) {
      if (ImGui::Selectable(std::string(engine_label(id)).c_str(), shown == id))
        draft = id == in_use ? std::nullopt : std::optional(id);
    }
    ImGui::EndCombo();
  }
  ImGui::EndDisabled();
  panel::tip(
      i18n::tr("Backend"),
      i18n::tr("Runs SPAN and the interpolation. TensorRT: NVIDIA RTX, fastest, engines built once per "
               "model and size. Vulkan: every graphics card. Recommended: TensorRT on NVIDIA RTX, Vulkan "
               "elsewhere."),
      panel::Apply::Reload);
  if (!choices.engine_note.empty()) note(choices.engine_note);
  gpu_choice(gpu, choices.gpus);
  if (ImGui::Button(i18n::tr("Install more engines…"))) actions.emplace_back(OpenEngineScreen{});
  panel::tip(i18n::tr("Install more engines…"),
             i18n::tr("Opens the engines screen. Playback resumes at the same spot."));
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Text, kNote);
  ImGui::TextUnformatted(i18n::tr_format("Installed: {}", choices.chosen_engines).c_str());
  ImGui::PopStyleColor();
}

void engines_advanced(const SettingsChoices& choices, std::string& engines_dir, FolderField& folder,
                      bool& fast_precision) {
  section(i18n::tr("Engines"));
  // La porte du moteur Vulkan (gpu::kStrictPrecision, net_gate.h) : float16 gardé quand la perte ne se voit
  // pas. Sans effet sous TensorRT.
  ImGui::Checkbox(i18n::tr("Speed before precision (Vulkan)"), &fast_precision);
  panel::tip(i18n::tr("Speed before precision (Vulkan)"),
             i18n::tr("Vulkan engines keep half precision when the difference is invisible. Speed: about 2× "
                      "on upscaling. Quality impact: imperceptible. Recommended: on."),
             panel::Apply::Restart);
  install_folder_row(choices, engines_dir, folder);
  if (ImGui::TreeNodeEx(i18n::tr("Built engines"), ImGuiTreeNodeFlags_SpanAvailWidth)) {
    if (choices.engines.empty()) note(i18n::tr("No engine built yet."));
    for (const EngineBadge& engine : choices.engines) {
      // L'état d'abord, sur sa ligne ; le nom, long, passe à la ligne sans déborder ni cacher l'état
      // (B23-11).
      ImGui::PushStyleColor(ImGuiCol_Text, engine.loaded ? kAccent : kNote);
      ImGui::TextUnformatted(engine.loaded ? i18n::tr("Loaded") : i18n::tr("Cached"));
      ImGui::PopStyleColor();
      panel::wrapped(engine.name.c_str());
      ImGui::PushStyleColor(ImGuiCol_Text, kNote);
      panel::wrapped(engine.detail.c_str());
      ImGui::PopStyleColor();
      ImGui::Spacing();
    }
    ImGui::TreePop();
  }
}

void stages_note(const SettingsChoices& choices, std::vector<Input>& actions) {
  if (choices.stage_note.empty() && choices.motion_note.empty()) return;
  ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
  if (!choices.stage_note.empty()) panel::wrapped(choices.stage_note);
  if (!choices.motion_note.empty()) panel::wrapped(choices.motion_note);
  ImGui::PopStyleColor();
  if (ImGui::Button(i18n::tr("Retry"))) actions.emplace_back(RetryStages{});
  panel::tip(i18n::tr("Retry"), i18n::tr("Prepares these engines again."), panel::Apply::Reload);
}

}  // namespace ajp::app::play
