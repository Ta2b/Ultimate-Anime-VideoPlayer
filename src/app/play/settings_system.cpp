#include "app/play/settings_system.h"

#include <expected>

#include <imgui.h>

#include "app/play/panel_style.h"

namespace ajp::app::play {

using panel::kWarn;
using panel::note;
using panel::section;

void system_tab(const PlayerPlace& place, SystemNote& last) {
  section("Lecteur par défaut");
  ImGui::TextWrapped(
      "Les vidéos (mkv, mp4, webm, avi, mov…) s'ouvrent dans AJP depuis le gestionnaire de fichiers.");
#ifdef __APPLE__
  static_cast<void>(last);
  note(register_player(place).value_or(std::string()));
#else
  if (last.pending && last.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = last.pending->get();
    last = done ? SystemNote{.text = *done} : SystemNote{.text = done.error().detail, .refused = true};
  }
  const bool busy = last.pending != nullptr;
  const auto start = [&](std::expected<std::string, Refusal> (*action)(const PlayerPlace&)) {
    last.pending = std::make_shared<std::future<std::expected<std::string, Refusal> > >(
        std::async(std::launch::async, action, place));
    last.text.clear();
    last.refused = false;
  };
  ImGui::BeginDisabled(busy);
  if (ImGui::Button("Définir par défaut")) start(&choose_as_default);
  ImGui::SameLine();
  if (ImGui::Button("Ajouter à « Ouvrir avec »")) start(&register_player);
  ImGui::SameLine();
  if (ImGui::Button("Retirer##inscription")) start(&unregister_player);
  ImGui::EndDisabled();
  if (busy) {
    note("En cours…");
  } else if (last.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    ImGui::TextWrapped("%s", last.text.c_str());
    ImGui::PopStyleColor();
  } else if (!last.text.empty()) {
    note(last.text);
  }
#endif
}

}  // namespace ajp::app::play
