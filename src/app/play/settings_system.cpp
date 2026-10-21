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
  const auto show = [&](const std::expected<std::string, Refusal>& done) {
    last = done ? SystemNote{.text = *done} : SystemNote{.text = done.error().detail, .refused = true};
  };
  if (ImGui::Button("Définir par défaut")) show(choose_as_default(place));
  ImGui::SameLine();
  if (ImGui::Button("Ajouter à « Ouvrir avec »")) show(register_player(place));
  ImGui::SameLine();
  if (ImGui::Button("Retirer##inscription")) show(unregister_player(place));
  if (last.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    ImGui::TextWrapped("%s", last.text.c_str());
    ImGui::PopStyleColor();
  } else if (!last.text.empty()) {
    note(last.text);
  }
#endif
}

}  // namespace ajp::app::play
