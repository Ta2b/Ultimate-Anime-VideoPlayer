// Les couleurs, notes et titres de section communs au menu des paramètres et à ses onglets.
#pragma once

#include <string>

#include <imgui.h>

namespace ajp::app::play::panel {

inline constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};
inline constexpr ImVec4 kNote{0.66F, 0.68F, 0.74F, 1.0F};
inline constexpr ImVec4 kWarn{1.0F, 0.78F, 0.42F, 1.0F};

inline void note(const char* text) {
  ImGui::PushStyleColor(ImGuiCol_Text, kNote);
  ImGui::TextWrapped("%s", text);
  ImGui::PopStyleColor();
}
inline void note(const std::string& text) {
  note(text.c_str());
}

inline void section(const char* title) {
  ImGui::Spacing();
  ImGui::PushStyleColor(ImGuiCol_Text, kAccent);
  ImGui::TextUnformatted(title);
  ImGui::PopStyleColor();
  ImGui::Separator();
}

}  // namespace ajp::app::play::panel
