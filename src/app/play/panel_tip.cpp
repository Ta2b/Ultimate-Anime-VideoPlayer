#include "app/play/panel_tip.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include <imgui.h>

#include "app/play/french_text.h"
#include "app/play/i18n.h"
#include "app/play/panel_state.h"
#include "core/text.h"

namespace ajp::app::play::panel {
namespace {

enum class Field : std::uint8_t { Speed, Quality, Recommended };
struct Marker {
  std::string_view text;
  Field field;
};
// Les repères des deux langues : les textes anglais et leurs traductions (i18n_fr_*.cpp) les écrivent ainsi.
constexpr std::array<Marker, 6> kMarkers{{
    {"Speed:", Field::Speed},
    {"Quality impact:", Field::Quality},
    {"Recommended:", Field::Recommended},
    {"Vitesse :", Field::Speed},
    {"Impact sur la qualité :", Field::Quality},  // i18n-ok : repère des traductions
    {"Recommandé :", Field::Recommended},         // i18n-ok : repère des traductions
}};

// La valeur d'un champ, sans le point final : « environ −3 % ».
std::string value_of(std::string_view s) {
  s = core::trim(s);
  if (s.ends_with('.')) s.remove_suffix(1);
  return std::string(core::trim(s));
}

// Le prochain repère à partir de `from` : sa place et lequel ; npos s'il n'y en a plus.
std::pair<std::size_t, const Marker*> next_marker(std::string_view text, std::size_t from) {
  std::size_t best = std::string_view::npos;
  const Marker* found = nullptr;
  for (const Marker& marker : kMarkers) {
    const std::size_t at = text.find(marker.text, from);
    if (at < best) {
      best = at;
      found = &marker;
    }
  }
  return {best, found};
}

ImVec4 with_alpha(ImVec4 c, float a) {
  c.w = a;
  return c;
}

constexpr ImVec4 kTipKey{0.62F, 0.64F, 0.70F, 1.0F};
constexpr ImVec4 kTipBlue{0.45F, 0.70F, 1.0F, 1.0F};
constexpr ImVec4 kTipAmber{1.0F, 0.76F, 0.40F, 1.0F};

// Un texte en gras : la police grasse si l'atlas l'a, sinon le texte tracé deux fois à un pixel d'écart.
void bold(const std::string& text, ImVec4 color) {
  const State* state = static_cast<const State*>(ImGui::GetIO().UserData);
  ImGui::PushStyleColor(ImGuiCol_Text, color);
  if (state != nullptr && state->bold != nullptr) {
    ImGui::PushFont(state->bold, 0.0F);
    ImGui::TextWrapped("%s", text.c_str());
    ImGui::PopFont();
  } else {
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::TextWrapped("%s", text.c_str());
    const float wrap = ImGui::GetContentRegionAvail().x;
    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(at.x + 1.0F, at.y),
                                        ImGui::GetColorU32(color), text.c_str(), nullptr, wrap);
  }
  ImGui::PopStyleColor();
}

// « Vitesse : environ −3 % » : le nom du champ en gris, sa valeur en clair.
void field_line(const char* key, const std::string& value) {
  if (value.empty()) return;
  ImGui::PushStyleColor(ImGuiCol_Text, kTipKey);
  ImGui::TextUnformatted(french_spaces(key).c_str());
  ImGui::PopStyleColor();
  ImGui::SameLine(0.0F, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::TextWrapped("%s", french_spaces(value).c_str());
}

}  // namespace

TipParts split_tip(std::string_view text) {
  TipParts parts;
  auto [at, marker] = next_marker(text, 0);
  parts.text = std::string(core::trim(text.substr(0, at)));
  while (marker != nullptr) {
    const std::size_t start = at + marker->text.size();
    const auto [next, next_mark] = next_marker(text, start);
    std::string_view value =
        text.substr(start, next == std::string_view::npos ? text.size() - start : next - start);
    switch (marker->field) {
      case Field::Speed:
        parts.speed = value_of(value);
        break;
      case Field::Quality:
        parts.quality = value_of(value);
        break;
      case Field::Recommended: {
        // La recommandation s'arrête à sa première phrase ; la suite (« RIFE seulement. ») va dessous.
        value = core::trim(value);
        const std::size_t end = value.find(". ");
        parts.recommended = value_of(value.substr(0, end));
        if (end != std::string_view::npos) parts.after = std::string(core::trim(value.substr(end + 1)));
        break;
      }
    }
    at = next;
    marker = next_mark;
  }
  return parts;
}

void tip(const char* title, std::string_view body, Apply apply) {
  if (!ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled | ImGuiHoveredFlags_DelayNormal)) return;
  const TipParts parts = split_tip(body);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0F, 12.0F));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0F, 4.0F));
  ImGui::BeginTooltip();
  ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0F);
  const auto gap = [] { ImGui::Dummy(ImVec2(0.0F, ImGui::GetFontSize() * 0.35F)); };
  bool first = true;
  if (title != nullptr && *title != '\0') {
    // Le nom du réglage sans l'identifiant ImGui (« Annuler##moteur »).
    std::string name(title);
    if (const std::size_t hash = name.find("##"); hash != std::string::npos) name.resize(hash);
    bold(name, ImGui::GetStyleColorVec4(ImGuiCol_Text));
    first = false;
  }
  if (!parts.text.empty()) {
    if (!first) gap();
    ImGui::TextWrapped("%s", french_spaces(parts.text).c_str());
    first = false;
  }
  if (!parts.speed.empty() || !parts.quality.empty()) {
    if (!first) gap();
    field_line(i18n::tr("Speed:"), parts.speed);
    field_line(i18n::tr("Quality impact:"), parts.quality);
    first = false;
  }
  if (!parts.after.empty()) {
    if (!first) gap();
    ImGui::PushStyleColor(ImGuiCol_Text, with_alpha(ImGui::GetStyleColorVec4(ImGuiCol_Text), 0.8F));
    ImGui::TextWrapped("%s", french_spaces(parts.after).c_str());
    ImGui::PopStyleColor();
    first = false;
  }
  if (!parts.recommended.empty()) {
    if (!first) gap();
    bold(french_spaces(i18n::tr_format("Recommended: {}", parts.recommended)), kTipBlue);
    first = false;
  }
  if (apply != Apply::Live) {
    if (!first) gap();
    ImGui::PushStyleColor(ImGuiCol_Text, kTipAmber);
    ImGui::TextWrapped("%s",
                       french_spaces(apply == Apply::Reload ? i18n::tr("Applies by reloading the video.")
                                                            : i18n::tr("Applies by restarting the player."))
                           .c_str());
    ImGui::PopStyleColor();
  }
  ImGui::PopTextWrapPos();
  ImGui::EndTooltip();
  ImGui::PopStyleVar(2);
}

void tip(const char* body) {
  tip(nullptr, body == nullptr ? std::string_view() : std::string_view(body));
}

}  // namespace ajp::app::play::panel
