#include "app/play/controls_draw.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numbers>
#include <string_view>

#include <imgui.h>

namespace ajp::app::play {
namespace {

// Géométrie de la barre, en pixels à l'échelle 1.
constexpr float kPad = 18.0F;
constexpr float kRow = 46.0F;
constexpr float kBarZone = 22.0F;
constexpr float kPanelExtra = 8.0F;
constexpr float kTitleSize = 17.0F * 1.1F;  // taille du titre (police de l'interface × 1,1)

// Couleurs sRGB : blanc pour le texte et les icônes, un bleu clair pour la progression.
constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};

ImU32 rgba(float r, float g, float b, float a) {
  return ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, std::clamp(a, 0.0F, 1.0F)));
}
ImU32 white(float a) {
  return rgba(1.0F, 1.0F, 1.0F, a);
}
ImU32 black(float a) {
  return rgba(0.0F, 0.0F, 0.0F, a);
}
ImU32 accent(float a) {
  return rgba(kAccent.x, kAccent.y, kAccent.z, a);
}

// Ce que les dessins partagent : la liste, l'échelle, l'opacité de la barre.
struct Pen {
  ImDrawList* list = nullptr;
  float scale = 1.0F;
  float opacity = 1.0F;
};

ImVec2 at(ImVec2 c, float x, float y) {
  return {c.x + x, c.y + y};
}

void icon_play(const Pen& pen, ImVec2 c, float size) {
  const float h = size * 0.5F;
  pen.list->AddTriangleFilled(at(c, -h * 0.62F, -h * 0.78F), at(c, -h * 0.62F, h * 0.78F),
                              at(c, h * 0.84F, 0.0F), white(pen.opacity));
}

void icon_pause(const Pen& pen, ImVec2 c, float size) {
  const float h = size * 0.5F;
  const float w = h * 0.38F;
  const float rounding = w * 0.25F;
  pen.list->AddRectFilled(at(c, -h * 0.62F, -h * 0.76F), at(c, -h * 0.62F + w, h * 0.76F), white(pen.opacity),
                          rounding);
  pen.list->AddRectFilled(at(c, h * 0.62F - w, -h * 0.76F), at(c, h * 0.62F, h * 0.76F), white(pen.opacity),
                          rounding);
}

void icon_volume(const Pen& pen, ImVec2 c, float size, float volume, bool muted) {
  const float h = size * 0.5F;
  const ImU32 color = white(pen.opacity);
  pen.list->AddRectFilled(at(c, -h * 0.92F, -h * 0.3F), at(c, -h * 0.42F, h * 0.3F), color);
  pen.list->AddQuadFilled(at(c, -h * 0.46F, -h * 0.3F), at(c, h * 0.04F, -h * 0.78F),
                          at(c, h * 0.04F, h * 0.78F), at(c, -h * 0.46F, h * 0.3F), color);
  const float stroke = std::max(1.5F, size * 0.08F);
  if (muted || volume <= 0.0F) {
    pen.list->AddLine(at(c, h * 0.32F, -h * 0.34F), at(c, h * 0.92F, h * 0.34F), color, stroke);
    pen.list->AddLine(at(c, h * 0.32F, h * 0.34F), at(c, h * 0.92F, -h * 0.34F), color, stroke);
    return;
  }
  const ImVec2 origin = at(c, h * 0.02F, 0.0F);
  constexpr float kArc = 0.85F;  // demi-angle des ondes, en radians
  pen.list->PathArcTo(origin, h * 0.42F, -kArc, kArc);
  pen.list->PathStroke(color, ImDrawFlags_None, stroke);
  if (volume > 0.5F) {
    pen.list->PathArcTo(origin, h * 0.82F, -kArc, kArc);
    pen.list->PathStroke(color, ImDrawFlags_None, stroke);
  }
}

// Quatre équerres : tournées vers l'extérieur pour entrer en plein écran, vers le centre pour en
// sortir.
void icon_fullscreen(const Pen& pen, ImVec2 c, float size, bool leave) {
  const float h = size * 0.42F;
  const float arm = h * 0.6F;
  const float stroke = std::max(1.5F, size * 0.09F);
  constexpr std::array<std::array<float, 2>, 4> kCorners{
      {{-1.0F, -1.0F}, {1.0F, -1.0F}, {1.0F, 1.0F}, {-1.0F, 1.0F}}};
  for (const auto& [sx, sy] : kCorners) {
    std::array<ImVec2, 3> points{};
    if (leave) {
      const ImVec2 elbow = at(c, sx * (h - arm), sy * (h - arm));
      points = {at(elbow, sx * arm, 0.0F), elbow, at(elbow, 0.0F, sy * arm)};
    } else {
      const ImVec2 elbow = at(c, sx * h, sy * h);
      points = {at(elbow, 0.0F, -sy * arm), elbow, at(elbow, -sx * arm, 0.0F)};
    }
    pen.list->AddPolyline(points.data(), static_cast<int>(points.size()), white(pen.opacity),
                          ImDrawFlags_None, stroke);
  }
}

void icon_gear(const Pen& pen, ImVec2 c, float size) {
  const float r = size * 0.34F;
  const ImU32 color = white(pen.opacity);
  pen.list->AddCircle(c, r * 0.68F, color, 0, r * 0.46F);
  constexpr int kTeeth = 8;
  for (int i = 0; i < kTeeth; ++i) {
    const float angle = static_cast<float>(i) * 2.0F * std::numbers::pi_v<float> / kTeeth;
    const ImVec2 d{std::cos(angle), std::sin(angle)};
    const ImVec2 n{-d.y, d.x};
    const float half = r * 0.22F;
    const float r0 = r * 0.8F;
    const float r1 = r * 1.3F;
    pen.list->AddQuadFilled(at(c, d.x * r0 + n.x * half, d.y * r0 + n.y * half),
                            at(c, d.x * r1 + n.x * half * 0.8F, d.y * r1 + n.y * half * 0.8F),
                            at(c, d.x * r1 - n.x * half * 0.8F, d.y * r1 - n.y * half * 0.8F),
                            at(c, d.x * r0 - n.x * half, d.y * r0 - n.y * half), color);
  }
}

// Bouton rond invisible : un disque clair au survol ; true au clic.
bool round_button(const Pen& pen, const char* id, ImVec2 center, float size) {
  ImGui::SetCursorScreenPos({center.x - size * 0.5F, center.y - size * 0.5F});
  const bool pressed = ImGui::InvisibleButton(id, {size, size});
  if (ImGui::IsItemHovered() || ImGui::IsItemActive())
    pen.list->AddCircleFilled(center, size * 0.44F, white(0.16F * pen.opacity));
  return pressed;
}

void tooltip(std::string_view text) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
    ImGui::SetTooltip("%.*s", static_cast<int>(text.size()), text.data());
}

// Bulle de texte centrée sur x, posée au-dessus de bottom, gardée entre min_x et max_x. Au
// premier plan : elle dépasse le haut de la fenêtre de la barre.
void bubble(const Pen& pen, float x, float bottom, std::string_view text, float min_x, float max_x) {
  ImDrawList* list = ImGui::GetForegroundDrawList();
  const ImVec2 size = ImGui::CalcTextSize(text.data(), text.data() + text.size());
  const ImVec2 pad{9.0F * pen.scale, 4.0F * pen.scale};
  const float width = size.x + 2.0F * pad.x;
  const float x0 = std::clamp(x - width * 0.5F, min_x, std::max(min_x, max_x - width));
  const ImVec2 p0{x0, bottom - size.y - 2.0F * pad.y};
  list->AddRectFilled(p0, {x0 + width, bottom}, black(0.82F * pen.opacity), 6.0F * pen.scale);
  list->AddText({p0.x + pad.x, p0.y + pad.y}, white(pen.opacity), text.data(), text.data() + text.size());
}

void draw_title(const Pen& pen, std::string_view title, float width) {
  if (title.empty()) return;
  const float pad = 18.0F * pen.scale;
  const float size = ImGui::GetFontSize() * 1.1F;
  pen.list->PushClipRect({pad, 0.0F}, {width - pad, size * 3.0F});
  pen.list->AddText(ImGui::GetFont(), size, {pad, pad}, white(0.94F * pen.opacity), title.data(),
                    title.data() + title.size());
  pen.list->PopClipRect();
}

void draw_toast(const Controls& controls, float width, float scale, std::int64_t now_us) {
  const auto text = controls.toast(now_us);
  if (!text) return;
  const float opacity = controls.toast_opacity(now_us);
  ImDrawList* list = ImGui::GetForegroundDrawList();
  const ImVec2 size = ImGui::CalcTextSize(text->data(), text->data() + text->size());
  const ImVec2 pad{16.0F * scale, 8.0F * scale};
  const ImVec2 p0{width - size.x - 2.0F * pad.x - kPad * scale, kPad * scale};  // en haut à droite
  list->AddRectFilled(p0, {p0.x + size.x + 2.0F * pad.x, p0.y + size.y + 2.0F * pad.y},
                      black(0.72F * opacity), (size.y + 2.0F * pad.y) * 0.5F);
  list->AddText({p0.x + pad.x, p0.y + pad.y}, white(opacity), text->data(), text->data() + text->size());
}

}  // namespace

float controls_height(float scale) {
  return (kPad + kRow + kBarZone + kPanelExtra) * scale;
}

float title_height(float scale) {
  return (kPad * 1.5F + kTitleSize) * scale;
}

bool draw_controls(Controls& controls, const ControlsView& view, const UiLook& look, std::int64_t now_us,
                   std::vector<Input>& actions) {
  const ImGuiIO& io = ImGui::GetIO();
  const float width = io.DisplaySize.x;
  const float height = io.DisplaySize.y;
  const float s = look.scale;
  const float opacity = controls.opacity(now_us, view.paused);
  bool held = false;

  if (opacity > 0.0F) {
    const float pad = kPad * s;
    const float row = kRow * s;
    const float bar_zone = kBarZone * s;
    const float panel = controls_height(s);

    // Dégradés sous la barre et sous le titre : le texte reste lisible sur une image claire.
    ImDrawList* back = ImGui::GetBackgroundDrawList();
    const float shade = panel * 2.0F;
    back->AddRectFilledMultiColor({0.0F, height - shade}, {width, height}, black(0.0F), black(0.0F),
                                  black(0.74F * opacity), black(0.74F * opacity));
    back->AddRectFilledMultiColor({0.0F, 0.0F}, {width, 96.0F * s}, black(0.55F * opacity),
                                  black(0.55F * opacity), black(0.0F), black(0.0F));
    draw_title(Pen{back, s, opacity}, look.title, width);

    ImGui::SetNextWindowPos({0.0F, height - panel});
    ImGui::SetNextWindowSize({width, panel});
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, opacity);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.0F, 0.0F});
    constexpr ImGuiWindowFlags kPanel =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::Begin("##controls", nullptr, kPanel);
    const Pen pen{ImGui::GetWindowDrawList(), s, opacity};

    // Barre de progression : fine, plus épaisse au survol, bulle du temps sous la souris.
    const float x0 = pad;
    const float x1 = width - pad;
    const float bar_y = height - pad - row - bar_zone * 0.5F;
    ImGui::SetCursorScreenPos({x0, bar_y - bar_zone * 0.5F});
    ImGui::InvisibleButton("##seek", {x1 - x0, bar_zone});
    const bool bar_hovered = ImGui::IsItemHovered();
    const bool bar_active = ImGui::IsItemActive();
    const double pointer = std::clamp(static_cast<double>((io.MousePos.x - x0) / (x1 - x0)), 0.0, 1.0);
    const bool seekable = view.duration_us > 0;
    if (seekable) {
      if (ImGui::IsItemActivated()) {
        controls.press(pointer);
      } else if (bar_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0F * s)) {
        if (auto seek = controls.drag_to(pointer, view.duration_us, now_us))
          actions.emplace_back(Command{*seek});
      }
      if (ImGui::IsItemDeactivated() && controls.dragging())
        actions.emplace_back(Command{controls.release(pointer, view.duration_us)});
    }
    const double played = controls.drag_fraction().value_or(fraction_of(view.position_us, view.duration_us));
    const bool lit = (bar_hovered || bar_active) && seekable;
    const float thick = (lit ? 6.0F : 3.5F) * s;
    const float xp = x0 + static_cast<float>(played) * (x1 - x0);
    pen.list->AddRectFilled({x0, bar_y - thick * 0.5F}, {x1, bar_y + thick * 0.5F}, white(0.28F * opacity),
                            thick * 0.5F);
    if (lit) {
      const float xh = x0 + static_cast<float>(pointer) * (x1 - x0);
      if (xh > xp)
        pen.list->AddRectFilled({xp, bar_y - thick * 0.5F}, {xh, bar_y + thick * 0.5F}, white(0.3F * opacity),
                                thick * 0.5F);
    }
    pen.list->AddRectFilled({x0, bar_y - thick * 0.5F}, {std::max(xp, x0 + thick), bar_y + thick * 0.5F},
                            accent(opacity), thick * 0.5F);
    if (lit) {
      pen.list->AddCircleFilled({xp, bar_y}, 7.5F * s, accent(opacity));
      const double shown = controls.dragging() ? played : pointer;
      bubble(pen, x0 + static_cast<float>(shown) * (x1 - x0), bar_y - 12.0F * s,
             format_clock(time_at(shown, view.duration_us), view.duration_us), x0, x1);
    }

    // Rangée de boutons : lecture, volume, temps à gauche ; réglages et plein écran à droite.
    const float cy = height - pad - row * 0.5F;
    const float icon = 22.0F * s;
    float x = pad + row * 0.5F;
    if (round_button(pen, "##play", {x, cy}, row)) actions.emplace_back(Command{cmd::TogglePause{}});
    tooltip(view.paused ? "Lecture (espace)" : "Pause (espace)");
    if (view.paused) {
      icon_play(pen, {x, cy}, icon);
    } else {
      icon_pause(pen, {x, cy}, icon);
    }
    x += row;
    if (round_button(pen, "##mute", {x, cy}, row)) actions.emplace_back(ToggleMute{});
    tooltip(controls.muted() ? "Rétablir le son (m)" : "Couper le son (m)");
    icon_volume(pen, {x, cy}, icon, controls.volume(), controls.muted());

    const float vx0 = x + row * 0.5F + 2.0F * s;
    const float vw = 100.0F * s;
    ImGui::SetCursorScreenPos({vx0, cy - row * 0.5F});
    ImGui::InvisibleButton("##volume", {vw, row});
    if (ImGui::IsItemActive()) {
      const float wanted = std::clamp((io.MousePos.x - vx0) / vw, 0.0F, 1.0F) * kMaxVolume;
      if (std::abs(wanted - controls.volume()) > 0.004F || controls.muted())
        actions.emplace_back(Command{controls.set_volume(wanted)});
    }
    const bool volume_lit = ImGui::IsItemHovered() || ImGui::IsItemActive();
    const float level = controls.muted() ? 0.0F : controls.volume() / kMaxVolume;
    const float track = (volume_lit ? 4.0F : 3.0F) * s;
    pen.list->AddRectFilled({vx0, cy - track * 0.5F}, {vx0 + vw, cy + track * 0.5F}, white(0.3F * opacity),
                            track * 0.5F);
    pen.list->AddRectFilled({vx0, cy - track * 0.5F}, {vx0 + vw * level, cy + track * 0.5F}, white(opacity),
                            track * 0.5F);
    const float unity = vx0 + vw / kMaxVolume;  // repère des 100 %
    pen.list->AddLine({unity, cy - 5.0F * s}, {unity, cy + 5.0F * s}, white(0.45F * opacity), 1.0F * s);
    pen.list->AddCircleFilled({vx0 + vw * level, cy}, (volume_lit ? 7.0F : 5.5F) * s, white(opacity));
    if (volume_lit) {
      const std::string percent = controls.muted()
                                      ? std::string("Muet")
                                      : std::format("{} %", std::lround(controls.volume() * 100.0F));
      bubble(pen, vx0 + vw * level, cy - 12.0F * s, percent, x0, x1);
    }

    const std::string clock = std::format("{} / {}", format_clock(view.position_us, view.duration_us),
                                          format_clock(view.duration_us, view.duration_us));
    const ImVec2 clock_size = ImGui::CalcTextSize(clock.c_str());
    pen.list->AddText({vx0 + vw + 18.0F * s, cy - clock_size.y * 0.5F}, white(0.95F * opacity),
                      clock.c_str());

    float rx = width - pad - row * 0.5F;
    if (round_button(pen, "##fullscreen", {rx, cy}, row)) actions.emplace_back(ToggleFullscreen{});
    tooltip(view.fullscreen ? "Quitter le plein écran (f, Échap)" : "Plein écran (f)");
    icon_fullscreen(pen, {rx, cy}, icon, view.fullscreen);
    rx -= row;
    if (round_button(pen, "##settings", {rx, cy}, row)) ImGui::OpenPopup("##settings_menu");
    tooltip("Réglages");
    icon_gear(pen, {rx, cy}, icon);
    ImGui::SetNextWindowPos({rx + row * 0.5F, height - panel - 4.0F * s}, ImGuiCond_Always, {1.0F, 1.0F});
    if (ImGui::BeginPopup("##settings_menu")) {
      if (ImGui::MenuItem("Statistiques", "Tab", view.stats)) actions.emplace_back(Command{cmd::ToggleOsd{}});
      ImGui::EndPopup();
    }

    held = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) || bar_active ||
           controls.dragging() || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
    ImGui::End();
    ImGui::PopStyleVar(2);
  }
  controls.set_held(held);
  draw_toast(controls, width, s, now_us);
  return opacity > 0.0F || controls.toast_opacity(now_us) > 0.0F;
}

}  // namespace ajp::app::play
