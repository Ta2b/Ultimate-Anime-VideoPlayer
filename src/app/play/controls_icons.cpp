#include "app/play/controls_icons.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace ajp::app::play::icons {
namespace {

constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};

ImU32 rgba(float r, float g, float b, float a) {
  return ImGui::ColorConvertFloat4ToU32(ImVec4(r, g, b, std::clamp(a, 0.0F, 1.0F)));
}

ImVec2 at(ImVec2 c, float x, float y) {
  return {c.x + x, c.y + y};
}

}  // namespace

ImU32 white(float a) {
  return rgba(1.0F, 1.0F, 1.0F, a);
}
ImU32 black(float a) {
  return rgba(0.0F, 0.0F, 0.0F, a);
}
ImU32 accent(float a) {
  return rgba(kAccent.x, kAccent.y, kAccent.z, a);
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

void icon_subtitles(const Pen& pen, ImVec2 c, float size, bool on) {
  const float w = size * 0.48F;
  const float h = size * 0.36F;
  const float stroke = std::max(1.5F, size * 0.08F);
  const ImU32 color = white(pen.opacity * (on ? 1.0F : 0.55F));
  pen.list->AddRect(at(c, -w, -h), at(c, w, h), color, size * 0.12F, ImDrawFlags_None, stroke);
  pen.list->AddLine(at(c, -w * 0.55F, h * 0.05F), at(c, w * 0.15F, h * 0.05F), color, stroke);
  pen.list->AddLine(at(c, w * 0.32F, h * 0.05F), at(c, w * 0.55F, h * 0.05F), color, stroke);
  pen.list->AddLine(at(c, -w * 0.55F, h * 0.5F), at(c, -w * 0.2F, h * 0.5F), color, stroke);
  pen.list->AddLine(at(c, -w * 0.02F, h * 0.5F), at(c, w * 0.55F, h * 0.5F), color, stroke);
  if (on) {  // trait sous l'icône quand une piste est choisie
    pen.list->AddLine(at(c, -w * 0.6F, h + size * 0.2F), at(c, w * 0.6F, h + size * 0.2F),
                      accent(pen.opacity), stroke);
  }
}

void icon_audio_track(const Pen& pen, ImVec2 c, float size) {
  const float h = size * 0.5F;
  const ImU32 color = white(pen.opacity);
  const float stroke = std::max(1.5F, size * 0.09F);
  pen.list->AddCircleFilled(at(c, -h * 0.32F, h * 0.5F), h * 0.3F, color);
  pen.list->AddLine(at(c, -h * 0.05F, h * 0.5F), at(c, -h * 0.05F, -h * 0.75F), color, stroke);
  pen.list->AddQuadFilled(at(c, -h * 0.05F - stroke * 0.5F, -h * 0.8F), at(c, h * 0.55F, -h * 0.45F),
                          at(c, h * 0.55F, -h * 0.15F), at(c, -h * 0.05F - stroke * 0.5F, -h * 0.5F), color);
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

void icon_chapter(const Pen& pen, ImVec2 c, float size, int direction) {
  const float h = size * 0.5F;
  const float d = direction < 0 ? -1.0F : 1.0F;
  const ImU32 color = white(pen.opacity);
  pen.list->AddTriangleFilled(at(c, -d * h * 0.62F, -h * 0.66F), at(c, -d * h * 0.62F, h * 0.66F),
                              at(c, d * h * 0.42F, 0.0F), color);
  const float bar = std::max(2.0F, size * 0.12F);
  const float x = d * h * 0.5F;
  pen.list->AddRectFilled(at(c, std::min(x, x + d * bar), -h * 0.66F),
                          at(c, std::max(x, x + d * bar), h * 0.66F), color, bar * 0.3F);
}

}  // namespace ajp::app::play::icons
