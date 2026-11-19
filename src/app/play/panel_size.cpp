#include "app/play/panel_size.h"

#include <algorithm>
#include <cmath>

namespace kirari::app::play {

namespace {

float share_of(float size, float room) {
  return room > 0.0F ? std::clamp(size / room, 0.0F, 1.0F) : 1.0F;
}

PanelRect shares_of(PanelRect rect, PanelArea area) {
  return {.x = share_of(rect.x, area.width),
          .y = share_of(rect.y, area.height),
          .width = share_of(rect.width, area.width),
          .height = share_of(rect.height, area.height)};
}

bool moved(PanelRect a, PanelRect b) {
  return std::abs(a.x - b.x) > 0.5F || std::abs(a.y - b.y) > 0.5F || std::abs(a.width - b.width) > 0.5F ||
         std::abs(a.height - b.height) > 0.5F;
}

}  // namespace

PanelRect fixed_panel(PanelArea area, float widest, float narrowest, float keep_left) {
  const float width =
      std::min(std::clamp(area.width - keep_left, std::min(narrowest, widest), widest), area.width);
  return {.x = area.width - width, .y = 0.0F, .width = width, .height = area.height};
}

PanelRect PanelSize::target(PanelArea area, PanelRect first) {
  if (!placed_) {
    share_ = shares_of(first, area);
    placed_ = true;
  }
  area_changed_ = area != area_;
  area_ = area;
  const float width = share_.width * area.width;
  const float height = share_.height * area.height;
  // Le panneau reste dans la place : un bord poussé dehors revient au bord.
  return {.x = std::clamp(share_.x * area.width, 0.0F, std::max(0.0F, area.width - width)),
          .y = std::clamp(share_.y * area.height, 0.0F, std::max(0.0F, area.height - height)),
          .width = width,
          .height = height};
}

void PanelSize::observe(PanelRect rect) {
  // Un rectangle raboté par le minimum du panneau, ou imposé parce que la place a changé, ne compte pas.
  if (!area_changed_ && last_.width > 0.0F && moved(rect, last_)) share_ = shares_of(rect, area_);
  last_ = rect;
}

}  // namespace kirari::app::play
