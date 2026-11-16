#include "app/play/panel_size.h"

#include <algorithm>
#include <cmath>

namespace ajp::app::play {

namespace {

float share_of(float size, float room) {
  return room > 0.0F ? std::clamp(size / room, 0.0F, 1.0F) : 1.0F;
}

}  // namespace

PanelArea PanelSize::target(PanelArea area, float default_width) {
  if (width_share_ <= 0.0F) width_share_ = share_of(std::min(default_width, area.width), area.width);
  if (height_share_ <= 0.0F) height_share_ = 1.0F;
  area_changed_ = area != area_;
  area_ = area;
  return {width_share_ * area.width, height_share_ * area.height};
}

void PanelSize::observe(PanelArea size) {
  // Une taille rabotée par le minimum du panneau, ou imposée parce que la place a changé, ne compte pas.
  const bool dragged =
      std::abs(size.width - last_.width) > 0.5F || std::abs(size.height - last_.height) > 0.5F;
  if (!area_changed_ && dragged && last_.width > 0.0F) {
    width_share_ = share_of(size.width, area_.width);
    height_share_ = share_of(size.height, area_.height);
  }
  last_ = size;
}

}  // namespace ajp::app::play
