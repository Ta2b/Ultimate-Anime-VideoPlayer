#include "core/moving_area.h"

#include <algorithm>

namespace ajp::core {

MovingArea moving_area(std::span<const std::uint32_t> deviation, std::uint32_t blocks_x,
                       std::uint32_t limit) {
  MovingArea out;
  if (blocks_x == 0) return out;
  out.total = static_cast<std::uint32_t>(deviation.size());
  out.x0 = blocks_x;
  out.y0 = UINT32_MAX;
  for (std::size_t i = 0; i < deviation.size(); ++i) {
    if (deviation[i] <= limit) continue;
    const auto x = static_cast<std::uint32_t>(i % blocks_x);
    const auto y = static_cast<std::uint32_t>(i / blocks_x);
    ++out.blocks;
    out.x0 = std::min(out.x0, x);
    out.x1 = std::max(out.x1, x + 1);
    out.y0 = std::min(out.y0, y);
    out.y1 = std::max(out.y1, y + 1);
  }
  if (out.blocks == 0) out.x0 = out.y0 = 0;
  return out;
}

}  // namespace ajp::core
