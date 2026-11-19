#include "kernels/ref/motion_doubt.h"

#include <cstddef>
#include <format>

#include "kernels/spec/motion_doubt.h"

namespace kirari::kernels::ref {

Result<Motion> motion_doubt(const Motion& motion, const DoubtGrid& grid, float fade) {
  if (auto ok = check_motion(motion); !ok) return std::unexpected(ok.error());
  if (grid.cols < 1 || grid.rows < 1 || grid.cols > kDoubtMaxCells || grid.rows > kDoubtMaxCells) {
    return fail(ErrorCode::DimensionOutOfRange,
                std::format("grille de doute : {}x{} hors de [1, {}]", grid.cols, grid.rows, kDoubtMaxCells));
  }
  if (grid.weights.size() != static_cast<std::size_t>(grid.cols) * static_cast<std::size_t>(grid.rows)) {
    return fail(ErrorCode::SizeMismatch, std::format("grille de doute : {} poids pour {}x{}",
                                                     grid.weights.size(), grid.cols, grid.rows));
  }
  if (!(fade >= 0.0F && fade <= 1.0F)) {
    return fail(ErrorCode::ParameterOutOfRange, std::format("masque du fondu {} hors de [0, 1]", fade));
  }
  Motion out = motion;
  const Window& area = motion.area;
  for (int y = area.y; y < area.y + area.height; ++y) {
    const float v = doubt_coord(y - area.y, grid.rows, area.height);
    for (int x = area.x; x < area.x + area.width; ++x) {
      const float u = doubt_coord(x - area.x, grid.cols, area.width);
      const float w = doubt_at(grid.weights.data(), grid.cols, grid.rows, u, v);
      for (PlaneF& p : out.flow) p.at(x, y) = doubt_flow(p.at(x, y), w);
      out.mask.at(x, y) = doubt_mask(out.mask.at(x, y), fade, w);
    }
  }
  return out;
}

}  // namespace kirari::kernels::ref
