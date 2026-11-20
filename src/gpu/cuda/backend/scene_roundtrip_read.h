// Format et lecture du témoin aller-retour NVOF, indépendants des files et du pilote CUDA.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include "kernels/spec/scene_flow.h"

namespace kirari::gpu::cuda {

struct RoundtripShape {
  int width = 0;
  int height = 0;
  constexpr int cols() const { return kernels::scene_flow_cells(width); }
  constexpr int rows() const { return kernels::scene_flow_cells(height); }
  constexpr std::size_t row_words() const { return 2U * static_cast<std::size_t>(cols()); }
  constexpr std::size_t words() const { return row_words() * static_cast<std::size_t>(rows()); }
  constexpr bool valid() const {
    return width >= kernels::kSceneFlowMinSide && height >= kernels::kSceneFlowMinSide;
  }
};

// Un seul retour en vol. Le titulaire de done peut relire les deux champs, puis rendre les tampons.
class RoundtripTicket {
 public:
  bool accepts(std::uint64_t ticket) const { return pending_ == 0 && ticket > last_; }
  void start(std::uint64_t ticket) { last_ = pending_ = ticket; }
  std::uint64_t pending() const { return pending_; }
  std::uint64_t take() { return std::exchange(pending_, 0); }

 private:
  std::uint64_t last_ = 0;
  std::uint64_t pending_ = 0;
};

struct RoundtripStats {
  std::size_t checked = 0;
  std::size_t outside = 0;
  std::size_t open = 0;
};

inline float roundtrip_cell(float pixel, int pixels, int cells) {
  constexpr float kGrid = static_cast<float>(kernels::kSceneFlowGrid);
  if (cells <= 1) return 0.0F;
  const float before = (static_cast<float>(cells) - 1.5F) * kGrid;
  const float last = std::min(before + kGrid, static_cast<float>(pixels - 1));
  if (pixel >= before)
    return static_cast<float>(cells - 2) + std::min((pixel - before) / (last - before), 1.0F);
  return std::max((pixel - kGrid * 0.5F) / kGrid, 0.0F);
}

// Une cellule sur seize pour la mesure du journal. Coordonnées du flux, vecteurs signés S10.5.
inline std::optional<RoundtripStats> check_roundtrip(RoundtripShape shape,
                                                     std::span<const std::int16_t> forward,
                                                     std::span<const std::int16_t> reverse) {
  if (!shape.valid() || forward.size() != shape.words() || reverse.size() != shape.words()) return {};
  constexpr int kGrid = kernels::kSceneFlowGrid;
  constexpr int kProbeStep = 4;
  constexpr float kClosureAt720 = 1.25F;
  constexpr float kScale = 1.0F / 32.0F;
  const auto index = [&](int x, int y, int channel) {
    return static_cast<std::size_t>(y) * shape.row_words() + static_cast<std::size_t>(2 * x + channel);
  };
  const auto read = [&](float x, float y, int channel) {
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, shape.cols() - 1);
    const int y1 = std::min(y0 + 1, shape.rows() - 1);
    const float top =
        std::lerp(static_cast<float>(reverse[index(x0, y0, channel)]),
                  static_cast<float>(reverse[index(x1, y0, channel)]), x - static_cast<float>(x0));
    const float bottom =
        std::lerp(static_cast<float>(reverse[index(x0, y1, channel)]),
                  static_cast<float>(reverse[index(x1, y1, channel)]), x - static_cast<float>(x0));
    return std::lerp(top, bottom, y - static_cast<float>(y0)) * kScale;
  };
  RoundtripStats out;
  const float limit = kClosureAt720 * static_cast<float>(shape.height) / 720.0F;
  for (int y = 0; y < shape.rows(); y += kProbeStep) {
    for (int x = 0; x < shape.cols(); x += kProbeStep) {
      const float dx = static_cast<float>(forward[index(x, y, 0)]) * kScale;
      const float dy = static_cast<float>(forward[index(x, y, 1)]) * kScale;
      const float tx = static_cast<float>(std::min(x * kGrid + kGrid / 2, shape.width - 1)) + dx;
      const float ty = static_cast<float>(std::min(y * kGrid + kGrid / 2, shape.height - 1)) + dy;
      if (tx < 0.0F || ty < 0.0F || tx > static_cast<float>(shape.width - 1) ||
          ty > static_cast<float>(shape.height - 1)) {
        ++out.outside;
        continue;
      }
      const float rx = roundtrip_cell(tx, shape.width, shape.cols());
      const float ry = roundtrip_cell(ty, shape.height, shape.rows());
      const float ex = dx + read(rx, ry, 0);
      const float ey = dy + read(rx, ry, 1);
      ++out.checked;
      if (ex * ex + ey * ey > limit * limit) ++out.open;
    }
  }
  return out;
}

}  // namespace kirari::gpu::cuda
