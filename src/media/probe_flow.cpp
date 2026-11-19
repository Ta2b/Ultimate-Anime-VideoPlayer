// Le mouvement d'une image d'après les vecteurs de NVOF (motion_of) et la luma réduite de la pré-analyse
// (probe_luma_of) : pur calcul, sans GPU (media/probe_flow.h). La session NVOF est dans probe_flow_nvof.cpp.
#include "media/probe_flow.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "media/yuv_sample.h"

namespace kirari::media {
namespace {

// Un bloc est texturé si l'écart moyen de ses points à leur moyenne atteint kTextured niveaux sur 255 : en
// dessous, un aplat (fréquent en anime) se retrouve partout et ne dit rien d'une coupe.
constexpr float kTextured = 3.0F;
// Apparié : l'écart du bloc centré à son correspondant centré sous kMatchRatio fois son propre détail (deux
// contenus sans rapport s'écartent d'environ une à une fois et demie leur détail), ou sous kMatchNoise
// niveaux (le bruit de compression d'un bloc très détaillé).
constexpr float kMatchRatio = 0.5F;
constexpr float kMatchNoise = 2.0F;
// Moins d'un bloc sur kMinTexturedShare⁻¹ texturé : rien à juger.
constexpr float kMinTexturedShare = 0.05F;
constexpr float kStillPx = 0.5F;

constexpr int kSide = kProbeFlowGrid;
constexpr int kPoints = kSide * kSide;

struct Block {
  std::array<float, kPoints> centred{};
  float detail = 0.0F;
};

// Le bloc de 4 × 4 de luma dont le coin est (x0, y0), bords répétés hors de l'image, centré sur sa moyenne.
Block block_at(std::span<const std::uint8_t> luma, int x0, int y0) {
  Block out;
  float sum = 0.0F;
  for (int y = 0; y < kSide; ++y) {
    const int sy = std::clamp(y0 + y, 0, kProbeFlowHeight - 1);
    for (int x = 0; x < kSide; ++x) {
      const int sx = std::clamp(x0 + x, 0, kProbeFlowWidth - 1);
      const float v = luma[static_cast<std::size_t>(sy * kProbeFlowWidth + sx)];
      out.centred.at(static_cast<std::size_t>(y * kSide + x)) = v;
      sum += v;
    }
  }
  const float mean = sum / static_cast<float>(kPoints);
  float detail = 0.0F;
  for (float& v : out.centred) {
    v -= mean;
    detail += std::abs(v);
  }
  out.detail = detail / static_cast<float>(kPoints);
  return out;
}

float gap(const Block& a, const Block& b) {
  float sum = 0.0F;
  for (std::size_t i = 0; i < a.centred.size(); ++i) sum += std::abs(a.centred.at(i) - b.centred.at(i));
  return sum / static_cast<float>(kPoints);
}

float median(std::vector<float> values) {
  if (values.empty()) return 0.0F;
  const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), mid, values.end());
  return *mid;
}

}  // namespace

std::optional<ProbeMotion> motion_of(std::span<const std::uint8_t> previous,
                                     std::span<const std::uint8_t> current,
                                     std::span<const FlowVector> vectors) {
  constexpr auto kLumaSize = static_cast<std::size_t>(kProbeFlowWidth) * kProbeFlowHeight;
  constexpr auto kCells = static_cast<std::size_t>(kProbeFlowCellsX) * kProbeFlowCellsY;
  if (previous.size() != kLumaSize || current.size() != kLumaSize || vectors.size() != kCells)
    return std::nullopt;
  std::size_t textured = 0;
  std::size_t unmatched = 0;
  std::size_t still = 0;
  double motion = 0.0;
  std::vector<float> xs;
  std::vector<float> ys;
  xs.reserve(kCells);
  ys.reserve(kCells);
  for (int cy = 0; cy < kProbeFlowCellsY; ++cy) {
    for (int cx = 0; cx < kProbeFlowCellsX; ++cx) {
      const int x0 = cx * kSide;
      const int y0 = cy * kSide;
      const Block here = block_at(current, x0, y0);
      if (here.detail < kTextured) continue;
      ++textured;
      const FlowVector v = vectors[static_cast<std::size_t>(cy * kProbeFlowCellsX + cx)];
      const float fx = static_cast<float>(v.x) / 32.0F;
      const float fy = static_cast<float>(v.y) / 32.0F;
      const auto dx = static_cast<int>(std::lround(fx));
      const auto dy = static_cast<int>(std::lround(fy));
      // Le vecteur dans les deux sens, et nul : le meilleur des trois.
      const std::array<std::array<int, 3>, 3> tries{{{dx, dy, 1}, {-dx, -dy, -1}, {0, 0, 0}}};
      float best = 0.0F;
      int sign = 0;
      for (std::size_t i = 0; i < tries.size(); ++i) {
        const auto [tx, ty, s] = tries.at(i);
        const float g = gap(here, block_at(previous, x0 + tx, y0 + ty));
        if (i == 0 || g < best) {
          best = g;
          sign = s;
        }
      }
      if (best > std::max(kMatchRatio * here.detail, kMatchNoise)) {
        ++unmatched;
        continue;
      }
      const float mx = static_cast<float>(sign) * fx;
      const float my = static_cast<float>(sign) * fy;
      const float length = std::hypot(mx, my);
      xs.push_back(mx);
      ys.push_back(my);
      motion += length;
      if (length < kStillPx) ++still;
    }
  }
  if (static_cast<float>(textured) < kMinTexturedShare * static_cast<float>(kCells)) return std::nullopt;
  const std::size_t matched = textured - unmatched;
  const auto per = [](double part, std::size_t whole) {
    return whole == 0 ? 0.0F : static_cast<float>(part / static_cast<double>(whole));
  };
  const float shift_x = median(xs);
  const float shift_y = median(ys);
  double residual = 0.0;
  for (std::size_t i = 0; i < xs.size(); ++i) residual += std::hypot(xs[i] - shift_x, ys[i] - shift_y);
  return ProbeMotion{.unmatched = per(static_cast<double>(unmatched), textured),
                     .textured = per(static_cast<double>(textured), kCells),
                     .shift_x = shift_x,
                     .shift_y = shift_y,
                     .mean_motion = per(motion, matched),
                     .residual = per(residual, matched),
                     .still = per(static_cast<double>(still), matched)};
}

std::optional<ProbeLuma> probe_luma_of(const DecodedFrame& frame) {
  std::optional<yuv::Layout> layout = yuv::layout_of(frame.format);
  if (frame.format == PixelFormat::P016 || frame.format == PixelFormat::Yuv444p16)
    layout = yuv::Layout{.bits = 16, .semi_planar = frame.format == PixelFormat::P016, .high = true};
  if (!layout || frame.data == nullptr || frame.data->memory() != FrameMemory::Host) return std::nullopt;
  if (frame.width < kProbeFlowWidth / 8 || frame.height < kProbeFlowHeight / 8) return std::nullopt;
  const auto plane = frame.data->host_plane(0);
  const std::size_t stride = frame.data->host_stride(0);
  const int down = layout->bits - 8;
  ProbeLuma out(static_cast<std::size_t>(kProbeFlowWidth) * kProbeFlowHeight);
  // Moyenne de la zone de la source que couvre chaque point (un point seul si la source est plus petite).
  for (int oy = 0; oy < kProbeFlowHeight; ++oy) {
    const int y_from = oy * frame.height / kProbeFlowHeight;
    const int y_to = std::max(y_from + 1, (oy + 1) * frame.height / kProbeFlowHeight);
    for (int ox = 0; ox < kProbeFlowWidth; ++ox) {
      const int x_from = ox * frame.width / kProbeFlowWidth;
      const int x_to = std::max(x_from + 1, (ox + 1) * frame.width / kProbeFlowWidth);
      long sum = 0;
      for (int y = y_from; y < y_to; ++y)
        for (int x = x_from; x < x_to; ++x) sum += yuv::sample(plane, stride, x, y, *layout);
      const long count = static_cast<long>(y_to - y_from) * (x_to - x_from);
      const long value = ((sum + count / 2) / count) >> down;
      out[static_cast<std::size_t>(oy * kProbeFlowWidth + ox)] =
          static_cast<std::uint8_t>(std::clamp(value, 0L, 255L));
    }
  }
  return out;
}

}  // namespace kirari::media
