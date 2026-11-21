#include <limits>

#include "gpu/cuda/backend/scene_glow_kernels.h"
#include "kernels/spec/scene_glow.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kMaxSide = 4096;
constexpr int kGrid = 4;
constexpr int kMatchRadius = 16;
constexpr double kMinMoveSquared = 16.0 * 16.0;
constexpr double kMaxCost = 128.0;
constexpr double kMaxLumaDelta = 12.0;
constexpr double kHueCosine = 0.90;
constexpr float kExperimentalAmount = 0.35F;

struct Sample {
  int left;
  int right;
  int top;
  int bottom;
  double s;
  double t;

  __device__ double mix(double a, double b, double c, double d) const {
    const double high = a * (1.0 - s) + b * s;
    const double low = c * (1.0 - s) + d * s;
    return high * (1.0 - t) + low * t;
  }
};

__device__ Sample weights(double x, double y, int cols, int rows, int step) {
  const double u = fmin(fmax((x + 0.5) / step - 0.5, 0.0), cols - 1.0);
  const double v = fmin(fmax((y + 0.5) / step - 0.5, 0.0), rows - 1.0);
  const int left = static_cast<int>(u);
  const int top = static_cast<int>(v);
  return {left, min(left + 1, cols - 1), top, min(top + 1, rows - 1), u - left, v - top};
}

template <class T>
__device__ const T* row_at(const T* base, int row, std::size_t pitch) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(base) + row * pitch);
}

__device__ double sample_byte(const std::uint8_t* plane, std::size_t pitch, const Sample& at) {
  const auto* top = row_at(plane, at.top, pitch);
  const auto* bottom = row_at(plane, at.bottom, pitch);
  return at.mix(top[at.left], top[at.right], bottom[at.left], bottom[at.right]);
}

__device__ bool valid_anchor(const GpuAnchor& a, int width, int height) {
  return a.score > 0.0F && a.score <= 1.0F && a.x >= 0 && a.x < width && a.y >= 0 && a.y < height;
}

__device__ bool compatible(const GpuAnchor& a, const GpuAnchor& b) {
  return a.hue_x * b.hue_x + a.hue_y * b.hue_y >= kHueCosine;
}

__device__ bool confirmed_fixed(const GlowMatch& p, const GpuAnchor& from, int x, int y, int cols, int rows) {
  for (int row = max(y - 1, 0); row <= min(y + 1, rows - 1); ++row) {
    for (int col = max(x - 1, 0); col <= min(x + 1, cols - 1); ++col) {
      const GpuAnchor& prior = p.first[row * cols + col];
      if (valid_anchor(prior, p.width, p.height) && compatible(prior, from)) return true;
    }
  }
  return false;
}

__global__ void glow_match(GlowMatch p, int anchor_cols, int anchor_rows, int guide_cols, int search_count) {
  const int thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (thread >= search_count) return;
  const GpuAnchor start = p.first[thread];
  if (!valid_anchor(start, p.width, p.height)) return;

  const int flow_width = p.width / p.divisor;
  const int flow_height = p.height / p.divisor;
  const int cols = (flow_width + kGrid - 1) / kGrid;
  const int rows = (flow_height + kGrid - 1) / kGrid;
  const double x = (start.x + 0.5) / p.divisor - 0.5;
  const double y = (start.y + 0.5) / p.divisor - 0.5;
  const Sample at = weights(x, y, cols, rows, kGrid);
  const auto* top = row_at(p.flow, at.top, p.flow_pitch);
  const auto* bottom = row_at(p.flow, at.bottom, p.flow_pitch);
  const short2 a = top[at.left];
  const short2 b = top[at.right];
  const short2 c = bottom[at.left];
  const short2 d = bottom[at.right];
  const double dx = at.mix(a.x, b.x, c.x, d.x) * (1.0 / 32.0) * p.divisor;
  const double dy = at.mix(a.y, b.y, c.y, d.y) * (1.0 / 32.0) * p.divisor;
  const double target_x = x + dx / p.divisor;
  const double target_y = y + dy / p.divisor;
  if (dx * dx + dy * dy < kMinMoveSquared || target_x < -0.5 || target_x > flow_width - 0.5 ||
      target_y < -0.5 || target_y > flow_height - 0.5)
    return;
  if (sample_byte(p.cost, p.cost_pitch, at) > kMaxCost) return;
  const Sample before = weights(x, y, flow_width, flow_height, 1);
  const Sample after = weights(target_x, target_y, flow_width, flow_height, 1);
  if (fabs(sample_byte(p.luma_first, p.luma_first_pitch, before) -
           sample_byte(p.luma_second, p.luma_second_pitch, after)) > kMaxLumaDelta)
    return;

  const double projected_x = start.x + dx;
  const double projected_y = start.y + dy;
  const int cx = __double2int_rd(projected_x / p.anchor_block);
  const int cy = __double2int_rd(projected_y / p.anchor_block);
  const int reach = (kMatchRadius + p.anchor_block - 1) / p.anchor_block;
  for (int row = max(cy - reach, 0); row <= min(cy + reach, anchor_rows - 1); ++row) {
    for (int col = max(cx - reach, 0); col <= min(cx + reach, anchor_cols - 1); ++col) {
      const int offset = row * anchor_cols + col;
      const GpuAnchor now = p.second[offset];
      if (!valid_anchor(now, p.width, p.height) || !compatible(start, now)) continue;
      const double ax = now.x - projected_x;
      const double ay = now.y - projected_y;
      if (ax * ax + ay * ay > kMatchRadius * kMatchRadius) continue;
      if (confirmed_fixed(p, now, col, row, anchor_cols, anchor_rows)) continue;
      const int guide_cell = now.y / p.guide_block * guide_cols + now.x / p.guide_block;
      atomicMax(&p.guide[guide_cell], __float_as_uint(now.score * kExperimentalAmount));
    }
  }
}

bool valid_pitch(std::size_t pitch, std::size_t row_bytes, std::size_t align, int rows) {
  return pitch >= row_bytes && pitch % align == 0 &&
         pitch <= std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(rows);
}

bool valid_match_geometry(const GlowMatch& p) {
  return p.width > 0 && p.width <= kMaxSide && p.height > 0 && p.height <= kMaxSide && p.divisor > 0 &&
         p.divisor <= 4 && p.anchor_block > 0 && p.anchor_block <= kMaxSide && p.guide_block > 0 &&
         p.guide_block <= kMaxSide && p.first && p.second && p.flow && p.cost && p.luma_first &&
         p.luma_second && p.guide;
}

}  // namespace

cudaError_t launch_glow_match(const GlowMatch& p, cudaStream_t stream) {
  if (!valid_match_geometry(p)) return cudaErrorInvalidValue;
  const int flow_width = p.width / p.divisor, flow_height = p.height / p.divisor;
  if (flow_width == 0 || flow_height == 0) return cudaErrorInvalidValue;
  const int cols = glow_div_up(flow_width, kGrid), rows = glow_div_up(flow_height, kGrid);
  if (reinterpret_cast<std::uintptr_t>(p.flow) % alignof(short2) != 0 ||
      !valid_pitch(p.flow_pitch, cols * sizeof(short2), alignof(short2), rows) ||
      !valid_pitch(p.cost_pitch, cols, 1, rows) ||
      !valid_pitch(p.luma_first_pitch, flow_width, 1, flow_height) ||
      !valid_pitch(p.luma_second_pitch, flow_width, 1, flow_height))
    return cudaErrorInvalidPitchValue;

  const int anchor_cols = glow_div_up(p.width, p.anchor_block);
  const int anchor_rows = glow_div_up(p.height, p.anchor_block);
  const int guide_cols = glow_div_up(p.width, p.guide_block);
  const int guide_rows = glow_div_up(p.height, p.guide_block);
  if (p.guide_count != static_cast<std::size_t>(guide_cols * guide_rows)) return cudaErrorInvalidValue;
  if (auto ok = cudaMemsetAsync(p.guide, 0, p.guide_count * sizeof(*p.guide), stream); ok != cudaSuccess)
    return ok;
  const int count = anchor_cols * anchor_rows;
  constexpr int kThreads = 128;
  glow_match<<<glow_div_up(count, kThreads), kThreads, 0, stream>>>(p, anchor_cols, anchor_rows, guide_cols,
                                                                    count);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
