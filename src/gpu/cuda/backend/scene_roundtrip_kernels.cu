// Le désaccord des deux sens, réduit directement dans la grille des tuiles de la carte.
#include <cuda_runtime.h>

#include "gpu/cuda/backend/scene_roundtrip_kernels.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kSide = kernels::kSceneTileCells;
constexpr int kGrid = kernels::kSceneFlowGrid;
constexpr unsigned kWarp = 0xFFFFFFFFU;
constexpr unsigned kFirst = 0x0000FFFFU;
constexpr int kCells = kSide * kSide;
static_assert(kCells == 16);

template <typename T>
__device__ T at(SceneRoundtripParams::Plane p, int x, int y) {
  const auto* ptr = static_cast<const unsigned char*>(p.pixels) + static_cast<std::size_t>(y) * p.pitch;
  return reinterpret_cast<const T*>(ptr)[x];
}

__device__ float interpolate(float a, float b, float f) {
  return a + (b - a) * f;
}

__device__ float cell(float pixel, int pixels, int cells) {
  const float before = (static_cast<float>(cells) - 1.5F) * kGrid;
  const float last = fminf(before + kGrid, static_cast<float>(pixels - 1));
  if (pixel >= before) return static_cast<float>(cells - 2) + fminf((pixel - before) / (last - before), 1.0F);
  return fmaxf((pixel - 0.5F * kGrid) / kGrid, 0.0F);
}

__device__ bool inside(float x, float y, int width, int height) {
  return x >= 0.0F && x <= static_cast<float>(width - 1) && y >= 0.0F && y <= static_cast<float>(height - 1);
}

__device__ float sample_luma(SceneRoundtripParams::Plane p, float x, float y, int w, int h) {
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = min(x0 + 1, w - 1);
  const int y1 = min(y0 + 1, h - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  return interpolate(interpolate(at<unsigned char>(p, x0, y0), at<unsigned char>(p, x1, y0), fx),
                     interpolate(at<unsigned char>(p, x0, y1), at<unsigned char>(p, x1, y1), fx), fy);
}

__device__ float2 sample_flow(SceneRoundtripParams::Plane p, float x, float y, int w, int h) {
  const int x0 = static_cast<int>(x);
  const int y0 = static_cast<int>(y);
  const int x1 = min(x0 + 1, w - 1);
  const int y1 = min(y0 + 1, h - 1);
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
  const short2 a = at<short2>(p, x0, y0);
  const short2 b = at<short2>(p, x1, y0);
  const short2 c = at<short2>(p, x0, y1);
  const short2 d = at<short2>(p, x1, y1);
  return {interpolate(interpolate(a.x, b.x, fx), interpolate(c.x, d.x, fx), fy) / 32.0F,
          interpolate(interpolate(a.y, b.y, fx), interpolate(c.y, d.y, fx), fy) / 32.0F};
}

__device__ bool disagrees(SceneRoundtripParams::Plane a, SceneRoundtripParams::Plane b, float x, float y,
                          float dx, float dy, int w, int h) {
  constexpr float kOffsets[5][2]{{0, 0}, {-1.5F, -1.5F}, {1.5F, -1.5F}, {-1.5F, 1.5F}, {1.5F, 1.5F}};
  int changed = 0;
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    const float px = x + kOffsets[i][0];
    const float py = y + kOffsets[i][1];
    const float qx = px + dx;
    const float qy = py + dy;
    if (inside(px, py, w, h) && inside(qx, qy, w, h) &&
        fabsf(sample_luma(a, px, py, w, h) - sample_luma(b, qx, qy, w, h)) > 12.0F) {
      ++changed;
    }
  }
  return changed >= 2;
}

__device__ std::uint8_t count(unsigned bits) {
  return static_cast<std::uint8_t>(__popc(bits));
}

__global__ void roundtrip(SceneRoundtripParams p, int cols, int rows) {
  const int t = static_cast<int>(threadIdx.x);
  const int i = t % kCells;
  const int x = static_cast<int>(blockIdx.x) * kSide + i % kSide;
  const int y = static_cast<int>(blockIdx.y) * kSide + i / kSide;
  const bool from_a = t < kCells;
  const auto forward = from_a ? p.flow_a : p.flow_b;
  const auto reverse = from_a ? p.flow_b : p.flow_a;
  bool visible = false;
  bool open = false;
  bool photo = false;
  if (x < cols && y < rows) {
    const short2 f = at<short2>(forward, x, y);
    const float sx = static_cast<float>(min(x * kGrid + kGrid / 2, p.width - 1));
    const float sy = static_cast<float>(min(y * kGrid + kGrid / 2, p.height - 1));
    const float dx = static_cast<float>(f.x) / 32.0F;
    const float dy = static_cast<float>(f.y) / 32.0F;
    const float qx = sx + dx;
    const float qy = sy + dy;
    visible = inside(qx, qy, p.width, p.height);
    if (visible) {
      const float2 back = sample_flow(reverse, cell(qx, p.width, cols), cell(qy, p.height, rows), cols, rows);
      const float ex = dx + back.x;
      const float ey = dy + back.y;
      const float limit = 1.25F * static_cast<float>(p.height) / 720.0F;
      open = ex * ex + ey * ey > limit * limit;
      if (open) {
        photo = disagrees(from_a ? p.luma_a : p.luma_b, from_a ? p.luma_b : p.luma_a, sx, sy, dx, dy, p.width,
                          p.height);
      }
    }
  }
  const unsigned valid_bits = __ballot_sync(kWarp, visible);
  const unsigned open_bits = __ballot_sync(kWarp, open);
  const unsigned photo_bits = __ballot_sync(kWarp, photo);
  if (t == 0) {
    p.out[static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x] = {
        .visible_a = count(valid_bits & kFirst),
        .open_a = count(open_bits & kFirst),
        .photo_a = count(photo_bits & kFirst),
        .visible_b = count(valid_bits >> kCells),
        .open_b = count(open_bits >> kCells),
        .photo_b = count(photo_bits >> kCells)};
  }
}

}  // namespace

cudaError_t launch_scene_roundtrip(const SceneRoundtripParams& p, cudaStream_t stream) {
  const int cols = kernels::scene_flow_cells(p.width);
  const int rows = kernels::scene_flow_cells(p.height);
  const dim3 tiles(static_cast<unsigned>(kernels::scene_flow_tiles(p.width)),
                   static_cast<unsigned>(kernels::scene_flow_tiles(p.height)));
  roundtrip<<<tiles, 32, 0, stream>>>(p, cols, rows);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
