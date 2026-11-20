// Noyaux du softmax splatting de GMFSS (voir gmfss_kernels.h) ; référence kernels/ref/gmfss.cpp. Un lot :
// gridDim.z est le travail, chaque bloc une tuile 16 × 16 de sa source ; les tuiles hors d'un travail plus
// petit que le plus grand sortent aussitôt. Les lectures (mouvement sur une autre grille, splat tiré,
// composition) passent par la bibliothèque de warp (warp_lib.cuh) : une prise bornée par point, partagée par
// tous les plans qu'il lit.
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/gmfss_kernels.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "gpu/cuda/backend/warp_lib.cuh"
#include "kernels/spec/gmfss.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kBlock = 16;
static_assert(kBlock * kBlock == kPixelThreads);
// Splat tuilé (étude 3) : la tuile pousse ses contributions dans une fenêtre partagée de
// kSpan × kSpan cellules, placée sur le déplacement moyen de la tuile (un panoramique y reste
// entier) avec kMargin cellules de jeu de chaque côté ; une contribution hors de la fenêtre va
// directement aux sommes globales. kPass plans par passe, plus celui des poids à la première :
// Avec une marge de quatre, chaque bloc réserve 11,25 Kio.
constexpr int kMargin = 4;
constexpr int kSpan = kBlock + 2 * kMargin;
constexpr int kCells = kSpan * kSpan;
constexpr int kPass = 4;

__host__ __device__ int tiles_across(int size) {
  return (size + kBlock - 1) / kBlock;
}

__device__ std::size_t at(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

__device__ std::size_t weight_offset(const SoftSplatParams& p, std::size_t plane) {
  return static_cast<std::size_t>(p.weight_planes > 0 ? p.weight_planes : p.channels) * plane;
}

struct ShiftBounds {
  float3 shift;
  int4 bounds;  // min x, min y, max x, max y des coins nord-ouest
};

template <bool Min>
__device__ __forceinline__ int warp_bound(int x) {
#if __CUDA_ARCH__ >= 800
  if constexpr (Min) return __reduce_min_sync(0xFFFFFFFFU, x);
  return __reduce_max_sync(0xFFFFFFFFU, x);
#else
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    const int next = __shfl_down_sync(0xFFFFFFFFU, x, offset);
    x = Min ? min(x, next) : max(x, next);
  }
  return x;
#endif
}

// Le même déplacement moyen qu'avant. Pour les fenêtres bornées, les extrémums des
// projections rejoignent les sommes dans les deux mêmes barrières.
template <bool Crop>
__device__ ShiftBounds block_sum3(float3 v, int x, int y, bool active) {
  __shared__ float3 total;
  __shared__ int4 bounds;
  constexpr int top = 0x3fffffff;
  const int t = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  if (t == 0) {
    total = make_float3(0.0F, 0.0F, 0.0F);
    if constexpr (Crop) bounds = make_int4(top, top, -top, -top);
  }
  __syncthreads();
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    v.x += __shfl_down_sync(0xFFFFFFFFU, v.x, offset);
    v.y += __shfl_down_sync(0xFFFFFFFFU, v.y, offset);
    v.z += __shfl_down_sync(0xFFFFFFFFU, v.z, offset);
  }
  int4 edge{};
  if constexpr (Crop) {
    edge.y = warp_bound<true>(active ? y : top);
    edge.w = warp_bound<false>(active ? y : -top);
    edge.x = warp_bound<true>(active ? x : top);
    edge.z = warp_bound<false>(active ? x : -top);
  }
  if (t % kWarpSize == 0) {
    atomicAdd(&total.x, v.x);
    atomicAdd(&total.y, v.y);
    atomicAdd(&total.z, v.z);
    if constexpr (Crop) {
      atomicMin(&bounds.y, edge.y);
      atomicMax(&bounds.w, edge.w);
      atomicMin(&bounds.x, edge.x);
      atomicMax(&bounds.z, edge.z);
    }
  }
  __syncthreads();
  if constexpr (!Crop) return {total, {}};
  return {total, bounds};
}

// Où va le pixel (x, y) du travail p : sa cible (tx, ty), sa métrique z ; inactif hors du plan, hors de
// ]−1, taille[ (aucun coin dans le plan), NaN et infinis compris.
struct SplatTarget {
  std::size_t i = 0;  // le pixel dans un plan de la source
  float tx = 0.0F;
  float ty = 0.0F;
  float z = 0.0F;
  bool active = false;
};

// La prise bilinéaire bornée au plan w × h en (x, y) (spec/warp.h : bornes en clamp, NaN donne le bord bas).
__device__ __forceinline__ warp::Tap plane_tap(float x, float y, int w, int h) {
  return warp::tap_at(0, 0, x, y, warp::Bounds{0, 0, w, h});
}

__device__ __forceinline__ SplatTarget splat_target(const SoftSplatParams& p, int x, int y,
                                                    std::size_t plane) {
  SplatTarget s;
  const bool pixel = x < p.width && y < p.height;
  float fx = 0.0F;
  float fy = 0.0F;
  if (pixel) {
    const auto* motion = static_cast<const __half*>(p.motion);
    s.i = at(x, y, p.width);
    fx = __half2float(motion[s.i]);
    fy = __half2float(motion[plane + s.i]);
    s.z = __half2float(motion[2 * plane + s.i]);
  }
  s.tx = static_cast<float>(x) + p.s * fx;
  s.ty = static_cast<float>(y) + p.s * fy;
  s.active = pixel && s.tx > -1.0F && s.tx < static_cast<float>(p.width) && s.ty > -1.0F &&
             s.ty < static_cast<float>(p.height);
  return s;
}

// Les quatre coins bilinéaires de la cible : leur poids (exp(s · z) borné, comme la référence, fois la part
// bilinéaire), leur somme globale dans le premier plan, leur cellule dans la fenêtre partagée (ox, oy) ou −1.
struct SplatCorners {
  unsigned inside = 0;  // bit k : le coin k est dans le plan
  float weight[4] = {};
  float* corner[4] = {};
  int cell[4] = {};
};

__device__ __forceinline__ SplatCorners splat_corners(const SoftSplatParams& p, const SplatTarget& s, int ox,
                                                      int oy, int cx0, int cy0) {
  SplatCorners q;
  // z borné comme la référence (NaN : le bas de l'intervalle, comme fmaxf).
  const float bounded_z = fminf(fmaxf(s.z, -kernels::kGmfssMetricLimit), kernels::kGmfssMetricLimit);
  const float scale = s.active ? expf(p.s * bounded_z) : 0.0F;  // une fois par pixel, pour tous les plans
  const float east = s.tx - static_cast<float>(cx0);
  const float west = static_cast<float>(cx0 + 1) - s.tx;
  const float south = s.ty - static_cast<float>(cy0);
  const float north = static_cast<float>(cy0 + 1) - s.ty;
  const int cx[4] = {cx0, cx0 + 1, cx0, cx0 + 1};
  const int cy[4] = {cy0, cy0, cy0 + 1, cy0 + 1};
  const float share[4] = {west * north, east * north, west * south, east * south};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const bool in = s.active && cx[k] >= 0 && cx[k] < p.width && cy[k] >= 0 && cy[k] < p.height;
    q.inside |= in ? 1U << k : 0U;
    q.weight[k] = scale * share[k];
    q.corner[k] = p.sums + (in ? at(cx[k], cy[k], p.width) : 0);
    const int lx = cx[k] - ox;
    const int ly = cy[k] - oy;
    q.cell[k] = lx >= 0 && lx < kSpan && ly >= 0 && ly < kSpan ? ly * kSpan + lx : -1;
  }
  return q;
}

// value fois le poids de chaque coin du plan : dans la fenêtre (window, un plan de cellules), ou aux sommes
// globales décalées de offset.
__device__ __forceinline__ void splat_value(const SplatCorners& q, float* window, std::size_t offset,
                                            float value) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    if ((q.inside >> k & 1U) == 0) continue;
    if (q.cell[k] >= 0) {
      atomicAdd(&window[q.cell[k]], value * q.weight[k]);
    } else {
      atomicAdd(q.corner[k] + offset, value * q.weight[k]);
    }
  }
}

// La fenêtre vers les sommes globales : une addition par cellule touchée et par plan (count plans à partir
// de first, et celui des poids si with_weights).
__device__ __forceinline__ void flush_window(const SoftSplatParams& p, const float* cells, int ox, int oy,
                                             int first, int count, bool with_weights, std::size_t plane,
                                             int t) {
  const std::size_t weights = weight_offset(p, plane);
  for (int k = t; k < kCells; k += kPixelThreads) {
    const int gx = ox + k % kSpan;
    const int gy = oy + k / kSpan;
    if (gx < 0 || gy < 0 || gx >= p.width || gy >= p.height) continue;
    float* sum = p.sums + at(gx, gy, p.width);
#pragma unroll 1
    for (int c = 0; c < count; ++c) {
      const float v = cells[c * kCells + k];
      if (v != 0.0F) atomicAdd(sum + static_cast<std::size_t>(first + c) * plane, v);
    }
    const float w = with_weights ? cells[kPass * kCells + k] : 0.0F;
    if (w != 0.0F) atomicAdd(sum + weights, w);
  }
}

struct WindowRect {
  int x = 0, y = 0, cols = 0, count = 0, reciprocal = 0;

  // Pour i < 576 et 1 <= cols <= 24, ceil(2^16 / cols) multiplie sans changer floor(i / cols).
  __device__ __forceinline__ int row(int i) const { return (i * reciprocal) >> 16; }
  __device__ __forceinline__ int cell(int i, int r) const { return (y + r) * kSpan + x + i - r * cols; }
};

constexpr bool window_reciprocal_exact() {
  for (int w = 1; w <= kSpan; ++w)
    for (int i = 0; i < kCells; ++i)
      if ((i * ((65536 + w - 1) / w) >> 16) != i / w) return false;
  return true;
}
static_assert(window_reciprocal_exact());

__device__ WindowRect window_rect(int4 bounds, int ox, int oy, int width, int height) {
  const int left = max(0, max(ox, bounds.x));
  const int right = min(width, min(ox + kSpan, bounds.z + 2));
  const int top = max(0, max(oy, bounds.y)), bottom = min(height, min(oy + kSpan, bounds.w + 2));
  const int cols = max(0, right - left), rows = max(0, bottom - top);
  return {left - ox, top - oy, cols, cols * rows, cols == 0 ? 0 : (65536 + cols - 1) / cols};
}

__device__ void clear_rect(float* cells, WindowRect rect, int count, bool weights, int t) {
  for (int i = t; i < rect.count; i += kPixelThreads) {
    const int k = rect.cell(i, rect.row(i));
#pragma unroll
    for (int c = 0; c < kPass; ++c)
      if (c < count) cells[c * kCells + k] = 0.F;
    if (weights) cells[kPass * kCells + k] = 0.F;
  }
}

__device__ void flush_rect(const SoftSplatParams& p, const float* cells, WindowRect rect,
                           int ox, int oy, int first, int count, bool weights, std::size_t plane, int t) {
  for (int i = t; i < rect.count; i += kPixelThreads) {
    const int row = rect.row(i);
    const int k = rect.cell(i, row);
    float* sum = p.sums + at(ox + rect.x + i - row * rect.cols, oy + rect.y + row, p.width);
#pragma unroll 1
    for (int c = 0; c < count; ++c) {
      const float v = cells[c * kCells + k];
      if (v != 0.F) atomicAdd(sum + static_cast<std::size_t>(first + c) * plane, v);
    }
    const float w = weights ? cells[kPass * kCells + k] : 0.F;
    if (w != 0.F) atomicAdd(sum + weight_offset(p, plane), w);
  }
}

template <bool Crop>
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) splat_all(SoftSplatBatch b) {
  __shared__ float cells[(kPass + 1) * kCells];
  const SoftSplatParams& p = b.jobs[blockIdx.z];
  const int x0 = static_cast<int>(blockIdx.x) * kBlock;
  const int y0 = static_cast<int>(blockIdx.y) * kBlock;
  if (x0 >= p.width || y0 >= p.height) return;  // tout le bloc
  const int t = static_cast<int>(threadIdx.y) * kBlock + static_cast<int>(threadIdx.x);
  const int x = x0 + static_cast<int>(threadIdx.x);
  const int y = y0 + static_cast<int>(threadIdx.y);
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const SplatTarget s = splat_target(p, x, y, plane);
  const int cx0 = s.active ? static_cast<int>(floorf(s.tx)) : 0;
  const int cy0 = s.active ? static_cast<int>(floorf(s.ty)) : 0;
  const ShiftBounds shift_bounds =
      block_sum3<Crop>(s.active ? make_float3(s.tx - static_cast<float>(x), s.ty - static_cast<float>(y), 1.F)
                               : make_float3(0.F, 0.F, 0.F), cx0, cy0, s.active);
  const float3 shift = shift_bounds.shift;
  if (shift.z == 0.0F) return;  // tout le bloc : rien n'arrive dans le plan
  const int ox = x0 - kMargin + __float2int_rn(shift.x / shift.z);
  const int oy = y0 - kMargin + __float2int_rn(shift.y / shift.z);
  const SplatCorners q = splat_corners(p, s, ox, oy, cx0, cy0);
  WindowRect rect;
  if constexpr (Crop) rect = window_rect(shift_bounds.bounds, ox, oy, p.width, p.height);
  const __half* from = static_cast<const __half*>(p.source) + s.i;
  const std::size_t weights = weight_offset(p, plane);
#pragma unroll 1
  for (int first = 0; first < p.channels; first += kPass) {
    const int count = min(kPass, p.channels - first);
    const bool with_weights = first == 0 && !p.reuse_weights;
    if constexpr (!Crop) {
      const int cleared = with_weights ? (kPass + 1) * kCells : kPass * kCells;
      for (int k = t; k < cleared; k += kPixelThreads) cells[k] = 0.0F;
    } else {
      clear_rect(cells, rect, count, with_weights, t);
    }
    __syncthreads();
    if (q.inside != 0) {
#pragma unroll 1
      for (int c = 0; c < count; ++c) {
        const std::size_t global = static_cast<std::size_t>(first + c) * plane;
        splat_value(q, cells + c * kCells, global, __half2float(from[global]));
      }
      if (with_weights) splat_value(q, cells + kPass * kCells, weights, 1.0F);
    }
    __syncthreads();
    if constexpr (!Crop)
      flush_window(p, cells, ox, oy, first, count, with_weights, plane, t);
    else
      flush_rect(p, cells, rect, ox, oy, first, count, with_weights, plane, t);
    __syncthreads();
  }
}

// Le mouvement lu sur sa grille (kernels::MotionGrid), écrit à la taille de la source : le splat le lit
// ensuite pixel par pixel, comme celui de son propre niveau.
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) motion_on_grid(MotionGridParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const auto* motion = static_cast<const __half*>(p.motion);
  const std::size_t grid =
      static_cast<std::size_t>(p.motion_width) * static_cast<std::size_t>(p.motion_height);
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const float u = (static_cast<float>(x) + 0.5F) / p.scale - 0.5F + p.x;
  const float v = (static_cast<float>(y) + 0.5F) / p.scale - 0.5F + p.y;
  auto* out = static_cast<__half*>(p.out);
  const std::size_t i = at(x, y, p.width);
  const warp::Tap tap = plane_tap(u, v, p.motion_width, p.motion_height);  // une prise pour les trois plans
  out[i] = __float2half_rn(warp::bilinear(motion, tap, p.motion_width) * p.scale);
  out[plane + i] = __float2half_rn(warp::bilinear(motion + grid, tap, p.motion_width) * p.scale);
  out[2 * plane + i] = __float2half_rn(warp::bilinear(motion + 2 * grid, tap, p.motion_width));
}

// Deux pixels voisins par fil : sommes lues en float2 et sorties écrites en half2 quand le travail
// le permet (pairs), un pixel puis l'autre sinon. Les blocs du premier travail remplissent aussi le
// tenseur de l'instant (fill) quand le lot en a un, sans bloc de plus.
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) normalize_all(SoftSplatBatch b) {
  if (blockIdx.z == 0 && b.fill != nullptr) {
    auto* fill = static_cast<__half*>(b.fill);
    const __half value = __float2half_rn(b.fill_value);
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * gridDim.y * kPixelThreads;
    for (std::size_t k = (static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * kPixelThreads +
                         threadIdx.y * kBlock + threadIdx.x;
         k < b.fill_count; k += stride)
      fill[k] = value;
  }
  const SoftSplatParams& p = b.jobs[blockIdx.z];
  if (p.out_pitch != 0) return;
  const int x = 2 * static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width);
  const float* weights = p.sums + weight_offset(p, plane);
  if (p.pairs != 0) {
    const float2 weight = *reinterpret_cast<const float2*>(weights + i);
    const float2 total =
        make_float2(weight.x + kernels::kGmfssNormalizeEpsilon, weight.y + kernels::kGmfssNormalizeEpsilon);
    auto* out = static_cast<__half2*>(p.out);
    for (int c = 0; c < p.channels; ++c) {
      const std::size_t k = static_cast<std::size_t>(c) * plane + i;
      const float2 sum = *reinterpret_cast<const float2*>(p.sums + k);
      out[k / 2] = __floats2half2_rn(sum.x / total.x, sum.y / total.y);
    }
    if (p.coverage != 0) {
      const std::size_t k = static_cast<std::size_t>(p.channels) * plane + i;
      out[k / 2] = __floats2half2_rn(weight.x / total.x, weight.y / total.y);
    }
    return;
  }
  auto* out = static_cast<__half*>(p.out);
  for (int e = 0; e < 2 && x + e < p.width; ++e) {
    const float weight = weights[i + e];
    const float total = weight + kernels::kGmfssNormalizeEpsilon;
    for (int c = 0; c < p.channels; ++c) {
      const std::size_t k = static_cast<std::size_t>(c) * plane + i + e;
      out[k] = __float2half_rn(p.sums[k] / total);
    }
    if (p.coverage != 0)
      out[static_cast<std::size_t>(p.channels) * plane + i + e] = __float2half_rn(weight / total);
  }
}

// Les numérateurs restent en plans FP32 dans le L2. La tuile change l'ordre des écritures, après la même
// division et le même arrondi FP16 que normalize_all. Quatre canaux voisins tiennent dans un uint2.
constexpr int kGridPixels = 32;
constexpr int kGridChannels = 8;
constexpr int kGridThreads = 128;

__global__ void normalize_grid(SoftSplatParams p) {
  __shared__ __half tile[kGridChannels][kGridPixels + 1];
  const int tid = static_cast<int>(threadIdx.x);
  const auto plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const auto first = static_cast<std::size_t>(blockIdx.x) * kGridPixels;
  const int c0 = static_cast<int>(blockIdx.y) * kGridChannels;
  const int px = tid % kGridPixels;
  const std::size_t i = first + static_cast<std::size_t>(px);
  if (i < plane) {
    const float total = p.sums[weight_offset(p, plane) + i] + kernels::kGmfssNormalizeEpsilon;
    for (int c = tid / kGridPixels; c < kGridChannels && c0 + c < p.channels;
         c += kGridThreads / kGridPixels) {
      const float sum = p.sums[static_cast<std::size_t>(c0 + c) * plane + i];
      tile[c][px] = __float2half_rn(sum / total);
    }
  }
  __syncthreads();
  if (tid >= kGridPixels * (kGridChannels / 4)) return;
  constexpr int kGroups = kGridChannels / 4;
  const int local = tid / kGroups;
  const int c = 4 * (tid % kGroups);
  const std::size_t pixel = first + static_cast<std::size_t>(local);
  if (pixel >= plane || c0 + c >= p.channels) return;
  const auto pair = [&](int at) -> std::uint32_t {
    return static_cast<std::uint32_t>(__half_as_ushort(tile[at][local])) |
           (static_cast<std::uint32_t>(__half_as_ushort(tile[at + 1][local])) << 16U);
  };
  auto* out = static_cast<__half*>(p.out) + pixel * static_cast<std::size_t>(p.out_pitch) + c0 + c;
  *reinterpret_cast<uint2*>(out) = make_uint2(pair(c), pair(c + 2));
}

// Chaque groupe de huit canaux se suit dans son propre plan. Les demi-flottants des deux pixels
// voisins restent dans le même mot jusqu'à la transposition, puis chaque fil écrit quatre ou huit canaux.
__global__ void normalize_grid_c8(SoftSplatParams p) {
  __shared__ std::uint32_t tile[kGridChannels][kGridPixels + 1];
  const int tid = static_cast<int>(threadIdx.x), pair = tid % kGridPixels;
  const int from_channel = max(0, static_cast<int>(blockIdx.y) * kGridChannels - p.out_first % kGridChannels);
  const int to_channel = p.out_first + from_channel, lane = to_channel % kGridChannels;
  const int count = min(kGridChannels - lane, p.channels - from_channel);
  const auto plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const auto first = static_cast<std::size_t>(blockIdx.x) * 2 * kGridPixels;
  const auto pixel_pair = first + 2 * static_cast<std::size_t>(pair);
  if (pixel_pair < plane) {
    const float2 w = *reinterpret_cast<const float2*>(p.sums + weight_offset(p, plane) + pixel_pair);
    const float2 total =
        make_float2(w.x + kernels::kGmfssNormalizeEpsilon, w.y + kernels::kGmfssNormalizeEpsilon);
    for (int c = tid / kGridPixels; c < count; c += kGridThreads / kGridPixels) {
      const auto at = static_cast<std::size_t>(from_channel + c) * plane + pixel_pair;
      const float2 sum = *reinterpret_cast<const float2*>(p.sums + at);
      const __half2 result = __floats2half2_rn(sum.x / total.x, sum.y / total.y);
      tile[c][pair] = static_cast<std::uint32_t>(__half_as_ushort(__low2half(result))) |
                      (static_cast<std::uint32_t>(__half_as_ushort(__high2half(result))) << 16U);
    }
  }
  __syncthreads();
  if (tid >= 2 * kGridPixels) return;
  const auto pixel = first + static_cast<std::size_t>(tid);
  if (pixel >= plane) return;
  auto* out = static_cast<__half*>(p.out) + static_cast<std::size_t>(to_channel - lane) * plane +
              pixel * kGridChannels + lane;
  const auto packed = [&](int a, int b) {
    return __byte_perm(tile[a][tid / 2], tile[b][tid / 2], tid % 2 == 0 ? 0x5410 : 0x7632);
  };
  const uint2 lo = make_uint2(packed(0, 1), packed(2, 3));
  if (count == kGridChannels)
    *reinterpret_cast<uint4*>(out) = make_uint4(lo.x, lo.y, packed(4, 5), packed(6, 7));
  else
    *reinterpret_cast<uint2*>(out) = lo;
}

// Un fil par pixel de sortie : les deux côtés agrandis au pixel, ceux de RIFE et la fusion en bilinéaire.
__global__ void __launch_bounds__(kPixelThreads) compose(ComposeParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.out_width || y >= p.out_height) return;
  const std::size_t out_plane =
      static_cast<std::size_t>(p.out_width) * static_cast<std::size_t>(p.out_height);
  const std::size_t low_plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t fusion_plane =
      static_cast<std::size_t>(p.fusion_width) * static_cast<std::size_t>(p.fusion_height);
  const float xw =
      (static_cast<float>(x) + 0.5F) * static_cast<float>(p.width) / static_cast<float>(p.out_width) - 0.5F;
  const float yw =
      (static_cast<float>(y) + 0.5F) * static_cast<float>(p.height) / static_cast<float>(p.out_height) - 0.5F;
  const auto* ha = static_cast<const __half*>(p.high_a);
  const auto* hb = static_cast<const __half*>(p.high_b);
  const auto* la = static_cast<const __half*>(p.low_a);
  const auto* lb = static_cast<const __half*>(p.low_b);
  const auto* fusion = static_cast<const __half*>(p.fusion);
  const std::size_t i = at(x, y, p.out_width);
  // Une prise pour les huit lectures de RIFE, une pour les trois de la fusion (bornée à sa fenêtre).
  const warp::Tap rife_tap = plane_tap(xw, yw, p.width, p.height);
  const float w0h = (1.0F - p.t) * __half2float(ha[3 * out_plane + i]);
  const float w1h = p.t * __half2float(hb[3 * out_plane + i]);
  const float w0l = (1.0F - p.t) * warp::bilinear(la + 3 * low_plane, rife_tap, p.width);
  const float w1l = p.t * warp::bilinear(lb + 3 * low_plane, rife_tap, p.width);
  const float high_total = w0h + w1h + kernels::kGmfssNormalizeEpsilon;
  const float low_total = w0l + w1l + kernels::kGmfssNormalizeEpsilon;
  // La fusion dans sa fenêtre : coordonnées bornées à l'image de RIFE, décalées du coin.
  const warp::Tap fused_tap =
      warp::tap_at(p.left, p.top, xw, yw, warp::Bounds{p.left, p.top, p.left + p.width, p.top + p.height});
  auto* out = static_cast<__half*>(p.out);
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const std::size_t oc = static_cast<std::size_t>(c) * out_plane;
    const std::size_t lc = static_cast<std::size_t>(c) * low_plane;
    const float high = (w0h * __half2float(ha[oc + i]) + w1h * __half2float(hb[oc + i])) / high_total;
    const float low = (w0l * warp::bilinear(la + lc, rife_tap, p.width) +
                       w1l * warp::bilinear(lb + lc, rife_tap, p.width)) /
                      low_total;
    const float fused =
        warp::bilinear(fusion + static_cast<std::size_t>(c) * fusion_plane, fused_tap, p.fusion_width);
    out[oc + i] = __float2half_rn(fminf(fmaxf(high + fused - low, 0.0F), 1.0F));
  }
}

// Un fil par pixel de sortie : la source lue là d'où vient le pixel, d'après le flot poussé.
__global__ void __launch_bounds__(kPixelThreads) pull(PullParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width);
  const auto* flow = static_cast<const __half*>(p.flow);
  // Catmull-Rom de warp_lib (spec/warp.h, RifeWarp::CatmullRom) : une prise pour tous les plans.
  const warp::Bounds bounds{0, 0, p.width, p.height};
  const warp::Tap tap =
      warp::tap_at(x, y, -p.s * __half2float(flow[i]), -p.s * __half2float(flow[plane + i]), bounds);
  const auto* source = static_cast<const __half*>(p.source);
  auto* out = static_cast<__half*>(p.out);
  for (int c = 0; c < p.channels; ++c) {
    const std::size_t pc = static_cast<std::size_t>(c) * plane;
    out[pc + i] = __float2half_rn(warp::catmull_rom(source + pc, tap, p.width, bounds));
  }
  if (p.coverage != 0) out[static_cast<std::size_t>(p.channels) * plane + i] = flow[2 * plane + i];
}

// Un fil par pixel : l'écart des sources sur son voisinage 3 × 3, puis la clé tirée vers leur mélange.
__global__ void __launch_bounds__(kPixelThreads) still(StillParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const auto* a = static_cast<const __half*>(p.a);
  const auto* b = static_cast<const __half*>(p.b);
  float d = 0.0F;
  for (int dy = -1; dy <= 1; ++dy) {
    const int sy = min(max(y + dy, 0), p.height - 1);
    for (int dx = -1; dx <= 1; ++dx) {
      const std::size_t j = at(min(max(x + dx, 0), p.width - 1), sy, p.width);
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        const std::size_t k = static_cast<std::size_t>(c) * plane + j;
        d = fmaxf(d, fabsf(__half2float(a[k]) - __half2float(b[k])));
      }
    }
  }
  const float m = fminf(
      fmaxf((kernels::kGmfssStillHigh - d) / (kernels::kGmfssStillHigh - kernels::kGmfssStillLow), 0.0F),
      1.0F);
  if (m <= 0.0F) return;  // la clé telle quelle
  auto* key = static_cast<__half*>(p.key);
  const std::size_t i = at(x, y, p.width);
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const std::size_t k = static_cast<std::size_t>(c) * plane + i;
    const float value = __half2float(key[k]);
    const float source = (1.0F - p.t) * __half2float(a[k]) + p.t * __half2float(b[k]);
    key[k] = __float2half_rn(fminf(fmaxf(value + m * (source - value), 0.0F), 1.0F));
  }
}

}  // namespace

std::size_t soft_splat_sums_bytes(int width, int height, int channels) {
  return static_cast<std::size_t>(channels + 1) * static_cast<std::size_t>(width) *
         static_cast<std::size_t>(height) * sizeof(float);
}

cudaError_t launch_soft_splat_all(SoftSplatBatch& batch, void* sums, std::size_t sums_bytes,
                                  cudaStream_t stream) {
  if (batch.count < 1 || batch.count > kMaxSplatJobs) return cudaErrorInvalidValue;
  int width = 0;
  int height = 0;
  bool normalize_planes = batch.fill != nullptr;
  for (int j = 0; j < batch.count; ++j) {
    SoftSplatParams& p = batch.jobs[j];
    if (p.out_pitch > 0 && (p.out_pitch % 4 != 0 || p.channels < 4 || p.channels % 4 != 0 ||
                            p.out_pitch < p.channels || p.coverage != 0 ||
                            reinterpret_cast<std::uintptr_t>(p.out) % alignof(uint2) != 0))
      return cudaErrorInvalidValue;
    if (p.out_pitch < 0 &&
        (p.width % 2 != 0 || p.out_pitch % 16 != 0 || p.channels < 4 || p.channels % 4 != 0 ||
         p.out_first < 0 || p.out_first % 4 != 0 || p.out_first > -(p.out_pitch / 2) - p.channels ||
         p.coverage != 0 || reinterpret_cast<std::uintptr_t>(p.out) % alignof(uint4) != 0))
      return cudaErrorInvalidValue;
    normalize_planes = normalize_planes || p.out_pitch == 0;
    width = max(width, p.width);
    height = max(height, p.height);
    // Sommes alignées sur 256 octets (le lot) ; largeur paire : plans et pixels pairs alignés.
    p.pairs = p.width % 2 == 0 && reinterpret_cast<std::uintptr_t>(p.out) % 4 == 0 ? 1 : 0;
  }
  const cudaError_t cleared = cudaMemsetAsync(sums, 0, sums_bytes, stream);
  if (cleared != cudaSuccess) return cleared;
  const dim3 block(kBlock, kBlock);
  const auto rows = static_cast<unsigned>(tiles_across(height));
  const auto jobs = static_cast<unsigned>(batch.count);
  static const bool crop_window = [] {
    const char* value = std::getenv("KIRARI_AB_GMFSS_SPLAT_CROP");  // env: banc
    return value != nullptr && std::string_view(value) == "1";
  }();
  const dim3 push_grid(static_cast<unsigned>(tiles_across(width)), rows, jobs);
  if (crop_window && batch.count == 1 && batch.jobs[0].channels >= 8 && width >= 128 && height >= 96)
    splat_all<true><<<push_grid, block, 0, stream>>>(batch);
  else
    splat_all<false><<<push_grid, block, 0, stream>>>(batch);
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  if (normalize_planes) {
    normalize_all<<<dim3(static_cast<unsigned>(tiles_across((width + 1) / 2)), rows, jobs), block, 0, stream>>>(
        batch);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  }
  for (int j = 0; j < batch.count; ++j) {
    const SoftSplatParams& p = batch.jobs[j];
    if (p.out_pitch == 0) continue;
    const auto plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
    if (p.out_pitch < 0) {
      const dim3 grid(static_cast<unsigned>((plane + 2 * kGridPixels - 1) / (2 * kGridPixels)),
                      static_cast<unsigned>((p.out_first % kGridChannels + p.channels + kGridChannels - 1) /
                                            kGridChannels));
      normalize_grid_c8<<<grid, kGridThreads, 0, stream>>>(p);
    } else {
      const dim3 grid(static_cast<unsigned>((plane + kGridPixels - 1) / kGridPixels),
                      static_cast<unsigned>((p.channels + kGridChannels - 1) / kGridChannels));
      normalize_grid<<<grid, kGridThreads, 0, stream>>>(p);
    }
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  }
  return cudaGetLastError();
}

cudaError_t launch_motion_on_grid(const MotionGridParams& params, cudaStream_t stream) {
  if (params.width < 1 || params.height < 1 || params.motion_width < 1 || params.motion_height < 1 ||
      !(params.scale > 0.0F))
    return cudaErrorInvalidValue;
  const dim3 block(kBlock, kBlock);
  motion_on_grid<<<dim3(static_cast<unsigned>(tiles_across(params.width)),
                        static_cast<unsigned>(tiles_across(params.height))),
                   block, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_gmfss_compose(const ComposeParams& params, cudaStream_t stream) {
  if (params.width < 1 || params.height < 1 || params.out_width < params.width ||
      params.out_height < params.height || params.left < 0 || params.top < 0 ||
      params.left + params.width > params.fusion_width || params.top + params.height > params.fusion_height)
    return cudaErrorInvalidValue;
  const dim3 block(kBlock, kBlock);
  const dim3 grid(static_cast<unsigned>(tiles_across(params.out_width)),
                  static_cast<unsigned>(tiles_across(params.out_height)));
  compose<<<grid, block, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_gmfss_still(const StillParams& params, cudaStream_t stream) {
  if (params.width < 1 || params.height < 1) return cudaErrorInvalidValue;
  const dim3 block(kBlock, kBlock);
  const dim3 grid(static_cast<unsigned>(tiles_across(params.width)),
                  static_cast<unsigned>(tiles_across(params.height)));
  still<<<grid, block, 0, stream>>>(params);
  return cudaGetLastError();
}

cudaError_t launch_soft_pull(const PullParams& params, cudaStream_t stream) {
  if (params.width < 1 || params.height < 1 || params.channels < 1) return cudaErrorInvalidValue;
  const dim3 block(kBlock, kBlock);
  const dim3 grid(static_cast<unsigned>(tiles_across(params.width)),
                  static_cast<unsigned>(tiles_across(params.height)));
  pull<<<grid, block, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace kirari::gpu::cuda
