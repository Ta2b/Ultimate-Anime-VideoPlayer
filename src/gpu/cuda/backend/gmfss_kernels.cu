// Noyaux du softmax splatting de GMFSS (voir gmfss_kernels.h) ; référence kernels/ref/gmfss.cpp. Un lot :
// gridDim.z est le travail, chaque bloc une tuile 16 × 16 de sa source ; les tuiles hors d'un travail plus
// petit que le plus grand sortent aussitôt.
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/gmfss_kernels.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "kernels/spec/gmfss.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;
static_assert(kBlock * kBlock == kPixelThreads);
// Splat tuilé (étude 3) : la tuile pousse ses contributions dans une fenêtre partagée de
// kSpan × kSpan cellules, placée sur le déplacement moyen de la tuile (un panoramique y reste
// entier) avec kMargin cellules de jeu de chaque côté ; une contribution hors de la fenêtre va
// directement aux sommes globales. kPass plans par passe, plus celui des poids à la première :
// 20 Ko de mémoire partagée, quatre blocs par SM.
constexpr int kMargin = 8;
constexpr int kSpan = kBlock + 2 * kMargin;
constexpr int kCells = kSpan * kSpan;
constexpr int kPass = 4;

__host__ __device__ int tiles_across(int size) {
  return (size + kBlock - 1) / kBlock;
}

__device__ std::size_t at(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

// Somme sur le bloc de trois valeurs, rendue à tous les fils.
__device__ float3 block_sum3(float3 v) {
  __shared__ float3 total;
  const int t = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  if (t == 0) total = make_float3(0.0F, 0.0F, 0.0F);
  __syncthreads();
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    v.x += __shfl_down_sync(0xFFFFFFFFU, v.x, offset);
    v.y += __shfl_down_sync(0xFFFFFFFFU, v.y, offset);
    v.z += __shfl_down_sync(0xFFFFFFFFU, v.z, offset);
  }
  if (t % kWarpSize == 0) {
    atomicAdd(&total.x, v.x);
    atomicAdd(&total.y, v.y);
    atomicAdd(&total.z, v.z);
  }
  __syncthreads();
  return total;
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
                                                      int oy) {
  SplatCorners q;
  // z borné comme la référence (NaN : le bas de l'intervalle, comme fmaxf).
  const float bounded_z = fminf(fmaxf(s.z, -kernels::kGmfssMetricLimit), kernels::kGmfssMetricLimit);
  const float scale = s.active ? expf(p.s * bounded_z) : 0.0F;  // une fois par pixel, pour tous les plans
  const int cx0 = s.active ? static_cast<int>(floorf(s.tx)) : 0;
  const int cy0 = s.active ? static_cast<int>(floorf(s.ty)) : 0;
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
  const std::size_t weights = static_cast<std::size_t>(p.channels) * plane;
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
  const float3 shift =
      block_sum3(s.active ? make_float3(s.tx - static_cast<float>(x), s.ty - static_cast<float>(y), 1.0F)
                          : make_float3(0.0F, 0.0F, 0.0F));
  if (shift.z == 0.0F) return;  // tout le bloc : rien n'arrive dans le plan
  const int ox = x0 - kMargin + __float2int_rn(shift.x / shift.z);
  const int oy = y0 - kMargin + __float2int_rn(shift.y / shift.z);
  const SplatCorners q = splat_corners(p, s, ox, oy);
  const __half* from = static_cast<const __half*>(p.source) + s.i;
  const std::size_t weights = static_cast<std::size_t>(p.channels) * plane;
#pragma unroll 1
  for (int first = 0; first < p.channels; first += kPass) {
    const int count = min(kPass, p.channels - first);
    const bool with_weights = first == 0;
    const int cleared = with_weights ? (kPass + 1) * kCells : kPass * kCells;
    for (int k = t; k < cleared; k += kPixelThreads) cells[k] = 0.0F;
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
    flush_window(p, cells, ox, oy, first, count, with_weights, plane, t);
    __syncthreads();
  }
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
  const int x = 2 * static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width);
  const float* weights = p.sums + static_cast<std::size_t>(p.channels) * plane;
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
  for (int j = 0; j < batch.count; ++j) {
    SoftSplatParams& p = batch.jobs[j];
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
  splat_all<<<dim3(static_cast<unsigned>(tiles_across(width)), rows, jobs), block, 0, stream>>>(batch);
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  normalize_all<<<dim3(static_cast<unsigned>(tiles_across((width + 1) / 2)), rows, jobs), block, 0, stream>>>(
      batch);
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
