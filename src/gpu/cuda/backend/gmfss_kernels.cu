// Noyaux du softmax splatting de GMFSS (voir gmfss_kernels.h). Même calcul que
// kernels/glsl/gmfss_splat.comp, gmfss_normalize.comp et gmfss_compose.comp ; référence
// kernels/ref/gmfss.cpp. Un lot : gridDim.z est le travail, chaque bloc une tuile 16 × 16 de sa
// source ; les tuiles hors d'un travail plus petit que le plus grand sortent aussitôt.
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/bilinear.cuh"
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

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) splat_all(SoftSplatBatch b) {
  __shared__ float cells[(kPass + 1) * kCells];
  const SoftSplatParams& p = b.jobs[blockIdx.z];
  const int x0 = static_cast<int>(blockIdx.x) * kBlock;
  const int y0 = static_cast<int>(blockIdx.y) * kBlock;
  if (x0 >= p.width || y0 >= p.height) return;  // tout le bloc
  const int t = static_cast<int>(threadIdx.y) * kBlock + static_cast<int>(threadIdx.x);
  const int x = x0 + static_cast<int>(threadIdx.x);
  const int y = y0 + static_cast<int>(threadIdx.y);
  const bool pixel = x < p.width && y < p.height;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = pixel ? at(x, y, p.width) : 0;
  float fx = 0.0F;
  float fy = 0.0F;
  float z = 0.0F;
  if (pixel) {
    const auto* motion = static_cast<const __half*>(p.motion);
    if (p.direct != 0) {
      fx = __half2float(motion[i]);
      fy = __half2float(motion[plane + i]);
      z = __half2float(motion[2 * plane + i]);
    } else {
      const float u = (static_cast<float>(x) + 0.5F) / p.motion_scale - 0.5F + p.motion_x;
      const float v = (static_cast<float>(y) + 0.5F) / p.motion_scale - 0.5F + p.motion_y;
      const Planes planes{.data = motion,
                          .plane_w = p.motion_width,
                          .plane_h = p.motion_height,
                          .fw = p.motion_width,
                          .fh = p.motion_height};
      fx = p.motion_scale * Exact::sample(planes, 0, u, v);
      fy = p.motion_scale * Exact::sample(planes, 1, u, v);
      z = Exact::sample(planes, 2, u, v);
    }
  }
  const float tx = static_cast<float>(x) + p.s * fx;
  const float ty = static_cast<float>(y) + p.s * fy;
  // Hors de ]−1, taille[, aucun coin n'est dans le plan ; NaN et infinis échouent aussi.
  const bool active = pixel && tx > -1.0F && tx < static_cast<float>(p.width) && ty > -1.0F &&
                      ty < static_cast<float>(p.height);
  const float3 shift =
      block_sum3(active ? make_float3(tx - static_cast<float>(x), ty - static_cast<float>(y), 1.0F)
                        : make_float3(0.0F, 0.0F, 0.0F));
  if (shift.z == 0.0F) return;  // tout le bloc : rien n'arrive dans le plan
  const int ox = x0 - kMargin + __float2int_rn(shift.x / shift.z);
  const int oy = y0 - kMargin + __float2int_rn(shift.y / shift.z);
  // z borné comme dans le port GLSL (NaN : le bas de l'intervalle, comme fmaxf).
  const float bounded_z = fminf(fmaxf(z, -kernels::kGmfssMetricLimit), kernels::kGmfssMetricLimit);
  const float scale = active ? expf(p.s * bounded_z) : 0.0F;  // une fois par pixel, pour tous les plans
  const int cx0 = active ? static_cast<int>(floorf(tx)) : 0;
  const int cy0 = active ? static_cast<int>(floorf(ty)) : 0;
  const float east = tx - static_cast<float>(cx0);
  const float west = static_cast<float>(cx0 + 1) - tx;
  const float south = ty - static_cast<float>(cy0);
  const float north = static_cast<float>(cy0 + 1) - ty;
  const int cx[4] = {cx0, cx0 + 1, cx0, cx0 + 1};
  const int cy[4] = {cy0, cy0, cy0 + 1, cy0 + 1};
  const float share[4] = {west * north, east * north, west * south, east * south};
  float* corner[4];  // dans le premier plan des sommes globales
  int cell[4];       // dans la fenêtre partagée, ou −1
  float weight[4];
  unsigned inside = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const bool in = active && cx[k] >= 0 && cx[k] < p.width && cy[k] >= 0 && cy[k] < p.height;
    inside |= in ? 1U << k : 0U;
    weight[k] = scale * share[k];
    corner[k] = p.sums + (in ? at(cx[k], cy[k], p.width) : 0);
    const int lx = cx[k] - ox;
    const int ly = cy[k] - oy;
    cell[k] = lx >= 0 && lx < kSpan && ly >= 0 && ly < kSpan ? ly * kSpan + lx : -1;
  }
  const __half* from = static_cast<const __half*>(p.source) + i;
  const std::size_t weights = static_cast<std::size_t>(p.channels) * plane;
#pragma unroll 1
  for (int first = 0; first < p.channels; first += kPass) {
    const int count = min(kPass, p.channels - first);
    const bool with_weights = first == 0;
    for (int k = t; k < kPass * kCells; k += kPixelThreads) cells[k] = 0.0F;
    if (with_weights) {
      for (int k = t; k < kCells; k += kPixelThreads) cells[kPass * kCells + k] = 0.0F;
    }
    __syncthreads();
    if (inside != 0) {
#pragma unroll 1
      for (int c = 0; c < count; ++c) {
        const std::size_t global = static_cast<std::size_t>(first + c) * plane;
        const float value = __half2float(from[global]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          if ((inside >> k & 1U) == 0) continue;
          if (cell[k] >= 0) {
            atomicAdd(&cells[c * kCells + cell[k]], value * weight[k]);
          } else {
            atomicAdd(corner[k] + global, value * weight[k]);
          }
        }
      }
      if (with_weights) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          if ((inside >> k & 1U) == 0) continue;
          if (cell[k] >= 0) {
            atomicAdd(&cells[kPass * kCells + cell[k]], weight[k]);
          } else {
            atomicAdd(corner[k] + weights, weight[k]);
          }
        }
      }
    }
    __syncthreads();
    // La fenêtre vers les sommes globales : une addition par cellule touchée et par plan.
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
      if (with_weights) {
        const float v = cells[kPass * kCells + k];
        if (v != 0.0F) atomicAdd(sum + weights, v);
      }
    }
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
    if (p.coverage != 0)
      out[(static_cast<std::size_t>(p.channels) * plane + i) / 2] =
          __floats2half2_rn(weight.x / total.x, weight.y / total.y);
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

// Les plans lus en (xw, yw) par la composition : sources de travail, et fusion dans sa fenêtre.
struct ComposeReads {
  Planes low_a;
  Planes low_b;
  Planes fusion;
};

__device__ ComposeReads compose_reads(const GmfssComposeParams& p) {
  const auto low = [&](const void* data, cudaTextureObject_t texture) {
    return Planes{.data = static_cast<const __half*>(data),
                  .texture = texture,
                  .plane_w = p.low_width,
                  .plane_h = p.low_height,
                  .fw = p.low_width,
                  .fh = p.low_height};
  };
  return ComposeReads{.low_a = low(p.low_a, p.textures.low_a),
                      .low_b = low(p.low_b, p.textures.low_b),
                      .fusion = Planes{.data = static_cast<const __half*>(p.fusion),
                                       .texture = p.textures.fusion,
                                       .plane_w = p.fusion_width,
                                       .plane_h = p.fusion_height,
                                       .ox = p.window_x,
                                       .oy = p.window_y,
                                       .fw = p.low_width,
                                       .fh = p.low_height}};
}

__device__ float mix(float t, float p0, float c0, float p1, float c1) {
  const float w0 = (1.0F - t) * c0;
  const float w1 = t * c1;
  return (w0 * p0 + w1 * p1) / (w0 + w1 + kernels::kGmfssNormalizeEpsilon);
}

// high : les quatre plans (R, G, B, couverture) des deux sources agrandies au pixel ; rgb : sortie.
template <typename S>
__device__ void compose_pixel(const GmfssComposeParams& p, const ComposeReads& r, int x, int y,
                              const float (&high_a)[4], const float (&high_b)[4], float* rgb) {
  const float scale = static_cast<float>(p.scale);
  const float xw = (static_cast<float>(x) + 0.5F) / scale - 0.5F;
  const float yw = (static_cast<float>(y) + 0.5F) / scale - 0.5F;
  const float la = S::sample(r.low_a, 3, xw, yw);
  const float lb = S::sample(r.low_b, 3, xw, yw);
  for (int c = 0; c < 3; ++c) {
    const float high = mix(p.t, high_a[c], high_a[3], high_b[c], high_b[3]);
    const float lo = mix(p.t, S::sample(r.low_a, c, xw, yw), la, S::sample(r.low_b, c, xw, yw), lb);
    const float fused = S::sample(r.fusion, c, xw, yw);
    rgb[c] = fminf(fmaxf(high + fused - lo, 0.0F), 1.0F);
  }
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) compose(GmfssComposeParams p) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width);
  const auto* high_a = static_cast<const __half*>(p.high_a);
  const auto* high_b = static_cast<const __half*>(p.high_b);
  float ha[4];
  float hb[4];
  for (int c = 0; c < 4; ++c) {
    ha[c] = __half2float(high_a[static_cast<std::size_t>(c) * plane + i]);
    hb[c] = __half2float(high_b[static_cast<std::size_t>(c) * plane + i]);
  }
  float rgb[3];
  compose_pixel<Exact>(p, compose_reads(p), x, y, ha, hb, rgb);
  if (p.surface != 0) {
    store_rgba16(p.surface, x, y, __float2half_rn(rgb[0]), __float2half_rn(rgb[1]), __float2half_rn(rgb[2]));
    return;
  }
  auto* out = static_cast<__half*>(p.out);
  for (int c = 0; c < 3; ++c) out[static_cast<std::size_t>(c) * plane + i] = __float2half_rn(rgb[c]);
}

// Deux pixels voisins par fil : sources agrandies lues et sortie écrite en half2, sources de
// travail et fusion par les objets texture. À l'échelle 2, la fraction vaut 1/4 ou 3/4 : le
// filtre matériel la rend exactement.
__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) compose_filtered(GmfssComposeParams p) {
  const int x = 2 * static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= p.width || y >= p.height) return;  // largeur paire : x + 1 est dans l'image
  const std::size_t plane = static_cast<std::size_t>(p.width) / 2 * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width) / 2;
  const auto* high_a = static_cast<const __half2*>(p.high_a);
  const auto* high_b = static_cast<const __half2*>(p.high_b);
  float ha[2][4];
  float hb[2][4];
  for (int c = 0; c < 4; ++c) {
    const float2 va = __half22float2(high_a[static_cast<std::size_t>(c) * plane + i]);
    const float2 vb = __half22float2(high_b[static_cast<std::size_t>(c) * plane + i]);
    ha[0][c] = va.x;
    ha[1][c] = va.y;
    hb[0][c] = vb.x;
    hb[1][c] = vb.y;
  }
  const ComposeReads reads = compose_reads(p);
  float left[3];
  float right[3];
  compose_pixel<Filtered>(p, reads, x, y, ha[0], hb[0], left);
  compose_pixel<Filtered>(p, reads, x + 1, y, ha[1], hb[1], right);
  if (p.surface != 0) {
    store_rgba16(p.surface, x, y, __float2half_rn(left[0]), __float2half_rn(left[1]),
                 __float2half_rn(left[2]));
    store_rgba16(p.surface, x + 1, y, __float2half_rn(right[0]), __float2half_rn(right[1]),
                 __float2half_rn(right[2]));
    return;
  }
  auto* out = static_cast<__half2*>(p.out);
  for (int c = 0; c < 3; ++c)
    out[static_cast<std::size_t>(c) * plane + i] = __floats2half2_rn(left[c], right[c]);
}

__global__ void fill_half(__half* data, std::size_t count, __half value) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) data[i] = value;
}

}  // namespace

std::size_t soft_splat_sums_bytes(int width, int height, int channels) {
  return static_cast<std::size_t>(channels + 1) * static_cast<std::size_t>(width) *
         static_cast<std::size_t>(height) * sizeof(float);
}

cudaError_t launch_fill(void* data, std::size_t count, float value, cudaStream_t stream) {
  constexpr unsigned kThreads = 64;
  const auto blocks = static_cast<unsigned>((count + kThreads - 1) / kThreads);
  fill_half<<<blocks, kThreads, 0, stream>>>(static_cast<__half*>(data), count, __float2half_rn(value));
  return cudaGetLastError();
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

cudaError_t launch_gmfss_compose(const GmfssComposeParams& params, cudaStream_t stream) {
  const dim3 block(kBlock, kBlock);
  const GmfssComposeParams::Textures& t = params.textures;
  const auto aligned = [](const void* data) { return reinterpret_cast<std::uintptr_t>(data) % 4 == 0; };
  if (t.low_a != 0 && t.low_b != 0 && t.fusion != 0 && params.width % 2 == 0 && aligned(params.high_a) &&
      aligned(params.high_b) && aligned(params.out)) {
    const dim3 grid(static_cast<unsigned>(tiles_across(params.width / 2)),
                    static_cast<unsigned>(tiles_across(params.height)));
    compose_filtered<<<grid, block, 0, stream>>>(params);
  } else {
    const dim3 grid(static_cast<unsigned>(tiles_across(params.width)),
                    static_cast<unsigned>(tiles_across(params.height)));
    compose<<<grid, block, 0, stream>>>(params);
  }
  return cudaGetLastError();
}

}  // namespace ajp::gpu::cuda
