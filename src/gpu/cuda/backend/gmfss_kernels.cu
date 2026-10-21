// Noyaux du softmax splatting de GMFSS (voir gmfss_kernels.h). Même calcul que
// kernels/glsl/gmfss_splat.comp, gmfss_normalize.comp et gmfss_compose.comp ; référence
// kernels/ref/gmfss.cpp. Un lot : la grille est linéaire, chaque bloc retrouve son travail par les
// préfixes de la table, puis sa tuile 16 × 16 et son groupe de plans ; aucun bloc n'est lancé pour rien.
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/bilinear.cuh"
#include "gpu/cuda/backend/gmfss_kernels.h"
#include "gpu/cuda/backend/pixel_kernels.cuh"
#include "kernels/spec/gmfss.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kBlock = 16;
// Plans d'un groupe de l'addition : la géométrie d'un pixel se refait par groupe.
constexpr int kGroup = 16;

__host__ __device__ int tiles_across(int size) {
  return (size + kBlock - 1) / kBlock;
}

__host__ __device__ int groups_of(int channels) {
  return (channels + kGroup - 1) / kGroup;
}

__device__ std::size_t at(int x, int y, int width) {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
}

// Le travail du bloc et le rang du bloc dans ce travail.
__device__ int job_of(const unsigned* first, int count, unsigned block, unsigned& local) {
  int j = 0;
  while (j + 1 < count && block >= first[j + 1]) ++j;
  local = block - first[j];
  return j;
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) splat_all(SoftSplatBatch b) {
  unsigned local = 0;
  const SoftSplatParams& p = b.jobs[job_of(b.splat_blocks, b.count, blockIdx.x, local)];
  const int across = tiles_across(p.width);
  const int tiles = across * tiles_across(p.height);
  const int group = static_cast<int>(local) / tiles;
  const int tile = static_cast<int>(local) % tiles;
  const int x = (tile % across) * kBlock + static_cast<int>(threadIdx.x);
  const int y = (tile / across) * kBlock + static_cast<int>(threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width);
  const auto* motion = static_cast<const __half*>(p.motion);
  float fx = 0.0F;
  float fy = 0.0F;
  float z = 0.0F;
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
  const float tx = static_cast<float>(x) + p.s * fx;
  const float ty = static_cast<float>(y) + p.s * fy;
  // Hors de ]−1, taille[, aucun coin n'est dans le plan ; NaN et infinis échouent aussi.
  if (!(tx > -1.0F && tx < static_cast<float>(p.width) && ty > -1.0F && ty < static_cast<float>(p.height)))
    return;
  const float scale = expf(p.s * z);
  const int x0 = static_cast<int>(floorf(tx));
  const int y0 = static_cast<int>(floorf(ty));
  const float east = tx - static_cast<float>(x0);
  const float west = static_cast<float>(x0 + 1) - tx;
  const float south = ty - static_cast<float>(y0);
  const float north = static_cast<float>(y0 + 1) - ty;
  const int cx[4] = {x0, x0 + 1, x0, x0 + 1};
  const int cy[4] = {y0, y0, y0 + 1, y0 + 1};
  const float share[4] = {west * north, east * north, west * south, east * south};
  const int first = group * kGroup;
  const int last = min(p.channels, first + kGroup);
  // Chaque plan de source se lit une fois pour les quatre coins (au lieu d'une fois par coin), et
  // les adresses avancent d'un plan : sm_120 gardait sinon 32 adresses 64 bits et débordait.
  float* corner[4];
  float weight[4];
  unsigned inside = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const bool in = cx[k] >= 0 && cx[k] < p.width && cy[k] >= 0 && cy[k] < p.height;
    inside |= in ? 1U << k : 0U;
    weight[k] = scale * share[k];
    corner[k] = p.sums + static_cast<std::size_t>(first) * plane + (in ? at(cx[k], cy[k], p.width) : 0);
  }
  const __half* from = static_cast<const __half*>(p.source) + static_cast<std::size_t>(first) * plane + i;
  for (int c = first; c < last; ++c) {
    const float value = __half2float(*from);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if ((inside >> k & 1U) != 0) atomicAdd(corner[k], value * weight[k]);
      corner[k] += plane;
    }
    from += plane;
  }
  if (group != 0) return;
  // Après la boucle, corner[k] pointe sur le plan last ; celui des poids est p.channels.
  const std::size_t weights = static_cast<std::size_t>(p.channels - last) * plane;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    if ((inside >> k & 1U) != 0) atomicAdd(corner[k] + weights, weight[k]);
  }
}

__global__ void __launch_bounds__(kPixelThreads, kPixelBlocksPerSm) normalize_all(SoftSplatBatch b) {
  unsigned local = 0;
  const SoftSplatParams& p = b.jobs[job_of(b.normalize_blocks, b.count, blockIdx.x, local)];
  const int across = tiles_across(p.width);
  const int tile = static_cast<int>(local);
  const int x = (tile % across) * kBlock + static_cast<int>(threadIdx.x);
  const int y = (tile / across) * kBlock + static_cast<int>(threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width);
  const float weight = p.sums[static_cast<std::size_t>(p.channels) * plane + i];
  const float total = weight + kernels::kGmfssNormalizeEpsilon;
  auto* out = static_cast<__half*>(p.out);
  for (int c = 0; c < p.channels; ++c) {
    const std::size_t k = static_cast<std::size_t>(c) * plane + i;
    out[k] = __float2half_rn(p.sums[k] / total);
  }
  if (p.coverage != 0)
    out[static_cast<std::size_t>(p.channels) * plane + i] = __float2half_rn(weight / total);
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
  unsigned splat = 0;
  unsigned normalize = 0;
  for (int j = 0; j < batch.count; ++j) {
    const SoftSplatParams& p = batch.jobs[j];
    const auto tiles = static_cast<unsigned>(tiles_across(p.width) * tiles_across(p.height));
    batch.splat_blocks[j] = splat;
    batch.normalize_blocks[j] = normalize;
    splat += tiles * static_cast<unsigned>(groups_of(p.channels));
    normalize += tiles;
  }
  batch.splat_blocks[batch.count] = splat;
  batch.normalize_blocks[batch.count] = normalize;
  const cudaError_t cleared = cudaMemsetAsync(sums, 0, sums_bytes, stream);
  if (cleared != cudaSuccess) return cleared;
  const dim3 block(kBlock, kBlock);
  splat_all<<<splat, block, 0, stream>>>(batch);
  if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
  normalize_all<<<normalize, block, 0, stream>>>(batch);
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
