// Noyaux du softmax splatting de GMFSS (voir gmfss_kernels.h). Même calcul que
// kernels/glsl/gmfss_splat.comp et gmfss_normalize.comp ; référence kernels/ref/gmfss.cpp.
// Un lot : la grille est linéaire, chaque bloc retrouve son travail par les préfixes de la table,
// puis sa tuile 16 × 16 et son groupe de plans ; aucun bloc n'est lancé pour rien.
#include <cuda_fp16.h>

#include "gpu/cuda/backend/gmfss_kernels.h"
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

__global__ void splat_all(SoftSplatBatch b) {
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
  const float tx = static_cast<float>(x) + p.s * __half2float(motion[i]);
  const float ty = static_cast<float>(y) + p.s * __half2float(motion[plane + i]);
  // Hors de ]−1, taille[, aucun coin n'est dans le plan ; NaN et infinis échouent aussi.
  if (!(tx > -1.0F && tx < static_cast<float>(p.width) && ty > -1.0F && ty < static_cast<float>(p.height)))
    return;
  const float scale = expf(p.s * __half2float(motion[2 * plane + i]));
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
  const auto* source = static_cast<const __half*>(p.source);
  for (int k = 0; k < 4; ++k) {
    if (cx[k] < 0 || cx[k] >= p.width || cy[k] < 0 || cy[k] >= p.height) continue;
    const std::size_t j = at(cx[k], cy[k], p.width);
    const float weight = scale * share[k];
    for (int c = first; c < last; ++c) {
      atomicAdd(&p.sums[static_cast<std::size_t>(c) * plane + j],
                __half2float(source[static_cast<std::size_t>(c) * plane + i]) * weight);
    }
    if (group == 0) atomicAdd(&p.sums[static_cast<std::size_t>(p.channels) * plane + j], weight);
  }
}

__global__ void normalize_all(SoftSplatBatch b) {
  unsigned local = 0;
  const SoftSplatParams& p = b.jobs[job_of(b.normalize_blocks, b.count, blockIdx.x, local)];
  const int across = tiles_across(p.width);
  const int tile = static_cast<int>(local);
  const int x = (tile % across) * kBlock + static_cast<int>(threadIdx.x);
  const int y = (tile / across) * kBlock + static_cast<int>(threadIdx.y);
  if (x >= p.width || y >= p.height) return;
  const std::size_t plane = static_cast<std::size_t>(p.width) * static_cast<std::size_t>(p.height);
  const std::size_t i = at(x, y, p.width);
  const float total =
      p.sums[static_cast<std::size_t>(p.channels) * plane + i] + kernels::kGmfssNormalizeEpsilon;
  auto* out = static_cast<__half*>(p.out);
  for (int c = 0; c < p.channels; ++c) {
    const std::size_t k = static_cast<std::size_t>(c) * plane + i;
    out[k] = __float2half_rn(p.sums[k] / total);
  }
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

}  // namespace ajp::gpu::cuda
