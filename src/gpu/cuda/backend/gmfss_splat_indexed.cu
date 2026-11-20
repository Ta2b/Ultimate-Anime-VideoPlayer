// Le splat rassemble les contributions par destination. Les mêmes poids et indices servent à tous ses
// canaux. Les accumulations et la normalisation restent en FP32, avec un seul arrondi FP16 en sortie.
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>

#include "gpu/cuda/backend/gmfss_kernels.h"
#include "kernels/spec/gmfss.h"

namespace kirari::gpu::cuda {
namespace {

constexpr unsigned kSlots = 12;
constexpr unsigned kDenseThreshold = 128;
constexpr int kDensePlanes = kMaxIndexedSplatChannels + 1;
constexpr std::size_t kMaxPixels = 960U * 576U;

std::size_t dense_capacity(std::size_t pixels) {
  // Au plus quatre coins par source et au moins 129 contributions par destination dense.
  return 4 * pixels / (kDenseThreshold + 1);
}

struct Entry {
  int index;
  float weight;
};

struct SplatIndex {
  unsigned* counts;
  int* heads;
  Entry* entries;
  Entry* overflow;
  unsigned* dense_count;
  unsigned* dense_ids;
  float* dense_sums;
};

SplatIndex index_at(void* scratch, std::size_t pixels) {
  auto* counts = static_cast<unsigned*>(scratch);
  auto* heads = reinterpret_cast<int*>(counts + pixels + 1);
  auto* entries = reinterpret_cast<Entry*>(heads + pixels);
  auto* overflow = entries + kSlots * pixels;
  auto* dense_ids = reinterpret_cast<unsigned*>(overflow + 4 * pixels);
  auto* dense_sums = reinterpret_cast<float*>(dense_ids + pixels);
  return {counts, heads, entries, overflow, counts + pixels, dense_ids, dense_sums};
}

__device__ unsigned reserve_slot(unsigned* counter, int dest) {
#if __CUDA_ARCH__ >= 700
  const auto group = __match_any_sync(__activemask(), dest);
  const auto lanes = __popc(group);
  if (lanes > 1) {
    unsigned base = 0;
    const auto leader = __ffs(group) - 1;
    const unsigned lane = threadIdx.x % 32;
    if (static_cast<int>(lane) == leader) base = atomicAdd(counter, static_cast<unsigned>(lanes));
    return __shfl_sync(group, base, leader) + __popc(group & ((1U << lane) - 1U));
  }
#endif
  return atomicAdd(counter, 1U);
}

struct Projection {
  int left, top;
  float west, east, north, south, factor;

  __device__ float weight(int dx, int dy) const {
    return factor * ((dx == 0 ? west : east) * (dy == 0 ? north : south));
  }
};

__device__ bool project(SoftSplatParams p, int src, int pixels, Projection& q) {
  const auto* motion = static_cast<const __half*>(p.motion);
  const int y = src / p.width, x = src - y * p.width;
  const float tx = static_cast<float>(x) + p.s * __half2float(motion[src]);
  const float ty = static_cast<float>(y) + p.s * __half2float(motion[pixels + src]);
  if (!(tx > -1.F && ty > -1.F && tx < static_cast<float>(p.width) && ty < static_cast<float>(p.height)))
    return false;
  q.left = __float2int_rd(tx);
  q.top = __float2int_rd(ty);
  q.east = tx - static_cast<float>(q.left);
  q.south = ty - static_cast<float>(q.top);
  q.west = static_cast<float>(q.left + 1) - tx;
  q.north = static_cast<float>(q.top + 1) - ty;
  const float z = __half2float(motion[2 * pixels + src]);
  q.factor = expf(p.s * fminf(fmaxf(z, -kernels::kGmfssMetricLimit), kernels::kGmfssMetricLimit));
  return true;
}

__global__ void index_splat(SoftSplatParams p, SplatIndex index) {
  const int src = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int pixels = p.width * p.height;
  if (src >= pixels) return;
  Projection q;
  if (!project(p, src, pixels, q)) return;
#pragma unroll
  for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
    for (int dx = 0; dx < 2; ++dx) {
      const int yy = q.top + dy, xx = q.left + dx;
      if (xx < 0 || xx >= p.width || yy < 0 || yy >= p.height) continue;
      const int dest = yy * p.width + xx;
      const float weight = q.weight(dx, dy);
      const unsigned slot = reserve_slot(index.counts + dest, dest);
      if (slot < kSlots) {
        index.entries[static_cast<std::size_t>(slot) * pixels + dest] = {src, weight};
      } else if (slot < kDenseThreshold) {
        // Un identifiant par coin garantit que les producteurs écrivent des entrées distinctes.
        // Le rassemblement lit la liste après la fin du noyau.
        const int node = 4 * src + 2 * dy + dx;
        const int previous = atomicExch(index.heads + dest, node);
        index.overflow[node] = {previous, weight};
      } else if (slot == kDenseThreshold) {
        const unsigned id = atomicAdd(index.dense_count, 1U);
        index.dense_ids[dest] = id;
        float* sums = index.dense_sums + id * kDensePlanes;
        for (int c = 0; c < kDensePlanes; ++c) sums[c] = 0.F;
      }
    }
  }
}

__device__ void dense_add(float* value, float source, bool whole_warp) {
  if (whole_warp) {
    for (int d = 16; d > 0; d /= 2) source += __shfl_down_sync(0xffffffffU, source, d);
    if (threadIdx.x % 32 != 0) return;
  }
  atomicAdd(value, source);
}

// Les scènes normales sortent sur un drapeau produit sur le GPU. Le nombre de blocs est borné pour
// limiter le coût de ce lancement. Sur une convergence massive, les additions atomiques globales
// sont réduites à une par canal, par warp et par destination.
__global__ void scatter_dense(SoftSplatParams p, SplatIndex index, bool write_weights) {
  if (*index.dense_count == 0) return;
  const int pixels = p.width * p.height;
  for (int src = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); src < pixels;
       src += static_cast<int>(gridDim.x * blockDim.x)) {
    Projection q;
    if (!project(p, src, pixels, q)) continue;
    const auto* source = static_cast<const __half*>(p.source) + src;
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
      for (int dx = 0; dx < 2; ++dx) {
        const int yy = q.top + dy, xx = q.left + dx;
        if (xx < 0 || xx >= p.width || yy < 0 || yy >= p.height) continue;
        const int dest = yy * p.width + xx;
        if (index.counts[dest] <= kDenseThreshold) continue;
        const auto id = index.dense_ids[dest];
        const auto active = __activemask();
        const bool whole_warp = active == 0xffffffffU && __all_sync(active, id == __shfl_sync(active, id, 0));
        float* sums = index.dense_sums + id * kDensePlanes;
        const float weight = q.weight(dx, dy);
        if (write_weights) dense_add(sums, weight, whole_warp);
        for (int c = 0; c < p.channels; ++c)
          dense_add(sums + c + 1, __fmul_rn(__half2float(source[c * pixels]), weight), whole_warp);
      }
    }
  }
}

enum class Order { Planar, Pixel, Group8 };

template <Order Layout, int Chunk>
__device__ void store_sums(const SoftSplatParams& p, int channel, int dest, int pixels,
                           const float (&numerator)[Chunk], float denominator) {
  auto* out = static_cast<__half*>(p.out);
  if constexpr (Layout == Order::Group8 && Chunk == 8) {
    const int n = channel + p.out_first;
    const auto pair = [&](int i) {
      const __half2_raw word = __floats2half2_rn(numerator[i] / denominator, numerator[i + 1] / denominator);
      return static_cast<unsigned>(word.x) | (static_cast<unsigned>(word.y) << 16U);
    };
    const auto at = static_cast<std::size_t>(n) * pixels + dest * 8;
    *reinterpret_cast<uint4*>(out + at) = make_uint4(pair(0), pair(2), pair(4), pair(6));
  } else {
#pragma unroll
    for (int k = 0; k < Chunk; ++k) {
      const int c = channel + k;
      if (c >= p.channels) break;
      std::size_t at;
      if constexpr (Layout == Order::Planar) {
        at = static_cast<std::size_t>(c) * pixels + dest;
      } else if constexpr (Layout == Order::Pixel) {
        at = static_cast<std::size_t>(dest) * p.out_pitch + c;
      } else {
        const int n = c + p.out_first;
        at = static_cast<std::size_t>(n / 8) * 8 * pixels + dest * 8 + n % 8;
      }
      out[at] = __float2half_rn(numerator[k] / denominator);
    }
  }
}

template <Order Layout, int Chunk>
__global__ void gather_splat(SoftSplatParams p, SplatIndex index) {
  const int dest = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int channel = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) * Chunk;
  const int pixels = p.width * p.height;
  if (dest >= pixels || channel >= p.channels) return;
  const unsigned count = index.counts[dest];
  float numerator[Chunk]{};
  float weight_sum = 0.F;
  const auto* source = static_cast<const __half*>(p.source) + channel * pixels;
  const auto add = [&](int src, float weight) {
#pragma unroll
    for (int c = 0; c < Chunk; ++c) {
      if (channel + c < p.channels)
        numerator[c] = __fadd_rn(numerator[c], __fmul_rn(__half2float(source[c * pixels + src]), weight));
    }
    weight_sum += weight;
  };
  if (count > kDenseThreshold) {
    float* sums = index.dense_sums + index.dense_ids[dest] * kDensePlanes;
    weight_sum = sums[0];
#pragma unroll
    for (int c = 0; c < Chunk; ++c) {
      if (channel + c < p.channels) {
        numerator[c] = sums[channel + c + 1];
        sums[channel + c + 1] = 0.F;
      }
    }
  } else {
    for (unsigned slot = 0; slot < min(count, kSlots); ++slot) {
      const Entry item = index.entries[static_cast<std::size_t>(slot) * pixels + dest];
      add(item.index, item.weight);
    }
    if (count > kSlots) {
      for (int node = index.heads[dest]; node >= 0;) {
        const Entry item = index.overflow[node];
        add(node / 4, item.weight);
        node = item.index;
      }
    }
  }
  const float denominator = weight_sum + kernels::kGmfssNormalizeEpsilon;
  store_sums<Layout>(p, channel, dest, pixels, numerator, denominator);
  if constexpr (Layout == Order::Planar) {
    if (p.coverage != 0 && channel == 0)
      static_cast<__half*>(p.out)[static_cast<std::size_t>(p.channels) * pixels + dest] =
          __float2half_rn(weight_sum / denominator);
  }
}

__global__ void fill_instant(__half* out, std::size_t count, float value) {
  const std::size_t at = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (at < count) out[at] = __float2half_rn(value);
}

template <Order Layout>
void launch_gather(const SoftSplatParams& p, SplatIndex index, std::size_t pixels, cudaStream_t stream) {
  const dim3 threads(32, 4);
  const unsigned columns = static_cast<unsigned>((pixels + 31) / 32);
  if constexpr (Layout == Order::Group8) {
    if (p.channels > 64 && p.out_first % 8 == 0 && p.channels % 8 == 0 &&
        reinterpret_cast<std::uintptr_t>(p.out) % alignof(uint4) == 0) {
      const dim3 grid(columns, static_cast<unsigned>((p.channels + 31) / 32));
      gather_splat<Layout, 8><<<grid, threads, 0, stream>>>(p, index);
      return;
    }
  }
  if (p.channels <= 64) {
    const dim3 grid(columns, static_cast<unsigned>((p.channels + 7) / 8));
    gather_splat<Layout, 2><<<grid, threads, 0, stream>>>(p, index);
  } else {
    const dim3 grid(columns, static_cast<unsigned>((p.channels + 15) / 16));
    gather_splat<Layout, 4><<<grid, threads, 0, stream>>>(p, index);
  }
}

bool valid_job(const SoftSplatParams& p, std::size_t bytes) {
  const auto required = indexed_splat_bytes(p.width, p.height);
  if (required == 0 || required > bytes || p.channels <= 0 || p.channels > kMaxIndexedSplatChannels ||
      p.source == nullptr || p.motion == nullptr || p.out == nullptr || p.reuse_weights ||
      (p.out_pitch != 0 && p.coverage != 0))
    return false;
  if (p.out_pitch >= 0) return p.out_pitch == 0 || p.out_pitch >= p.channels;
  return p.out_pitch % 16 == 0 && p.out_first >= 0 && p.out_first <= -(p.out_pitch / 2) - p.channels;
}

bool same_projection(const SoftSplatParams& a, const SoftSplatParams& b) {
  return a.motion == b.motion && a.width == b.width && a.height == b.height && a.s == b.s;
}

cudaError_t enqueue_index(const SoftSplatParams& p, SplatIndex index, std::size_t pixels,
                          cudaStream_t stream) {
  if (const auto e = cudaMemsetAsync(index.counts, 0, (pixels + 1) * sizeof(unsigned), stream);
      e != cudaSuccess)
    return e;
  if (const auto e = cudaMemsetAsync(index.heads, 0xff, pixels * sizeof(int), stream); e != cudaSuccess)
    return e;
  index_splat<<<static_cast<unsigned>((pixels + 255) / 256), 256, 0, stream>>>(p, index);
  return cudaGetLastError();
}

cudaError_t enqueue_gather(const SoftSplatParams& p, SplatIndex index, std::size_t pixels,
                           cudaStream_t stream) {
  if (p.out_pitch < 0)
    launch_gather<Order::Group8>(p, index, pixels, stream);
  else if (p.out_pitch == 0)
    launch_gather<Order::Planar>(p, index, pixels, stream);
  else
    launch_gather<Order::Pixel>(p, index, pixels, stream);
  return cudaGetLastError();
}

}  // namespace

std::size_t indexed_splat_bytes(int width, int height) {
  if (width <= 0 || height <= 0) return 0;
  const auto pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (pixels > kMaxPixels) return 0;
  return pixels * (2 * sizeof(unsigned) + sizeof(int) + (kSlots + 4) * sizeof(Entry)) + sizeof(unsigned) +
         dense_capacity(pixels) * sizeof(float) * kDensePlanes;
}

cudaError_t launch_indexed_splats(const SoftSplatBatch& batch, void* scratch, std::size_t bytes,
                                  cudaStream_t stream) {
  if (batch.count < 1 || batch.count > kMaxSplatJobs || scratch == nullptr) return cudaErrorInvalidValue;
  for (int i = 0; i < batch.count; ++i)
    if (!valid_job(batch.jobs[i], bytes)) return cudaErrorInvalidValue;

  for (int i = 0; i < batch.count; ++i) {
    const SoftSplatParams& p = batch.jobs[i];
    const auto pixels = static_cast<std::size_t>(p.width) * p.height;
    const SplatIndex index = index_at(scratch, pixels);
    const bool rebuild = i == 0 || !same_projection(batch.jobs[i - 1], p);
    if (rebuild) {
      if (const auto e = enqueue_index(p, index, pixels, stream); e != cudaSuccess) return e;
    }
    scatter_dense<<<96, 256, 0, stream>>>(p, index, rebuild);
    if (const auto e = cudaGetLastError(); e != cudaSuccess) return e;
    if (const auto e = enqueue_gather(p, index, pixels, stream); e != cudaSuccess) return e;
  }
  if (batch.fill != nullptr && batch.fill_count != 0) {
    fill_instant<<<static_cast<unsigned>((batch.fill_count - 1) / 256 + 1), 256, 0, stream>>>(
        static_cast<__half*>(batch.fill), batch.fill_count, batch.fill_value);
    return cudaGetLastError();
  }
  return cudaSuccess;
}

}  // namespace kirari::gpu::cuda
