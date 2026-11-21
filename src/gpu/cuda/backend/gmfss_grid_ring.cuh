#pragma once
// Deux convolutions du GridNet à la suite par bandes. La première écrit chaque ligne une seule
// fois dans un anneau, et la seconde relit ses halos. Les produits et arrondis RIFE restent les mêmes.
#include <algorithm>

#include "gpu/cuda/backend/rife_native_kernels.cuh"

namespace kirari::gpu::cuda::gmfss_grid_ring {
namespace rn = rife_native;
inline constexpr int kChannels = 64, kBandRows = 96, kRingRows = 128;
static_assert(kBandRows + 2 * rn::kWarps + 1 <= kRingRows && (kRingRows & (kRingRows - 1)) == 0);

template <int Rows>
__device__ __forceinline__ void store_ring(const __half* s, const rn::Params& p, int y, int x) {
  constexpr int tile_height = Rows * rn::kWarps;
  constexpr int chunks = kChannels / 8, ld = kChannels + 8;
  for (int i = static_cast<int>(threadIdx.x); i < tile_height * rn::kTW * chunks; i += rn::kThreads) {
    const int q = i / chunks, c = (i % chunks) * 8;
    const int yy = y + q / rn::kTW, xx = x + q % rn::kTW;
    if (yy >= p.oh || xx >= p.ow) continue;
    const uint4 v = *reinterpret_cast<const uint4*>(s + q * ld + c);
    const long pixel = static_cast<long>(yy) * p.ow + xx;
    const long ring_pixel = static_cast<long>(yy & (kRingRows - 1)) * p.ow + xx;
    *reinterpret_cast<uint4*>(p.out + ring_pixel * p.out_ld + c) = rn::finish8(v, p, pixel, c);
  }
}

template <int Rows, bool ReadRing, bool Ahead>
__global__ void __launch_bounds__(rn::kThreads, 2) conv_band(rn::Params p, int first_y) {
  extern __shared__ __align__(16) unsigned char shared[];
  auto* s = reinterpret_cast<__half*>(shared);
  constexpr int tile_height = Rows * rn::kWarps;
  constexpr int ring_mask = kRingRows - 1;
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int y = first_y + static_cast<int>(blockIdx.y) * tile_height;
  const int x = static_cast<int>(blockIdx.x) * rn::kTW;
  constexpr auto acc_type = rn::Acc::kF16;
  rn::Frag<acc_type> acc[Rows * 2][kChannels / 8];
  if constexpr (ReadRing) {
    using G = rn::Geometry<3, 1, Rows>;
    const auto base = rn::smem_u32(s);
    constexpr int chunk = kChannels / 8, stride = kChannels + 8;
    for (int j = static_cast<int>(threadIdx.x); j < G::kIR * G::kIC * chunk; j += rn::kThreads) {
      const int q = j / chunk, c = 8 * (j % chunk);
      const int sy = y + q / G::kIC - 1, sx = x + q % G::kIC - 1;
      const bool ok = sy >= 0 && sy < p.ih && sx >= 0 && sx < p.iw;
      const long at = static_cast<long>(sy & ring_mask) * p.iw + sx;
      const auto* from = ok ? p.in + at * p.in_ld + c : p.in;
      rn::cp_async16(base + static_cast<std::uint32_t>((q * stride + c) * 2), from, ok);
    }
    rn::cp_async_wait_all();
    __syncthreads();
#pragma unroll
    for (int f = 0; f < Rows * 2; ++f)
#pragma unroll
      for (int n = 0; n < kChannels / 8; ++n) acc[f][n].clear();
    rn::accumulate<kChannels, kChannels, 3, 1, Rows, acc_type, Ahead>(acc, s, p.w, 0, 4, 0, 9, warp, lane);
  } else {
    rn::conv_sums<kChannels, kChannels, 3, 1, Rows, rn::In::kNhwc, acc_type, Ahead>(acc, s, p, y, x, 0, 0,
                                                                                    warp, lane);
  }
  __syncthreads();
  rn::stage<kChannels, Rows, acc_type>(acc, s, p.bias, warp, lane);
  __syncthreads();
  if constexpr (!ReadRing) {
    store_ring<Rows>(s, p, y, x);
  } else {
    rn::store<kChannels, Rows, rn::Out::kNhwc>(s, p, y, x, 0, 0);
  }
}

using Band = void (*)(rn::Params, int, int, cudaStream_t);

template <int Rows, bool ReadRing, bool Ahead>
void enqueue_band(rn::Params p, int first, int last, cudaStream_t stream) {
  constexpr int tile = Rows * rn::kWarps;
  constexpr int smem = rn::smem_bytes(kChannels, kChannels, 3, 1, Rows);
  const dim3 grid((p.ow + rn::kTW - 1) / rn::kTW, (last - first + tile - 1) / tile, 1);
  p.oh = last;
  conv_band<Rows, ReadRing, Ahead><<<grid, rn::kThreads, smem, stream>>>(p, first);
}

template <bool ReadRing>
Band pick_band(int rows, bool ahead) {
  if (rows == 1) return ahead ? enqueue_band<1, ReadRing, true> : enqueue_band<1, ReadRing, false>;
  return ahead ? enqueue_band<2, ReadRing, true> : enqueue_band<2, ReadRing, false>;
}

// Le lecteur garde les deux tranches K32 et la prélecture choisies par le régleur.
template <int Rows, bool Ahead>
__global__ void __launch_bounds__(rn::kThreads, 2) read_k32(rn::Params p, int first_y) {
  extern __shared__ __align__(16) unsigned char memory[];
  auto* shared = reinterpret_cast<__half*>(memory);
  constexpr auto acc = rn::Acc::kF16;
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int y = first_y + static_cast<int>(blockIdx.y) * (rn::kWarps * Rows);
  const int x = static_cast<int>(blockIdx.x) * rn::kTW;
  rn::Frag<acc> sums[Rows * 2][kChannels / 8];
  rn::conv_sums<32, kChannels, 3, 1, Rows, rn::In::kNhwc, acc, Ahead, kRingRows>(sums, shared, p, y, x, 0, 0,
                                                                                 warp, lane);
  __syncthreads();
  rn::stage<kChannels, Rows, acc>(sums, shared, p.bias, warp, lane);
  __syncthreads();
  rn::store<kChannels, Rows, rn::Out::kNhwc>(shared, p, y, x, 0, 0);
}

template <int Rows, bool Ahead>
void read_k32_band(rn::Params p, int first, int last, cudaStream_t stream) {
  constexpr int rows = Rows * rn::kWarps;
  constexpr int bytes = rn::smem_bytes(32, kChannels, 3, 1, Rows);
  const dim3 grid((p.ow + rn::kTW - 1) / rn::kTW, (last - first + rows - 1) / rows);
  p.oh = last;
  read_k32<Rows, Ahead><<<grid, rn::kThreads, bytes, stream>>>(p, first);
}

inline Band pick_exact(int kc, int ct, int rows, bool ahead) {
  if (ct != kChannels || (rows != 1 && rows != 2)) return nullptr;
  if (kc == kChannels) return pick_band<true>(rows, ahead);
  if (kc != 32) return nullptr;
  if (rows == 1) return ahead ? read_k32_band<1, true> : read_k32_band<1, false>;
  return ahead ? read_k32_band<2, true> : read_k32_band<2, false>;
}

template <class Producer>
inline void enqueue_bands(rn::Params first, rn::Params second, int first_rows, const Producer& produce,
                          Band consume, cudaStream_t stream, int band_rows = kBandRows) {
  const int tile = first_rows * rn::kWarps;
  int produced = 0;
  for (int y = 0; y < second.oh; y += band_rows) {
    const int end = std::min(y + band_rows, second.oh);
    // Le noyau suivant lit jusqu'à la ligne end. La tuile entière qui la contient est produite une fois.
    const int need = std::min(first.oh, (end + tile) / tile * tile);
    if (need > produced) produce(first, produced, need, stream);
    consume(second, y, end, stream);
    produced = need;
  }
}

inline void enqueue_residual(rn::Params first, rn::Params second, int first_rows, bool first_ahead,
                             int second_rows, bool second_ahead, cudaStream_t stream) {
  enqueue_bands(first, second, first_rows, pick_band<false>(first_rows, first_ahead),
                pick_band<true>(second_rows, second_ahead), stream);
}
}  // namespace kirari::gpu::cuda::gmfss_grid_ring
