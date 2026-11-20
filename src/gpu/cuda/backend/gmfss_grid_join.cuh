#pragma once

#include "gpu/cuda/backend/gmfss_grid_ring.cuh"

namespace kirari::gpu::cuda::gmfss_grid_join {
namespace rn = rife_native;
inline constexpr int kChannels = gmfss_grid_ring::kChannels;
inline constexpr int kBandRows = gmfss_grid_ring::kBandRows, kRingRows = gmfss_grid_ring::kRingRows;
inline constexpr int kWorkspaceRows = 3 * kRingRows;
inline constexpr int kRingMask = kRingRows - 1;
static_assert(kBandRows % 16 == 0 && kBandRows + 17 <= kRingRows);

template <int Kc, int Rows, bool Ahead>
__device__ __forceinline__ void sums_ring(rn::Frag<rn::Acc::kF16> (&acc)[Rows * 2][8], __half* s,
                                          const rn::Params& p, int y, int x, int warp, int lane) {
  using G = rn::Geometry<3, 1, Rows>;
  constexpr int kChunks = Kc / 8, kStride = Kc + 8;
  const auto base = rn::smem_u32(s);
#pragma unroll
  for (int f = 0; f < Rows * 2; ++f)
#pragma unroll
    for (int n = 0; n < kChannels / 8; ++n) acc[f][n].clear();

  // Même ordre que conv_sums : toutes les prises de la première tranche, puis celles de la suivante.
  // Chaque tranche remplace la précédente dans le halo, après la dernière lecture par les MMA.
#pragma unroll 1
  for (int slice = 0; slice < kChannels / Kc; ++slice) {
    if (slice != 0) __syncthreads();
    for (int j = static_cast<int>(threadIdx.x); j < G::kIR * G::kIC * kChunks; j += rn::kThreads) {
      const int q = j / kChunks, c = 8 * (j % kChunks);
      const int row = y + q / G::kIC - 1, col = x + q % G::kIC - 1;
      const bool valid = row >= 0 && row < p.ih && col >= 0 && col < p.iw;
      const long at = static_cast<long>(row & kRingMask) * p.iw + col;
      const auto* from = valid ? p.in + at * p.in_ld + slice * Kc + c : p.in;
      rn::cp_async16(base + static_cast<std::uint32_t>((q * kStride + c) * 2), from, valid);
    }
    rn::cp_async_wait_all();
    __syncthreads();
    if (p.in_slope != 1.F) {
      rn::prelu_tile<Kc, G::kIR * G::kIC>(s, p.in_slope);
      __syncthreads();
    }
    rn::accumulate<Kc, kChannels, 3, 1, Rows, rn::Acc::kF16, Ahead>(
        acc, s, p.w, slice, kChannels / 16, 0, 9, warp, lane);
  }
}

template <int Kc, int Rows, bool Phase, bool SkipRing, bool Ahead = false>
__global__ void __launch_bounds__(rn::kThreads, 2) conv_band(rn::Params p, int first_y) {
  static_assert(!Phase || (!SkipRing && !Ahead));
  static_assert(Kc == 32 || Kc == 64);
  extern __shared__ __align__(16) unsigned char bytes[];
  auto* s = reinterpret_cast<__half*>(bytes);
  constexpr auto kAcc = rn::Acc::kF16;
  constexpr int kRows = rn::kWarps * Rows, kChunks = 8, kStride = kChannels + 8;
  const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
  const int y = first_y + static_cast<int>(blockIdx.y) * kRows;
  const int x = static_cast<int>(blockIdx.x) * rn::kTW;
  const int z = static_cast<int>(blockIdx.z);
  rn::Frag<kAcc> acc[Rows * 2][kChunks];
  if constexpr (Phase) {
    rn::conv_sums<Kc, kChannels, 2, 1, Rows, rn::In::kNhwc, kAcc, false>(
        acc, s, p, y, x, z, 0, warp, lane);
  } else {
    sums_ring<Kc, Rows, Ahead>(acc, s, p, y, x, warp, lane);
  }
  __syncthreads();
  rn::stage<kChannels, Rows, kAcc>(acc, s, p.bias + static_cast<long>(z) * kChannels, warp, lane);
  __syncthreads();

  for (int i = static_cast<int>(threadIdx.x); i < kRows * rn::kTW * kChunks; i += rn::kThreads) {
    const int q = i / kChunks, c = (i % kChunks) * 8;
    const int yy = y + q / rn::kTW, xx = x + q % rn::kTW;
    if (yy >= p.oh || xx >= p.ow) continue;
    const uint4 v = *reinterpret_cast<const uint4*>(s + q * kStride + c);
    if constexpr (Phase) {
      const long row = (2 * yy + (z >> 1)) & kRingMask;
      const long pixel = row * (2 * p.ow) + 2 * xx + (z & 1);
      *reinterpret_cast<uint4*>(p.out + pixel * p.out_ld + c) = v;
    } else {
      const long dst = static_cast<long>(SkipRing ? yy : yy & kRingMask) * p.ow + xx;
      const long skip = static_cast<long>(SkipRing ? yy & kRingMask : yy) * p.ow + xx;
      *reinterpret_cast<uint4*>(p.out + dst * p.out_ld + c) = rn::finish8(v, p, skip, c);
    }
  }
}

template <int Kc, int Rows, bool Phase, bool SkipRing, bool Ahead = false>
void launch_band(rn::Params p, int first, int last, cudaStream_t stream) {
  constexpr int kTile = Rows * rn::kWarps;
  constexpr int kSmem = rn::smem_bytes(Kc, kChannels, Phase ? 2 : 3, 1, Rows);
  const dim3 grid((p.ow + rn::kTW - 1) / rn::kTW, (last - first + kTile - 1) / kTile, Phase ? 4 : 1);
  p.oh = last;
  conv_band<Kc, Rows, Phase, SkipRing, Ahead><<<grid, rn::kThreads, kSmem, stream>>>(p, first);
}

struct Join {
  rn::Params residual_first, residual_second, up_first, up_second;
};

struct Config {
  bool residual_ahead;
  int residual_k, up_k, up_rows;
  bool add_ahead;
};

inline gmfss_grid_ring::Band up_band(const Config& cfg) {
  if (cfg.up_k == 32)
    return cfg.up_rows == 1 ? launch_band<32, 1, true, false> : launch_band<32, 2, true, false>;
  return cfg.up_rows == 1 ? launch_band<64, 1, true, false> : launch_band<64, 2, true, false>;
}

inline constexpr int prefix(int rows, int tile, int end) {
  return std::min((rows + tile - 1) / tile * tile, end);
}

// Les quatre convolutions gardent des coordonnées logiques globales et n'écrivent chaque ligne qu'une fois.
// Les appels du même stream ordonnent les deux producteurs avant la lecture de leur bande.
inline void join(const Join& p, const Config& cfg, cudaStream_t stream) {
  constexpr int kResTile = 2 * rn::kWarps;
  const int up_tile = cfg.up_rows * rn::kWarps;
  const int height = p.up_second.oh;
  const auto transpose = up_band(cfg);
  const auto produce = gmfss_grid_ring::pick_band<false>(2, cfg.residual_ahead);
  const auto residual = cfg.residual_k == 32 ? launch_band<32, 1, false, false> : launch_band<64, 1, false, false>;
  const auto add = cfg.add_ahead ? launch_band<64, 2, false, true, true> : launch_band<64, 2, false, true>;
  int res_written = 0, up_written = 0;
  for (int start = 0; start < height; start += kBandRows) {
    const int end = std::min(start + kBandRows, height);
    const int res_need = prefix(end + 1, kResTile, height);
    const int up_need = prefix((end + 2) / 2, up_tile, p.up_first.oh);
    if (up_need > up_written) transpose(p.up_first, up_written, up_need, stream);
    if (res_need > res_written) produce(p.residual_first, res_written, res_need, stream);
    residual(p.residual_second, start, end, stream);
    add(p.up_second, start, end, stream);
    res_written = res_need;
    up_written = up_need;
  }
}

}  // namespace kirari::gpu::cuda::gmfss_grid_join
