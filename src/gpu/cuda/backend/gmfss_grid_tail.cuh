#pragma once

#include <algorithm>

#include "gpu/cuda/backend/gmfss_grid_rgb.cuh"
#include "gpu/cuda/backend/gmfss_shuffle_reuse.cuh"

namespace kirari::gpu::cuda::gmfss_grid_tail {
namespace rn = rife_native;
inline constexpr int kBandRows = 96;
inline constexpr int kInputTile = rn::kWarps, kOutputTile = 2 * rn::kWarps;
inline constexpr int kRingRows = gmfss_shuffle_reuse::kRingRows;
static_assert(kBandRows % kOutputTile == 0 && kBandRows + kOutputTile + 1 <= kRingRows);
static_assert((kRingRows & (kRingRows - 1)) == 0);

// Chaque pixel de PixelShuffle est écrit une fois. La dernière convolution lit jusqu'à la
// première ligne de la bande suivante, conservée dans l'anneau avec le halo du haut.
template <int Kc>
void enqueue(rn::Params first, const rn::Params& last, cudaStream_t stream) {
  constexpr int shared_a = gmfss_shuffle_reuse::smem_bytes<1>();
  constexpr int shared_b = rn::smem_bytes(Kc, 16, 3, 1, 2);
  const auto groups = [](int n, int tile) { return static_cast<unsigned>((n + tile - 1) / tile); };
  const int height = first.oh;
  int produced = 0;
  for (int y = 0; y < last.oh; y += kBandRows) {
    const int end = std::min(y + kBandRows, last.oh);
    const int need = std::min(height, (end + kOutputTile) / kOutputTile * kInputTile);
    if (need > produced) {
      first.oh = need;
      const dim3 blocks(groups(first.ow, rn::kTW), groups(need - produced, kInputTile), 2);
      gmfss_shuffle_reuse::conv_shuffle_ring<<<blocks, rn::kThreads, shared_a, stream>>>(first, produced);
      produced = need;
    }
    const dim3 blocks(groups(last.ow, rn::kTW), groups(end - y, kOutputTile));
    gmfss_grid_rgb::conv_rgb_ring<Kc><<<blocks, rn::kThreads, shared_b, stream>>>(last, y, end);
  }
}
}  // namespace kirari::gpu::cuda::gmfss_grid_tail
