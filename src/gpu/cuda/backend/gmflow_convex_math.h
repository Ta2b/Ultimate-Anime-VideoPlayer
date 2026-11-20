// Index de l'agrandissement convexe de GMFlow, commun au noyau et au rejeu sur processeur.
#pragma once

namespace kirari::gpu::cuda::gmflow_convex {

inline constexpr int kFactor = 4;
inline constexpr int kSubpixels = kFactor * kFactor;
inline constexpr int kNeighbors = 9;
inline constexpr int kChannels = kNeighbors * kSubpixels;
inline constexpr int kFeatures = 128;
inline constexpr int kInput = 144;

#if defined(__CUDACC__)
#define KIRARI_CONVEX_HD __host__ __device__
#else
#define KIRARI_CONVEX_HD
#endif

struct Grid {
  int height;
  int width;

  KIRARI_CONVEX_HD constexpr int pixels() const { return height * width; }
  KIRARI_CONVEX_HD constexpr int neighbor(int cell, int tap) const {
    const int y = cell / width + tap / 3 - 1;
    const int x = cell % width + tap % 3 - 1;
    return x < 0 || x >= width || y < 0 || y >= height ? -1 : y * width + x;
  }
  KIRARI_CONVEX_HD constexpr int expanded(int cell, int sub) const {
    const int y = kFactor * (cell / width) + sub / kFactor;
    const int x = kFactor * (cell % width) + sub % kFactor;
    return y * (kFactor * width) + x;
  }
};

KIRARI_CONVEX_HD constexpr int mask_channel(int tap, int sub) {
  return tap * kSubpixels + sub;
}

#undef KIRARI_CONVEX_HD

}  // namespace kirari::gpu::cuda::gmflow_convex
