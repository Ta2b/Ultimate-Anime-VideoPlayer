#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace kirari::gpu::cuda {

struct GpuAnchor {
  int x;
  int y;
  float hue_x;
  float hue_y;
  float score;
};

struct GlowAggregate {
  float red;
  float green;
  float blue;
  float pq;
  float centre_pq;
  int x;
  int y;
};

struct GlowSource {
  const void* rgb = nullptr;
  bool fp16 = true;
  int width = 0;
  int height = 0;
};

cudaError_t launch_glow_reduce(GlowSource source, GlowAggregate* aggregates, cudaStream_t stream);
cudaError_t launch_glow_crowns(GlowSource source, const GlowAggregate* aggregates, GpuAnchor* anchors,
                               cudaStream_t stream);

struct GlowMatch {
  int width;
  int height;
  int divisor;
  int anchor_block;
  int guide_block;
  const GpuAnchor* first;
  const GpuAnchor* second;
  const short2* flow;
  std::size_t flow_pitch;
  const std::uint8_t* cost;
  std::size_t cost_pitch;
  const std::uint8_t* luma_first;
  std::size_t luma_first_pitch;
  const std::uint8_t* luma_second;
  std::size_t luma_second_pitch;
  // Plan compact, mots float positifs stockés sous forme uint32_t pour atomicMax.
  std::uint32_t* guide;
  std::size_t guide_count;
};

constexpr int glow_div_up(int size, int divisor) {
  return (size + divisor - 1) / divisor;
}

cudaError_t launch_glow_match(const GlowMatch& match, cudaStream_t stream);

}  // namespace kirari::gpu::cuda
