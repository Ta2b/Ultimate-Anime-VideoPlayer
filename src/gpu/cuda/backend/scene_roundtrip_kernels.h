#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

#include "kernels/spec/scene_flow.h"

namespace kirari::gpu::cuda {

struct SceneRoundtripParams {
  struct Plane {
    const void* pixels = nullptr;
    std::size_t pitch = 0;
  };
  Plane luma_a;
  Plane luma_b;
  Plane flow_a;
  Plane flow_b;
  int width = 0;
  int height = 0;
  kernels::SceneRoundtripTile* out = nullptr;
};

cudaError_t launch_scene_roundtrip(const SceneRoundtripParams& p, cudaStream_t stream);

}  // namespace kirari::gpu::cuda
