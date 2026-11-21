#pragma once

#include <array>
#include <memory>
#include <optional>

#include "gpu/cuda/backend/gpu_timing.h"
#include "gpu/cuda/backend/nvof_session.h"
#include "gpu/cuda/backend/scene_glow_kernels.h"
#include "gpu/kernel_ops.h"
#include "kernels/spec/scene_glow.h"

namespace kirari::gpu::cuda {

struct GlowProfile {
  GpuTiming* reduce = nullptr;
  GpuTiming* crowns = nullptr;
  GpuTiming* match = nullptr;
};

class SceneGlow {
 public:
  static std::unique_ptr<SceneGlow> open(int width, int height);
  ~SceneGlow();
  SceneGlow(const SceneGlow&) = delete;
  SceneGlow& operator=(const SceneGlow&) = delete;

  bool shape(int width, int height) const { return width == width_ && height == height_; }
  void forget(int a_slot, int b_slot, bool keep_a);
  bool stage(cudaStream_t lane, int a_slot, int b_slot, bool keep_a, GlowSource a, GlowSource b,
             const GlowProfile& profile);
  // Après l'événement de la carte de scène. Le producteur prolonge la lecture des lumas jusqu'à la
  // fin du noyau, y compris si la relecture du guide est refusée après le lancement.
  bool match(NvofSession& session, int divisor, int a_slot, int b_slot, std::uint64_t ticket,
             GpuTiming* timing);
  std::optional<kernels::SceneGlowTiles> take(std::uint64_t ticket);

 private:
  SceneGlow(int width, int height);
  struct Source {
    GlowAggregate* aggregates = nullptr;
    GpuAnchor* anchors = nullptr;
    bool ready = false;
  };
  struct Read {
    float* host = nullptr;
    cudaEvent_t done = nullptr;
    std::uint64_t ticket = 0;
  };
  bool allocate();
  const int width_;
  const int height_;
  const int cols_;
  const int rows_;
  const std::size_t cells_;
  std::uint32_t* device_ = nullptr;
  cudaEvent_t anchors_ready_ = nullptr;
  std::array<Source, NvofSession::kLumas> sources_{};
  std::array<Read, kSceneFlowReadbacks> reads_{};
};

}  // namespace kirari::gpu::cuda
