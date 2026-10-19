// L'étage gpu de la tranche verticale sur un backend chargé (gpu/backend.h) : chaque image
// décodée est ingérée en RGB FP16, agrandie par le moteur SPAN (×2), puis écrite dans un slot
// du ring que render présente. Sans moteur, l'image source passe telle quelle.
#pragma once

#include <expected>
#include <memory>
#include <vector>

#include "app/play/pipeline.h"
#include "app/play/ring_stage.h"
#include "core/ring.h"
#include "gpu/backend.h"

namespace ajp::app::play {

struct UpscaleSetup {
  gpu::Backend* backend = nullptr;  // vit plus longtemps que l'étage
  gpu::Engine* engine = nullptr;    // SPAN aux formes de la source ; nullptr : sans agrandissement
  int source_width = 0;
  int source_height = 0;
  std::vector<RingSlotDesc> slots;  // ring de render, à la taille de sortie du moteur
};

class UpscaleStage final : public RingStage {
 public:
  // Refus : backend sans noyaux, slots absents ou d'une taille qui n'est pas la sortie du
  // moteur (ShapeInvalid), tenseurs refusés par le backend, thread refusé.
  static std::expected<std::unique_ptr<VideoStage>, Refusal> start(VideoStageInit init, UpscaleSetup setup);
  ~UpscaleStage() override { stop(); }

 private:
  UpscaleStage(VideoStageInit init, UpscaleSetup setup);
  std::expected<void, Refusal> produce(const DecodedFrame& frame, std::uint32_t slot,
                                       std::uint64_t wait_value, std::uint64_t signal_value) override;

  UpscaleSetup setup_;
  std::unique_ptr<gpu::TensorBuffer> input_;   // 1 × 3 × h × w
  std::unique_ptr<gpu::TensorBuffer> output_;  // 1 × 3 × 2h × 2w ; absent sans moteur
};

}  // namespace ajp::app::play
