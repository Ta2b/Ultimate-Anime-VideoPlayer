#include "app/play/upscale_stage.h"

#include <array>
#include <format>
#include <utility>

#include "app/play/options.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

gpu::Shape rgb_shape(int width, int height) {
  return gpu::Shape{.dims = {1, 3, height, width}};
}

}  // namespace

UpscaleStage::UpscaleStage(VideoStageInit init, UpscaleSetup setup)
    : RingStage(std::move(init), static_cast<std::uint32_t>(setup.slots.size()),
                setup.slots.front().generation),
      setup_(std::move(setup)) {}

std::expected<std::unique_ptr<VideoStage>, Refusal> UpscaleStage::start(VideoStageInit init,
                                                                        UpscaleSetup setup) {
  if (setup.backend == nullptr || setup.backend->kernels() == nullptr) {
    return refuse(RefusalCode::FieldMissing, "étage gpu : backend sans noyaux");
  }
  if (setup.slots.empty() || setup.slots.size() > kMaxRingSlots) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("étage gpu : 1 à {} slots", kMaxRingSlots));
  }
  const int factor = setup.engine != nullptr ? kSpanFactor : 1;
  for (const RingSlotDesc& slot : setup.slots) {
    if (slot.width != setup.source_width * factor || slot.height != setup.source_height * factor) {
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("slot {} : {}x{} au lieu de la sortie {}x{}", slot.index, slot.width,
                                slot.height, setup.source_width * factor, setup.source_height * factor));
    }
  }
  std::unique_ptr<UpscaleStage> stage(new UpscaleStage(std::move(init), std::move(setup)));
  gpu::Backend& backend = *stage->setup_.backend;
  auto input = backend.alloc_tensor(gpu::TensorType::F16,
                                    rgb_shape(stage->setup_.source_width, stage->setup_.source_height));
  if (!input) return std::unexpected(std::move(input.error()));
  stage->input_ = std::move(*input);
  if (stage->setup_.engine != nullptr) {
    auto output = backend.alloc_tensor(
        gpu::TensorType::F16,
        rgb_shape(stage->setup_.source_width * kSpanFactor, stage->setup_.source_height * kSpanFactor));
    if (!output) return std::unexpected(std::move(output.error()));
    stage->output_ = std::move(*output);
  }
  if (auto started = stage->RingStage::start("gpu"); !started)
    return std::unexpected(std::move(started.error()));
  return stage;
}

std::expected<void, Refusal> UpscaleStage::produce(const DecodedFrame& frame, std::uint32_t slot,
                                                   std::uint64_t wait_value, std::uint64_t signal_value) {
  if (frame.width != setup_.source_width || frame.height != setup_.source_height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("image {}x{} : la source est en {}x{}", frame.width, frame.height,
                              setup_.source_width, setup_.source_height));
  }
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const gpu::Tensor& rgb = input_->tensor();
  if (auto ok = ops.ingest(frame, gpu::IngestSpec{}, rgb); !ok) return ok;
  const gpu::RingWrite sync{.wait_value = wait_value, .signal_value = signal_value};
  if (setup_.engine == nullptr) return ops.write_ring(rgb, setup_.slots[slot], sync);
  const std::array inputs{rgb};
  std::array outputs{output_->tensor()};
  if (auto ok = setup_.backend->enqueue(*setup_.engine, inputs, outputs); !ok) return ok;
  return ops.write_ring(output_->tensor(), setup_.slots[slot], sync);
}

}  // namespace ajp::app::play
