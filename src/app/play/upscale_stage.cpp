#include "app/play/upscale_stage.h"

#include <algorithm>
#include <array>
#include <format>
#include <utility>

#include "app/play/options.h"
#include "app/play/rife_plan.h"
#include "core/log.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

gpu::Shape planes(int channels, int width, int height) {
  return gpu::Shape{.dims = {1, channels, height, width}};
}

}  // namespace

UpscaleStage::UpscaleStage(VideoStageInit init, UpscaleSetup setup)
    : RingStage(std::move(init), static_cast<std::uint32_t>(setup.slots.size()),
                setup.slots.front().generation),
      setup_(std::move(setup)) {}

UpscaleStage::~UpscaleStage() {
  stop();
  if (setup_.rife != nullptr) {
    log_channel("gpu")->info("RIFE ×{} : {} paires, {} interpolées, {} coupes de scène, {} dessins tenus",
                             setup_.rife_factor, pairs_, interpolated_, cuts_, held_);
  }
}

std::expected<std::unique_ptr<VideoStage>, Refusal> UpscaleStage::start(VideoStageInit init,
                                                                        UpscaleSetup setup) {
  if (setup.backend == nullptr || setup.backend->kernels() == nullptr) {
    return refuse(RefusalCode::FieldMissing, "étage gpu : backend sans noyaux");
  }
  if (setup.work_width == 0 && setup.work_height == 0) {
    setup.work_width = setup.source_width;
    setup.work_height = setup.source_height;
  }
  if (setup.work_width < 1 || setup.work_height < 1 || setup.work_width > setup.source_width ||
      setup.work_height > setup.source_height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("étage gpu : taille de travail {}x{} hors de la source {}x{}", setup.work_width,
                              setup.work_height, setup.source_width, setup.source_height));
  }
  if (setup.rife != nullptr && setup.live == nullptr &&
      (setup.rife_factor < 2 || setup.rife_factor > kMaxRifeFactor)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("étage gpu : RIFE ×{} hors de [2, {}]", setup.rife_factor, kMaxRifeFactor));
  }
  if (setup.slots.empty() || setup.slots.size() > kMaxRingSlots) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("étage gpu : 1 à {} slots", kMaxRingSlots));
  }
  const int factor = setup.engine != nullptr ? kSpanFactor : 1;
  for (const RingSlotDesc& slot : setup.slots) {
    if (slot.width != setup.work_width * factor || slot.height != setup.work_height * factor) {
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("slot {} : {}x{} au lieu de la sortie {}x{}", slot.index, slot.width,
                                slot.height, setup.work_width * factor, setup.work_height * factor));
    }
  }
  std::unique_ptr<UpscaleStage> stage(new UpscaleStage(std::move(init), std::move(setup)));
  if (auto ok = stage->allocate(); !ok) return std::unexpected(std::move(ok.error()));
  if (auto started = stage->RingStage::start("gpu"); !started)
    return std::unexpected(std::move(started.error()));
  return stage;
}

std::expected<void, Refusal> UpscaleStage::allocate() {
  gpu::Backend& backend = *setup_.backend;
  const int w = setup_.work_width;
  const int h = setup_.work_height;
  const auto alloc = [&](std::unique_ptr<gpu::TensorBuffer>& into, const gpu::Shape& shape) {
    auto tensor = backend.alloc_tensor(gpu::TensorType::F16, shape);
    if (tensor) into = std::move(*tensor);
    return tensor ? std::expected<void, Refusal>{} : std::unexpected(std::move(tensor.error()));
  };
  for (auto& rgb : rgb_) {
    if (auto ok = alloc(rgb, planes(3, w, h)); !ok) return ok;
  }
  if (setup_.engine != nullptr) {
    if (auto ok = alloc(output_, planes(3, w * kSpanFactor, h * kSpanFactor)); !ok) return ok;
  }
  if (setup_.rife != nullptr) {
    const kernels::RifeGeometry g = kernels::rife_geometry(w, h);
    if (auto ok = alloc(rife_in_, planes(kernels::kRifeInputChannels, g.padded_width, g.padded_height)); !ok)
      return ok;
    if (auto ok = alloc(rife_out_, planes(3, g.padded_width, g.padded_height)); !ok) return ok;
    if (auto ok = alloc(middle_, planes(3, w, h)); !ok) return ok;
  }
  return {};
}

std::expected<std::uint32_t, Refusal> UpscaleStage::outputs(const DecodedFrame& frame) {
  if (frame.width != setup_.source_width || frame.height != setup_.source_height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("image {}x{} : la source est en {}x{}", frame.width, frame.height,
                              setup_.source_width, setup_.source_height));
  }
  if (setup_.true_hdr && (frame.color.transfer == Transfer::Pq || frame.color.transfer == Transfer::Hlg)) {
    return refuse(RefusalCode::HdrUnavailable, "RTX HDR : la source est déjà HDR");
  }
  prepared_ = vendor_hdr::InputPreparation{.matrix = frame.color.matrix, .range = frame.color.range};
  gpu::KernelOps& ops = *setup_.backend->kernels();
  current_ ^= 1U;
  const gpu::Tensor& rgb = rgb_[current_]->tensor();
  gpu::IngestSpec spec;
  if (setup_.live) spec.reduce = setup_.live->reduction();
  if (auto ok = ops.ingest(frame, spec, rgb); !ok) return std::unexpected(std::move(ok.error()));
  std::uint32_t factor = setup_.rife != nullptr ? setup_.rife_factor : 1;
  if (setup_.rife != nullptr && setup_.live)
    factor = std::clamp(setup_.live->rife_factor(), 1U, kMaxRifeFactor);
  PairFacts facts{.factor = factor,
                  .same_epoch = previous_ && previous_->epoch == frame.epoch,
                  .gap_us = previous_ ? frame.pts_us - previous_->pts_us : 0};
  if (facts.factor > 1 && facts.same_epoch) {
    auto scores = ops.scores_now(rgb_[current_ ^ 1U]->tensor(), rgb);
    if (!scores) return std::unexpected(std::move(scores.error()));
    facts.scores = *scores;
    ++pairs_;
    if (setup_.live) setup_.live->count_pair();
    if (kernels::is_scene_cut(scores->scene_score)) {
      ++cuts_;
      if (setup_.live) setup_.live->count_cut();
    } else if (kernels::is_held(scores->worst_block)) {
      ++held_;
      if (setup_.live) setup_.live->count_held();
    }
  }
  outputs_ = rife_outputs(facts);
  if (outputs_ > 1) {
    ++interpolated_;
    if (setup_.live) setup_.live->count_interpolated();
  }
  return outputs_;
}

std::expected<RingStage::Output, Refusal> UpscaleStage::produce(const DecodedFrame& frame,
                                                                std::uint32_t output, std::uint32_t slot,
                                                                std::uint64_t wait_value,
                                                                std::uint64_t signal_value) {
  const gpu::RingWrite sync{.wait_value = wait_value, .signal_value = signal_value};
  const gpu::Tensor& now = rgb_[current_]->tensor();
  if (output + 1 >= outputs_) {  // l'image elle-même, en dernier
    if (auto ok = finish(now, slot, sync); !ok) return std::unexpected(std::move(ok.error()));
    previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
    return Output{.pts_us = frame.pts_us, .left = frame.id, .right = frame.id};
  }
  // Image intermédiaire k + 1 sur outputs_, entre la précédente et celle-ci (rife_outputs ne
  // donne plus d'une sortie que si la précédente existe).
  if (!previous_) return refuse(RefusalCode::SequenceOrder, "RIFE : pas d'image précédente");
  const Previous before = *previous_;
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const float t = static_cast<float>(output + 1) / static_cast<float>(outputs_);
  if (auto ok = ops.rife_input(rgb_[current_ ^ 1U]->tensor(), now, t, rife_in_->tensor()); !ok)
    return std::unexpected(std::move(ok.error()));
  const std::array inputs{rife_in_->tensor()};
  std::array results{rife_out_->tensor()};
  if (auto ok = setup_.backend->enqueue(*setup_.rife, inputs, results); !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = ops.rife_crop(rife_out_->tensor(), middle_->tensor()); !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = finish(middle_->tensor(), slot, sync); !ok) return std::unexpected(std::move(ok.error()));
  return Output{.pts_us = rife_output_pts(before.pts_us, frame.pts_us, output, outputs_),
                .left = before.id,
                .right = frame.id};
}

std::expected<void, Refusal> UpscaleStage::finish(const gpu::Tensor& rgb, std::uint32_t slot,
                                                  const gpu::RingWrite& sync) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (setup_.true_hdr && setup_.live) {
    if (const auto params = setup_.live->take_true_hdr()) {
      if (auto set = ops.set_true_hdr(*params); !set)
        log_channel("gpu")->warn("RTX HDR : réglages refusés ({})", set.error().detail);
    }
  }
  const auto write = [&](const gpu::Tensor& image) {
    return setup_.true_hdr ? ops.write_ring_true_hdr(image, prepared_, setup_.slots[slot], sync)
                           : ops.write_ring(image, setup_.slots[slot], sync);
  };
  if (setup_.engine == nullptr) return write(rgb);
  const std::array inputs{rgb};
  std::array outputs{output_->tensor()};
  if (auto ok = setup_.backend->enqueue(*setup_.engine, inputs, outputs); !ok) return ok;
  return write(output_->tensor());
}

}  // namespace ajp::app::play
