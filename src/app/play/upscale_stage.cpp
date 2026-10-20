#include "app/play/upscale_stage.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <utility>

#include "app/play/options.h"
#include "app/play/rife_plan.h"
#include "core/clock.h"
#include "core/log.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/optical_flow.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/warp.h"

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
  if (setup_.rife_motion != nullptr) {
    log_channel("gpu")->info("RIFE mouvement : {} passes ; {} paires à une passe, {} à plusieurs", rife_runs_,
                             single_pairs_, multi_pairs_);
  }
  if (setup_.optical_flow_grid > 0)
    log_channel("gpu")->info("flux optique matériel : {} paires", optical_flow_pairs_);
  if (gmfss()) log_channel("gpu")->info("GMFSS : {} paires", gmfss_pairs_);
  if (late_pairs_ > 0) {
    log_channel("gpu")->info("RIFE : {} paires non interpolées pour rattraper l'horloge", late_pairs_);
  }
  if (interpolates()) {
    log_channel("gpu")->info("RIFE {} : {} paires, {} interpolées, {} coupes de scène, {} dessins tenus",
                             factor_text(setup_.live ? setup_.live->rife_factor() : setup_.rife_factor),
                             pairs_, interpolated_, cuts_, held_);
  }
  if (!scores_wait_us_.empty()) {
    std::ranges::sort(scores_wait_us_);
    log_channel("gpu")->info(
        "profil hôte · attente des mesures de la paire : médiane {:.2f} ms, p95 {:.2f} ms ({})",
        static_cast<double>(scores_wait_us_[scores_wait_us_.size() / 2]) / 1000.0,
        static_cast<double>(scores_wait_us_[scores_wait_us_.size() * 95 / 100]) / 1000.0,
        scores_wait_us_.size());
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
  if ((setup.gmfss_pair == nullptr) != (setup.gmfss_fusion == nullptr))
    return refuse(RefusalCode::FieldMissing, "étage gpu : GMFSS demande ses deux moteurs");
  const int estimators = (setup.rife != nullptr ? 1 : 0) + (setup.rife_motion != nullptr ? 1 : 0) +
                         (setup.optical_flow_grid > 0 ? 1 : 0) + (setup.gmfss_pair != nullptr ? 1 : 0);
  if (estimators > 1) {
    return refuse(RefusalCode::FieldOutOfRange,
                  "étage gpu : RIFE image, RIFE mouvement, flux optique ou GMFSS, un seul des quatre");
  }
  if (setup.optical_flow_grid < 0 || setup.optical_flow_grid > 4) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("étage gpu : grille du flux optique {}", setup.optical_flow_grid));
  }
  if ((setup.rife != nullptr || setup.rife_motion != nullptr) &&
      !kernels::is_rife_input_planes(setup.rife_planes))
    return refuse(RefusalCode::InputMismatch,
                  std::format("étage gpu : RIFE à {} plans d'entrée", setup.rife_planes));
  if (estimators == 1 && setup.live == nullptr &&
      (setup.rife_factor.num <= setup.rife_factor.den || !valid_factor(setup.rife_factor))) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("étage gpu : RIFE {} hors de ]1, {}]",
                                                            factor_text(setup.rife_factor), kMaxRifeFactor));
  }
  if (setup.rife_width == 0 && setup.rife_height == 0) {
    setup.rife_width = setup.work_width;
    setup.rife_height = setup.work_height;
  }
  if (setup.rife_width < 1 || setup.rife_height < 1 || setup.rife_width > setup.work_width ||
      setup.rife_height > setup.work_height) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("étage gpu : RIFE en {}x{} hors du travail {}x{}", setup.rife_width,
                              setup.rife_height, setup.work_width, setup.work_height));
  }
  if ((setup.rife != nullptr || setup.gmfss_pair != nullptr) &&
      (setup.rife_width != setup.work_width || setup.rife_height != setup.work_height)) {
    return refuse(RefusalCode::ShapeInvalid,
                  "étage gpu : RIFE image et GMFSS à la taille de travail seulement");
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
  if (interpolates() && (setup_.rife_width != w || setup_.rife_height != h)) {
    for (auto& small : small_) {
      if (auto ok = alloc(small, planes(3, setup_.rife_width, setup_.rife_height)); !ok) return ok;
    }
  }
  if (interpolates()) {
    if (auto ok = alloc(middle_, planes(3, w, h)); !ok) return ok;
  }
  if (gmfss()) return allocate_gmfss();
  if (setup_.optical_flow_grid > 0) {
    const int cells_x = kernels::flow_cells(setup_.rife_width, setup_.optical_flow_grid);
    const int cells_y = kernels::flow_cells(setup_.rife_height, setup_.optical_flow_grid);
    for (MotionBuffers& mid : mids_) {
      if (auto ok = alloc(mid.flow, planes(4, cells_x, cells_y)); !ok) return ok;
      if (auto ok = alloc(mid.mask, planes(1, cells_x, cells_y)); !ok) return ok;
    }
  } else if (interpolates()) {
    const kernels::RifeGeometry g = kernels::rife_geometry(setup_.rife_width, setup_.rife_height);
    if (auto ok = alloc(rife_in_, planes(setup_.rife_planes, g.padded_width, g.padded_height)); !ok)
      return ok;
    if (setup_.rife != nullptr) {
      if (auto ok = alloc(rife_out_, planes(3, g.padded_width, g.padded_height)); !ok) return ok;
    } else {
      for (MotionBuffers& mid : mids_) {
        if (auto ok = alloc(mid.flow, planes(4, g.padded_width, g.padded_height)); !ok) return ok;
        if (auto ok = alloc(mid.mask, planes(1, g.padded_width, g.padded_height)); !ok) return ok;
      }
      if (auto ok = alloc(flow_, planes(4, g.padded_width, g.padded_height)); !ok) return ok;
      if (auto ok = alloc(mask_, planes(1, g.padded_width, g.padded_height)); !ok) return ok;
    }
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
  if (small_[current_]) {  // l'image de RIFE, réduite par Area
    if (auto ok = ops.ingest(frame, gpu::IngestSpec{}, small_[current_]->tensor()); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  Factor factor = interpolates() ? setup_.rife_factor : Factor{};
  if (interpolates() && setup_.live) factor = setup_.live->rife_factor();
  // La grille de sortie de la paire (core/cadence.h), refaite quand le facteur change : la paire
  // suivante repart alors de cette image.
  std::optional<core::CadenceStep> step;
  if (factor.num > factor.den && valid_factor(factor)) {
    if (!cadence_ || cadence_factor_ != factor) {
      auto made = core::CadenceAccumulator::create({.num = factor.num, .den = factor.den});
      if (!made) return std::unexpected(std::move(made.error()));
      cadence_.emplace(*made);
      cadence_factor_ = factor;
    }
    step = cadence_->push(core::stamp_of(frame));
  } else {
    cadence_.reset();
  }
  const bool on_grid = step && step->event == core::GridEvent::Pair && !step->instants().empty() &&
                       !step->instants().front().is_source();
  // ×N demandé, arrondi au-dessus (×1,5 compte pour ×2) ; 1 : pas d'interpolation.
  std::uint32_t wanted = on_grid ? static_cast<std::uint32_t>((factor.num + factor.den - 1) / factor.den) : 1;
  // En retard sur l'horloge maître (GPU débordé : 4K ×3, scène lourde) : le premier instant
  // intermédiaire de cette paire est déjà passé, l'image seule suffit pour rattraper.
  if (on_grid && setup_.clock != nullptr && previous_ && previous_->epoch == frame.epoch) {
    const ClockSnapshot now = setup_.clock->snapshot();
    const std::optional<std::int64_t> first = step->instants().front().pts_us;
    if (now.valid && now.epoch == frame.epoch && now.rate > 0.0 && first &&
        now.media_at(host_now_us()) > *first) {
      wanted = 1;
      ++late_pairs_;
      if (setup_.live) setup_.live->count_late();
    }
  }
  PairFacts facts{.factor = wanted,
                  .same_epoch = previous_ && previous_->epoch == frame.epoch,
                  .gap_us = previous_ ? frame.pts_us - previous_->pts_us : 0};
  if (facts.factor > 1 && facts.same_epoch) {
    const auto asked = std::chrono::steady_clock::now();
    auto scores = ops.scores_now(rife_source(current_ ^ 1U), rife_source(current_));
    scores_wait_us_.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - asked)
            .count());
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
  // L'image seule si l'interpolation ne tourne pas ; la grille repart alors d'elle.
  outputs_ = 1;
  intermediates_ = 0;
  instants_[0] = Instant{.t = 1.0F, .pts_us = frame.pts_us};
  const bool runs = on_grid && rife_runs(facts);
  if (on_grid && !runs && cadence_) cadence_->resync();
  if (runs && step) {
    outputs_ = 0;
    for (const core::OutputInstant& at : step->instants()) {
      const bool source = at.is_source();
      instants_[outputs_++] = Instant{
          .t = source ? 1.0F
                      : static_cast<float>(static_cast<double>(at.t.num) / static_cast<double>(at.t.den)),
          .pts_us = source ? frame.pts_us : at.pts_us.value_or(frame.pts_us)};
      if (!source) ++intermediates_;
    }
    ++interpolated_;
    if (setup_.live) setup_.live->count_interpolated();
  }
  // Garder les poses : la paire précédente compte si elle précède vraiment celle-ci ; une paire
  // tenue (dessin identique) n'a pas bougé. Le mouvement de cette paire va dans l'autre tampon.
  before_ = facts.same_epoch ? now_ : PairMotion::Unknown;
  const bool held = facts.scores && !kernels::is_scene_cut(facts.scores->scene_score) &&
                    kernels::is_held(facts.scores->worst_block);
  now_ = held ? PairMotion::Still : PairMotion::Unknown;
  if (intermediates_ > 0) mid_ ^= 1U;
  return outputs_;
}

std::expected<RingStage::Output, Refusal> UpscaleStage::produce(const DecodedFrame& frame,
                                                                std::uint32_t output, std::uint32_t slot,
                                                                std::uint64_t wait_value,
                                                                std::uint64_t signal_value) {
  const gpu::RingWrite sync{.wait_value = wait_value, .signal_value = signal_value};
  const gpu::Tensor& now = rgb_[current_]->tensor();
  const Instant& instant = instants_[std::min<std::size_t>(output, instants_.size() - 1)];
  const bool last = output + 1 >= outputs_;
  if (instant.t >= 1.0F) {  // l'image elle-même, toujours la dernière de la paire
    if (auto ok = finish(now, slot, sync); !ok) return std::unexpected(std::move(ok.error()));
    previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
    return Output{.pts_us = frame.pts_us, .left = frame.id, .right = frame.id};
  }
  // Image intermédiaire, entre la précédente et celle-ci (rife_runs demande la précédente).
  if (!previous_) return refuse(RefusalCode::SequenceOrder, "RIFE : pas d'image précédente");
  const Output made{.pts_us = instant.pts_us, .left = previous_->id, .right = frame.id};
  if (by_motion() || gmfss()) {
    auto ok = gmfss() ? interpolate_gmfss(output) : interpolate_by_motion(output);
    if (!ok) return std::unexpected(std::move(ok.error()));
  } else if (auto ok = interpolate_image(instant.t); !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  if (auto ok = finish(middle_->tensor(), slot, sync); !ok) return std::unexpected(std::move(ok.error()));
  // ×1,5 : l'image n'est pas sur la grille de cette paire, elle sert seulement de suivante.
  if (last) previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
  return made;
}

std::expected<void, Refusal> UpscaleStage::interpolate_image(float t) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (auto ok =
          ops.rife_input(rgb_[current_ ^ 1U]->tensor(), rgb_[current_]->tensor(), t, rife_in_->tensor());
      !ok) {
    return ok;
  }
  const std::array inputs{rife_in_->tensor()};
  std::array results{rife_out_->tensor()};
  if (auto ok = setup_.backend->enqueue(*setup_.rife, inputs, results); !ok) return ok;
  return ops.rife_crop(rife_out_->tensor(), middle_->tensor());
}

std::expected<void, Refusal> UpscaleStage::run_motion(float t, const gpu::Tensor& flow,
                                                      const gpu::Tensor& mask) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (auto ok = ops.rife_input(rife_source(current_ ^ 1U), rife_source(current_), t, rife_in_->tensor()); !ok)
    return ok;
  const std::array inputs{rife_in_->tensor()};
  std::array results{flow, mask};
  ++rife_runs_;
  return setup_.backend->enqueue(*setup_.rife_motion, inputs, results);
}

gpu::Motion UpscaleStage::mid_motion(std::uint32_t index) const {
  const MotionBuffers& mid = mids_[index];
  if (setup_.optical_flow_grid > 0) {  // le flux optique couvre toute sa grille
    const gpu::Shape& cells = mid.flow->tensor().shape;
    return {.flow = mid.flow->tensor(),
            .mask = mid.mask->tensor(),
            .area = kernels::Window{.x = 0,
                                    .y = 0,
                                    .width = static_cast<int>(cells.dims[3]),
                                    .height = static_cast<int>(cells.dims[2])}};
  }
  // RIFE estime à sa taille, dans ses plans remplis ; le warp met le mouvement à l'échelle.
  const kernels::RifeGeometry g = kernels::rife_geometry(setup_.rife_width, setup_.rife_height);
  return {.flow = mid.flow->tensor(),
          .mask = mid.mask->tensor(),
          .area = kernels::Window{
              .x = g.left, .y = g.top, .width = setup_.rife_width, .height = setup_.rife_height}};
}

std::expected<void, Refusal> UpscaleStage::interpolate_by_motion(std::uint32_t output) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  const float t = instants_[output].t;
  const gpu::Tensor& a = rgb_[current_ ^ 1U]->tensor();
  const gpu::Tensor& b = rgb_[current_]->tensor();
  const gpu::Motion middle = mid_motion(mid_);
  if (output == 0) {
    poses_ = setup_.live && setup_.live->keep_poses();
    single_run_ = false;
    if (setup_.optical_flow_grid > 0) {
      if (auto ok = ops.optical_flow(rife_source(current_ ^ 1U), rife_source(current_), middle); !ok)
        return ok;
      ++optical_flow_pairs_;
    } else {
      // RIFE au milieu ; s'il reste plus d'une image à faire, son mouvement maximal dit si cette
      // passe suffit (kernels/spec/warp.h, seuil mis à la hauteur de RIFE). Garder les poses
      // demande le mouvement du milieu à chaque instant : une seule passe.
      if (auto ok = run_motion(0.5F, middle.flow, middle.mask); !ok) return ok;
      if (poses_) {
        single_run_ = true;
      } else if (intermediates_ > 1) {
        auto biggest = ops.flow_max_now(middle);
        if (!biggest) return std::unexpected(std::move(biggest.error()));
        single_run_ = *biggest < kernels::single_run_motion(setup_.rife_height);
        if (single_run_) {
          ++single_pairs_;
        } else {
          ++multi_pairs_;
        }
      }
    }
    now_ = PairMotion::Moving;
  }
  if (poses_ && before_ != PairMotion::Unknown) {
    const gpu::Motion previous = mid_motion(mid_ ^ 1U);
    return ops.warp_blend_poses(a, b, middle, before_ == PairMotion::Moving ? &previous : nullptr, t,
                                middle_->tensor());
  }
  if (setup_.optical_flow_grid > 0 || single_run_ || t == 0.5F)
    return ops.warp_blend(a, b, middle, t, middle_->tensor());
  // Mouvement fort : RIFE à cet instant, son mouvement tel quel (t = 1/2 au warp : « estimé »).
  const gpu::Motion other{.flow = flow_->tensor(), .mask = mask_->tensor(), .area = middle.area};
  if (auto ok = run_motion(t, other.flow, other.mask); !ok) return ok;
  return ops.warp_blend(a, b, other, 0.5F, middle_->tensor());
}

std::expected<void, Refusal> UpscaleStage::finish(const gpu::Tensor& rgb, std::uint32_t slot,
                                                  const gpu::RingWrite& sync) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (setup_.true_hdr && setup_.live) {
    if (const auto params = setup_.live->take_true_hdr()) {
      if (auto set = ops.set_true_hdr(*params); !set) {
        log_channel("gpu")->warn("RTX HDR : réglages refusés ({})", set.error().detail);
      } else {
        black_ramp_pending_ = true;
      }
    }
    // Vrai noir : la rampe repasse par TrueHDR après un changement, au plus 4 fois par seconde
    // (un curseur qu'on fait glisser n'en lance pas une à chaque image).
    const std::int64_t now = host_now_us();
    if (black_ramp_pending_ && setup_.live->black_wanted() && now - black_ramp_at_us_ >= kBlackRampGapUs) {
      black_ramp_pending_ = false;
      black_ramp_at_us_ = now;
      const auto codes = black_ramp_codes();
      if (auto ramp = ops.true_hdr_ramp(codes)) {
        setup_.live->set_black_ramp(std::move(*ramp));
      } else {
        log_channel("gpu")->warn("vrai noir : rampe TrueHDR ({})", ramp.error().detail);
      }
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
