// L'étage gpu (upscale_stage.h) : sa construction, ses tampons et son bilan au journal. Sorti
// d'upscale_stage.cpp (C09-09).
#include <algorithm>
#include <array>
#include <format>
#include <utility>

#include "app/play/options.h"
#include "app/play/rife_plan.h"
#include "app/play/upscale_stage.h"
#include "core/log.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {
namespace {

gpu::Shape planes(int channels, int width, int height) {
  return gpu::Shape{.dims = {1, channels, height, width}};
}

}  // namespace

UpscaleStage::UpscaleStage(VideoStageInit init, UpscaleSetup setup)
    : RingStage(std::move(init), static_cast<std::uint32_t>(setup.slots.size()),
                setup.slots.front().generation, smooth_extra_frames(setup.smooth_drawings)),
      setup_(std::move(setup)) {}

UpscaleStage::~UpscaleStage() {
  stop();
  if (fences_mapped_) setup_.backend->kernels()->unmap_fences(setup_.yuv_slots->fence_block());
  if (setup_.rife_motion != nullptr) {
    log_channel("gpu")->info(
        "RIFE mouvement : {} passes ; {} paires à une passe, {} à plusieurs (dont {} sans passe au milieu)",
        rife_runs_, single_pairs_, multi_pairs_, direct_pairs_);
  }
  if (setup_.rife_motion_calm != nullptr)
    log_channel("gpu")->info("RIFE des paires calmes : {} passes sur {}", calm_runs_, rife_runs_);
  if (halves())
    log_channel("gpu")->info("échelle par paire : {} passes à demi-taille sur {}", half_.runs, rife_runs_);
  // RIFE à chaud, un compromis qualité contre vitesse (B45-07) : le milieu d'une paire calme qui accélère
  // garde l'échelle grossière de la paire d'avant. Ni le gain ni la perte ne sont mesurés : l'essai reste
  // hors défaut.
  if (setup_.rife_warm)
    log_channel("gpu")->info(
        "RIFE à chaud (qualité et gain non mesurés) : {} passes sur {} sans les blocs "
        "grossiers{}",
        warm_runs_, rife_runs_,
        warm_runs_ == 0 && rife_runs_ > 0
            ? " (ce moteur ne reprend pas à chaud : l'essai est sans effet hors du RIFE "
              "fusionné de Vulkan)"
            : "");
  if (moving_.pairs > 0) {
    const auto pairs = static_cast<double>(moving_.pairs);
    log_channel("gpu")->info(
        "Zone qui bouge sur {} paires : blocs changés {:.0f} %, boîte {:.0f} %, boîte sous la moitié {} fois",
        moving_.pairs, 100.0 * moving_.block_sum / pairs, 100.0 * moving_.box_sum / pairs,
        moving_.small_boxes);
  }
  if (spans_reused_ > 0)
    log_channel("gpu")->info("SPAN : {} passes évitées sur des dessins tenus", spans_reused_);
  if (encodes_kept_ + encodes_followed_ + encodes_after_gap_ + encodes_cold_ > 0)
    log_channel("gpu")->info(
        "RIFE : encodage repris sur {} passes, B seule sur {}, A et B sur {} après une paire sans passe et "
        "{} "
        "à froid",
        encodes_kept_, encodes_followed_, encodes_after_gap_, encodes_cold_);
  if (drba()) {
    log_channel("gpu")->info(
        "DistilDRBA : {} passes, dont {} sans l'image d'un côté (début, coupe) ; {} instants sans passe "
        "(côté tenu)",
        drba_.runs, drba_.edges, drba_.held);
  }
  if (setup_.smooth_drawings) {
    const auto& h = smoothed_holds_;
    log_channel("gpu")->info(
        "Smooth : {} tenues fondues en un mouvement (sur 2 sources : {}, 3 : {}, 4 : {}, 5 : {}, 6 : {})",
        h[2] + h[3] + h[4] + h[5] + h[6], h[2], h[3], h[4], h[5], h[6]);
  }
  if (setup_.late_fate) {
    log_channel("gpu")->info(
        "Rapide : {} instants laissés en retard sans calcul, {} calculés après leur instant", late_dropped_,
        late_computed_);
  }
  if (late_pairs_ > 0) {
    log_channel("gpu")->info("RIFE : {} paires en retard sur l'horloge, {} instants sautés pour la rattraper",
                             late_pairs_, late_instants_);
  }
  if (skipped_ > 0) {
    log_channel("gpu")->info(
        "{} images source sautées avant calcul, déjà passées : la chaîne est plus lente "
        "que la vidéo",
        skipped_);
  }
  if (interpolates()) {
    log_channel("gpu")->info("RIFE {} : {} paires, {} interpolées, {} coupes de scène, {} dessins tenus",
                             factor_text(setup_.live ? setup_.live->factor_now() : setup_.rife_factor),
                             pairs_, interpolated_, cuts_, held_);
  }
  if (setup_.live && setup_.live->cuts().active())
    log_channel("gpu")->info("coupes : {} désaccords entre RIFE et les plans de la pré-analyse",
                             cut_disagreements_);
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
  const int estimators = (setup.rife != nullptr ? 1 : 0) + (setup.rife_motion != nullptr ? 1 : 0) +
                         (setup.distil_drba != nullptr ? 1 : 0);
  if (estimators > 1) {
    return refuse(RefusalCode::FieldOutOfRange,
                  "étage gpu : RIFE image, RIFE mouvement ou DistilDRBA, un seul des trois");
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
  if ((setup.rife != nullptr || (setup.distil_drba != nullptr && !setup.drba_motion)) &&
      (setup.rife_width != setup.work_width || setup.rife_height != setup.work_height)) {
    return refuse(RefusalCode::ShapeInvalid,
                  "étage gpu : RIFE image et DistilDRBA image à la taille de travail seulement");
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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::expected<void, Refusal> UpscaleStage::allocate() {
  gpu::Backend& backend = *setup_.backend;
  const gpu::BackendCaps caps = backend.capabilities();
  copies_ = caps.lanes > 1 && !setup_.one_lane ? 2U : 1U;
  if (setup_.yuv_slots && !caps.yuv_targets) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("étage gpu : le backend {} n'écrit pas en YUV pour l'encodeur", caps.name));
  }
  ring_direct_ = caps.ring_targets && !setup_.yuv_slots &&
                 (warps_spanned() || (by_motion() && setup_.engine == nullptr));
  if (ring_direct_)
    log_channel("gpu")->info("images intermédiaires écrites directement dans le slot du ring");
  if (setup_.yuv_slots) {
    log_channel("gpu")->info("images écrites en YUV dans celles de l'encodeur");
    const std::span<std::byte> block = setup_.yuv_slots->fence_block();
    fences_mapped_ = !block.empty() && backend.kernels()->map_fences(block);
    if (fences_mapped_) log_channel("gpu")->info("signaux de l'export écrits par le GPU, sans rappel hôte");
  }
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
  const gpu::Shape out_shape = planes(3, w * kSpanFactor, h * kSpanFactor);
  if (warps_spanned()) {
    for (auto& spanned : spanned_) {
      if (auto ok = alloc(spanned, out_shape); !ok) return ok;
    }
    for (std::uint32_t i = 0; i < copies_ && !ring_direct_; ++i) {
      if (auto ok = alloc(middle_outs_[i], out_shape); !ok) return ok;
    }
  } else if (setup_.engine != nullptr) {
    if (auto ok = alloc(output_, out_shape); !ok) return ok;
  }
  if (interpolates() && (setup_.rife_width != w || setup_.rife_height != h)) {
    for (auto& small : small_) {
      if (auto ok = alloc(small, planes(3, setup_.rife_width, setup_.rife_height)); !ok) return ok;
    }
  }
  if (interpolates() && !warps_spanned() && !ring_direct_) {
    for (std::uint32_t i = 0; i < copies_; ++i) {
      if (auto ok = alloc(middles_[i], planes(3, w, h)); !ok) return ok;
    }
  }
  if (looks_ahead()) {
    if (auto ok = allocate_ahead(); !ok) return ok;
  }
  if (drba()) return allocate_drba();
  if (interpolates()) {
    const kernels::RifeGeometry g =
        kernels::rife_geometry(setup_.rife_width, setup_.rife_height, setup_.rife_align);
    for (std::uint32_t i = 0; i < copies_; ++i) {
      if (auto ok = alloc(rife_ins_[i], planes(setup_.rife_planes, g.padded_width, g.padded_height)); !ok)
        return ok;
    }
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
  return allocate_half();
}

}  // namespace ajp::app::play
