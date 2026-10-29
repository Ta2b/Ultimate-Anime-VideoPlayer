#include "app/play/upscale_stage.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <utility>

#include "app/play/options.h"
#include "app/play/rife_plan.h"
#include "core/auto_rate.h"
#include "core/clock.h"
#include "core/log.h"
#include "kernels/spec/detect.h"
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
    log_channel("gpu")->info(
        "RIFE mouvement : {} passes ; {} paires à une passe, {} à plusieurs (dont {} sans passe au milieu)",
        rife_runs_, single_pairs_, multi_pairs_, direct_pairs_);
  }
  if (gmfss()) log_channel("gpu")->info("GMFSS : {} paires", gmfss_pairs_);
  if (gmfss_fused()) {
    log_channel("gpu")->info(
        "GMFSS fusionné : caractéristiques de {} images calculées, {} reprises de la paire d'avant",
        gmfss_.features_of.computed(), gmfss_.features_of.reused());
  }
  if (spans_reused_ > 0)
    log_channel("gpu")->info("SPAN : {} passes évitées sur des dessins tenus", spans_reused_);
  if (drba()) {
    log_channel("gpu")->info("DistilDRBA : {} passes, dont {} sans l'image d'un côté (début, coupe)",
                             drba_.runs, drba_.edges);
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
  if (setup.gmfss_features != nullptr && setup.gmfss_pair == nullptr)
    return refuse(RefusalCode::FieldMissing,
                  "étage gpu : les caractéristiques de GMFSS sans ses deux moteurs");
  const int estimators = (setup.rife != nullptr ? 1 : 0) + (setup.rife_motion != nullptr ? 1 : 0) +
                         (setup.gmfss_pair != nullptr ? 1 : 0) + (setup.distil_drba != nullptr ? 1 : 0);
  if (estimators > 1) {
    return refuse(RefusalCode::FieldOutOfRange,
                  "étage gpu : RIFE image, RIFE mouvement, GMFSS ou DistilDRBA, un seul des quatre");
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
  if ((setup.rife != nullptr || setup.gmfss_pair != nullptr || setup.distil_drba != nullptr) &&
      (setup.rife_width != setup.work_width || setup.rife_height != setup.work_height)) {
    return refuse(RefusalCode::ShapeInvalid,
                  "étage gpu : RIFE image, GMFSS et DistilDRBA à la taille de travail seulement");
  }
  if (setup.yuv_slots && setup.true_hdr)
    return refuse(RefusalCode::FieldOutOfRange, "étage gpu : export en YUV sans RTX HDR seulement");
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
  const gpu::BackendCaps caps = backend.capabilities();
  copies_ = caps.lanes > 1 && !setup_.one_lane ? 2U : 1U;
  if (setup_.yuv_slots && !caps.yuv_targets) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("étage gpu : le backend {} n'écrit pas en YUV pour l'encodeur", caps.name));
  }
  ring_direct_ = caps.ring_targets && !setup_.true_hdr && !setup_.yuv_slots &&
                 (warps_spanned() || (by_motion() && setup_.engine == nullptr));
  if (ring_direct_)
    log_channel("gpu")->info("images intermédiaires écrites directement dans le slot du ring");
  if (setup_.yuv_slots) log_channel("gpu")->info("images écrites en YUV dans celles de l'encodeur");
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
  if (gmfss()) return allocate_gmfss();
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
  use_lane(gpu::Lane::Measure);  // ingestion, mesures de la paire, suivante en avance
  current_ ^= 1U;
  rife_in_pair_ = false;  // les images de rife_in() sont celles de la paire d'avant
  gpu::IngestSpec spec;
  if (setup_.live) spec.reduce = setup_.live->reduction();
  const bool ingested = looks_ahead() && take_ahead(frame, spec);  // déjà, derrière la précédente
  if (drba()) {
    if (auto ok = drba_arrive(frame, spec); !ok) return std::unexpected(std::move(ok.error()));
  } else if (!ingested) {
    if (auto ok = ops.ingest(frame, spec, rgb_[current_]->tensor()); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  if (small_[current_] && !ingested) {  // l'image de RIFE, réduite par Area
    if (auto ok = ops.ingest(frame, gpu::IngestSpec{}, small_[current_]->tensor()); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  Factor factor = interpolates() ? setup_.rife_factor : Factor{};
  if (interpolates() && setup_.live) factor = setup_.live->factor_now();
  // La grille de sortie de la paire (core/cadence.h), refaite quand le facteur change : la paire
  // suivante repart alors de cette image.
  std::optional<core::CadenceStep> step;
  // Auto libre : pas de facteur, la grille de la paire vient de l'horloge prédite et du temps de service.
  const bool free_mode = interpolates() && setup_.live && setup_.live->free_active();
  if (free_mode) {
    cadence_.reset();
    step = free_grid_.push(core::stamp_of(frame), free_inputs(frame.epoch));
  } else if (factor.num > factor.den &&
             (valid_factor(factor) || (setup_.live != nullptr && factor.den <= 64 &&
                                       factor.num <= core::kAutoMaxStep * factor.den))) {
    if (!cadence_) {
      auto made = core::CadenceAccumulator::create({.num = factor.num, .den = factor.den});
      if (!made) return std::unexpected(std::move(made.error()));
      cadence_.emplace(*made);
      cadence_factor_ = factor;
    } else if (cadence_factor_ != factor) {
      // Le mode auto bouge le facteur par petits pas : la grille garde sa phase (aucun saut de rythme).
      if (auto moved = cadence_->retarget({.num = factor.num, .den = factor.den}); !moved)
        return std::unexpected(std::move(moved.error()));
      cadence_factor_ = factor;
    }
    step = cadence_->push(core::stamp_of(frame));
  } else {
    cadence_.reset();
  }
  if (!free_mode) free_grid_.reset();
  const bool on_grid = step && step->event == core::GridEvent::Pair && !step->instants().empty() &&
                       !step->instants().front().is_source();
  // ×N demandé, arrondi au-dessus (×1,5 compte pour ×2) ; 1 : pas d'interpolation.
  std::uint32_t wanted = on_grid ? static_cast<std::uint32_t>((factor.num + factor.den - 1) / factor.den) : 1;
  if (free_mode) wanted = on_grid ? 2U : 1U;
  // En retard sur l'horloge maître (GPU débordé : 4K ×3, scène lourde) : les instants
  // intermédiaires de tête qui seront passés quand ils seront prêts (une période de sortie plus
  // loin, ou après le retard mesuré des images) sont sautés, les suivants faits ; l'image seule
  // s'il n'en reste aucun.
  std::size_t late = 0;
  double play_rate = 1.0;
  if (!free_mode && on_grid && setup_.clock != nullptr && previous_ && previous_->epoch == frame.epoch) {
    const ClockSnapshot now = setup_.clock->snapshot();
    if (now.valid && now.epoch == frame.epoch && now.rate > 0.0) {
      play_rate = now.rate;
      const std::int64_t period =
          std::max<std::int64_t>(frame.pts_us - previous_->pts_us, 0) * factor.den / factor.num;
      const std::int64_t ready = ready_media_us(now, host_now_us(), period, latency_us());
      std::size_t middles = 0;
      for (const core::OutputInstant& at : step->instants()) {
        if (at.is_source()) continue;
        if (late == middles && at.pts_us && *at.pts_us < ready) ++late;
        ++middles;
      }
      if (late > 0) {
        ++late_pairs_;
        late_instants_ += late;
        if (setup_.live) setup_.live->count_late();
      }
      if (middles > 0 && late == middles) wanted = 1;
    }
  }
  PairFacts facts{.factor = wanted,
                  .same_epoch = previous_ && previous_->epoch == frame.epoch,
                  .gap_us = previous_ && previous_->pts_us != kNoPts && frame.pts_us != kNoPts
                                ? frame.pts_us - previous_->pts_us
                                : 0};
  if (!facts.same_epoch) forget_motion();  // début ou saut : le mouvement d'avant ne dit plus rien
  std::int64_t lead_us = 0;                // attente des mesures de la paire, avant sa soumission
  if (facts.factor > 1 && facts.same_epoch) {
    const auto asked = std::chrono::steady_clock::now();
    auto scores = drba() ? drba_scores(frame) : pair_scores();
    // Les dernières attentes seulement : le journal de fin les résume, la mémoire reste bornée.
    const auto waited =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - asked)
            .count();
    if (scores_wait_us_.size() < kScoresWaitKept) {
      scores_wait_us_.push_back(waited);
    } else {
      scores_wait_us_[scores_wait_next_++ % kScoresWaitKept] = waited;
    }
    lead_us = waited;
    if (!scores) return std::unexpected(std::move(scores.error()));
    facts.scores = *scores;
    ++pairs_;
    if (setup_.live) setup_.live->count_pair();
    if (kernels::is_scene_cut(scores->scene_score)) {
      forget_motion();
      ++cuts_;
      if (setup_.live) {
        setup_.live->count_cut();
        setup_.live->auto_cut(host_now_us());
      }
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
  // Mode auto : seules les paires qui ont chargé le GPU disent s'il tient ; à ×1 rien ne tourne, le
  // temps seul compte. Coupe de scène ou dessin tenu : aucun verdict, mais le moteur vit.
  const bool auto_measured =
      interpolates() && setup_.live && facts.same_epoch && (runs || factor.num <= factor.den);
  if (interpolates() && setup_.live) {
    const std::int64_t at_us = host_now_us();
    if (!facts.same_epoch) {
      setup_.live->auto_settle(at_us);
    } else if (auto_measured) {
      const std::int64_t gap = facts.gap_us;
      setup_.live->auto_pair(
          {.now_us = at_us,
           .made_step = factor.num,
           .late = static_cast<int>(late),
           .period_us = gap >= 8'000 && gap <= 200'000
                            ? static_cast<std::int64_t>(static_cast<double>(gap) / play_rate)
                            : 0});
    } else {
      setup_.live->auto_alive(at_us);
    }
  }
  // Coupe ou dessin tenu : les instants de la paire manquent à dessein, pas faute de temps.
  uninterpolated_ =
      on_grid && !runs && facts.factor > 1 && facts.scores &&
      (kernels::is_scene_cut(facts.scores->scene_score) || kernels::is_held(facts.scores->worst_block));
  if (on_grid && !runs && cadence_) cadence_->resync();
  if (on_grid && !runs && free_mode) free_grid_.resync();
  if (runs && step) {
    outputs_ = 0;
    std::size_t skipped = 0;
    for (const core::OutputInstant& at : step->instants()) {
      const bool source = at.is_source();
      if (!source && skipped < late) {  // déjà en retard : sauté
        ++skipped;
        continue;
      }
      instants_[outputs_++] = Instant{
          .t = source ? 1.0F
                      : static_cast<float>(static_cast<double>(at.t.num) / static_cast<double>(at.t.den)),
          .pts_us = source ? frame.pts_us : at.pts_us.value_or(frame.pts_us)};
      if (!source) ++intermediates_;
    }
    ++interpolated_;
    if (setup_.live) setup_.live->count_interpolated();
  }
  if (auto_measured) setup_.live->auto_expect(frame.id.value, static_cast<int>(outputs_), lead_us);
  if (free_mode && facts.same_epoch) {
    setup_.live->auto_expect(frame.id.value, static_cast<int>(outputs_), lead_us);
    setup_.live->free_produced(static_cast<int>(outputs_), host_now_us());
  }
  // Garder les poses : la paire précédente compte si elle précède vraiment celle-ci ; une paire
  // tenue (dessin identique) n'a pas bougé. Le mouvement de cette paire va dans l'autre tampon.
  before_ = facts.same_epoch ? now_ : PairMotion::Unknown;
  const bool held = facts.scores && !kernels::is_scene_cut(facts.scores->scene_score) &&
                    kernels::is_held(facts.scores->worst_block);
  now_ = held ? PairMotion::Still : PairMotion::Unknown;
  // Dessin tenu : l'image agrandie de la précédente sert à celle-ci, sans passe de SPAN. Pas plus
  // de kMaxSpanReuse de suite, comptées depuis la dernière vraie passe : un fondu lent, tenu paire
  // par paire, ne fige pas l'image.
  reuse_span_ = setup_.engine != nullptr && held && spans_reused_in_row_ < kMaxSpanReuse &&
                (warps_spanned() || output_has_source_);
  spans_reused_in_row_ = reuse_span_ ? spans_reused_in_row_ + 1 : 0;
  if (reuse_span_) ++spans_reused_;
  if (intermediates_ > 0) mid_ ^= 1U;
  if (intermediates_ > 0 && (setup_.rife != nullptr || by_motion())) {
    if (auto ok = prepare_rife_input(); !ok) return std::unexpected(std::move(ok.error()));
  }
  if (drba()) {
    // Mesurée et sans coupe, la paire laisse a comme image d'avant à la paire suivante.
    drba_.continuous = facts.scores.has_value() && !kernels::is_scene_cut(facts.scores->scene_score);
    if (auto ok = drba_next(frame, spec, facts.factor > 1); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  spanned_now_ = false;  // SPAN de cette image : à sa première utilisation (span_current)
  // Avant RIFE et SPAN de cette image : la suivante, si elle attend, ingérée et mesurée.
  ahead_.ticket = 0;
  if (looks_ahead()) {
    if (auto ok = look_ahead(frame, spec); !ok) return std::unexpected(std::move(ok.error()));
  }
  return outputs_;
}

// Ce que la grille libre sait de l'instant : l'horloge maître prédite à la fin du calcul (retard vidéo mesuré
// compris), l'écart minimal de l'écran et le temps de service mesuré, en temps de contenu.
core::FreeInputs UpscaleStage::free_inputs(std::uint32_t epoch) const {
  core::FreeInputs inputs;
  const std::int64_t at = host_now_us();
  const auto [step_real, gap_real] = setup_.live->free_step(at);
  double rate = 1.0;
  const ClockSnapshot now = setup_.clock != nullptr ? setup_.clock->snapshot() : ClockSnapshot{};
  const bool running = now.valid && now.epoch == epoch && now.rate > 0.0;
  if (running) rate = now.rate;
  const auto media = [rate](std::int64_t real_us) {
    return static_cast<std::int64_t>(std::llround(static_cast<double>(real_us) * rate));
  };
  inputs.min_gap_us = std::max<std::int64_t>(media(gap_real), 1);
  inputs.step_us = media(step_real);
  inputs.max_instants = free_instants_limit(engine_kind());
  if (running) inputs.ready_us = ready_media_us(now, at, inputs.min_gap_us, latency_us());
  return inputs;
}

bool UpscaleStage::passed(const DecodedFrame& frame, const DecodedFrame& next) {
  if (setup_.clock == nullptr || frame.pts_us == kNoPts || next.pts_us == kNoPts) return false;
  const ClockSnapshot now = setup_.clock->snapshot();
  if (!now.valid || now.epoch != frame.epoch || now.rate <= 0.0) return false;
  // L'image d'après sera passée quand celle-ci serait prête : celle-ci ne paraîtrait jamais.
  if (next.pts_us > ready_media_us(now, host_now_us(), 0, latency_us())) return false;
  // La suite repart de l'image d'après, comme au début d'une époque : ni paire, ni grille de
  // cadence, ni fenêtre de DistilDRBA ne passent par-dessus l'image sautée.
  previous_.reset();
  cadence_.reset();
  free_grid_.reset();
  drba_.continuous = false;
  ++skipped_;
  if (setup_.live) setup_.live->count_skipped();
  return true;
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
    use_lane(gpu::Lane::Upscale);
    std::expected<void, Refusal> ok;
    if (warps_spanned()) {
      ok = span_current().and_then([&] { return write_slot(spanned(current_), slot, sync); });
    } else if (reuse_span_ && output_has_source_) {
      ok = write_slot(output_->tensor(), slot, sync);  // SPAN de la précédente, identique
    } else {
      ok = finish(now, slot, sync);
    }
    if (!ok) return std::unexpected(std::move(ok.error()));
    output_has_source_ = setup_.engine != nullptr;
    previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
    return Output{
        .pts_us = frame.pts_us, .left = frame.id, .right = frame.id, .uninterpolated = uninterpolated_};
  }
  // Image intermédiaire, entre la précédente et celle-ci (rife_runs demande la précédente).
  if (!previous_) return refuse(RefusalCode::SequenceOrder, "RIFE : pas d'image précédente");
  const Output made{.pts_us = instant.pts_us, .left = previous_->id, .right = frame.id};
  gmfss_.a = GmfssFrameKey{.id = previous_->id,
                           .epoch = previous_->epoch};  // GMFSS fusionné : les images de la paire
  gmfss_.b = GmfssFrameKey{.id = frame.id, .epoch = frame.epoch};
  use_lane(gpu::Lane::Interpolate);
  next_middle();  // l'autre tampon : SPAN ou l'écriture du ring lit peut-être encore le précédent
  const gpu::RingTarget ring{.slot = &setup_.slots[slot], .sync = sync};
  if (warps_spanned()) {
    const Target out = ring_direct_ ? Target{.ring = &ring} : Target{.tensor = &middle_out()};
    if (auto ok = gmfss() ? interpolate_gmfss_spanned(output, spanned(current_ ^ 1U), spanned(current_), out)
                          : interpolate_by_motion(output, spanned(current_ ^ 1U), spanned(current_), out);
        !ok) {
      return std::unexpected(std::move(ok.error()));
    }
    use_lane(gpu::Lane::Upscale);
    if (out.tensor != nullptr) {
      if (auto ok = write_slot(*out.tensor, slot, sync); !ok) return std::unexpected(std::move(ok.error()));
    }
    if (last) previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
    return made;
  }
  if (by_motion() && ring_direct_) {  // sans SPAN : le warp écrit la sortie elle-même
    if (auto ok = interpolate_by_motion(output, rgb_[current_ ^ 1U]->tensor(), now, Target{.ring = &ring});
        !ok)
      return std::unexpected(std::move(ok.error()));
    use_lane(gpu::Lane::Upscale);
    output_has_source_ = false;
    if (last) previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
    return made;
  }
  if (by_motion() || gmfss() || drba()) {
    auto ok = gmfss()  ? interpolate_gmfss(output)
              : drba() ? interpolate_drba(output)
                       : interpolate_by_motion(output, rgb_[current_ ^ 1U]->tensor(), now,
                                               Target{.tensor = &middle()});
    if (!ok) return std::unexpected(std::move(ok.error()));
  } else if (auto ok = interpolate_image(instant.t); !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  use_lane(gpu::Lane::Upscale);
  if (auto ok = finish(middle(), slot, sync); !ok) return std::unexpected(std::move(ok.error()));
  output_has_source_ = false;
  // ×1,5 : l'image n'est pas sur la grille de cette paire, elle sert seulement de suivante.
  if (last) previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
  return made;
}

std::expected<void, Refusal> UpscaleStage::span_current() {
  if (spanned_now_) return {};
  spanned_now_ = true;
  if (reuse_span_) {  // dessin tenu : le tampon de la précédente
    span_of_[current_] = span_of_[current_ ^ 1U];
    return {};
  }
  // L'autre tampon que celui de la précédente, qui reste le bord a des warps de la paire.
  span_of_[current_] = span_of_[current_ ^ 1U] ^ 1U;
  // Sur la voie de SPAN, au milieu du travail de la paire : la voie de l'appelant reprend après.
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Upscale);
  const std::array inputs{rgb_[current_]->tensor()};
  std::array results{spanned(current_)};
  auto ok = setup_.backend->enqueue(*setup_.engine, inputs, results);
  use_lane(back);
  return ok;
}

std::expected<void, Refusal> UpscaleStage::finish(const gpu::Tensor& rgb, std::uint32_t slot,
                                                  const gpu::RingWrite& sync) {
  if (setup_.engine == nullptr) return write_slot(rgb, slot, sync);
  const std::array inputs{rgb};
  std::array outputs{output_->tensor()};
  if (auto ok = setup_.backend->enqueue(*setup_.engine, inputs, outputs); !ok) return ok;
  return write_slot(output_->tensor(), slot, sync);
}

std::expected<void, Refusal> UpscaleStage::write_slot(const gpu::Tensor& image, std::uint32_t slot,
                                                      const gpu::RingWrite& sync) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (setup_.yuv_slots) return ops.write_yuv(image, setup_.yuv_slots->target(slot, sync.signal_value));
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
  return setup_.true_hdr ? ops.write_ring_true_hdr(image, prepared_, setup_.slots[slot], sync)
                         : ops.write_ring(image, setup_.slots[slot], sync);
}

}  // namespace ajp::app::play
