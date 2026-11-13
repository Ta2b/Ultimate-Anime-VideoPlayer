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
#include "core/trace.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/rife.h"
#include "kernels/spec/warp.h"

namespace ajp::app::play {
namespace {

// Un facteur de l'Auto hors de la liste des facteurs réglables : palier / granularité (24 au plus,
// core::kFineGranularity), ou sur écran fixe la fraction exacte d'un palier calé (11011/3200 à 165 Hz,
// core::AutoLimits::exact), dont les termes tiennent dans ceux de l'accumulateur de cadence.
constexpr std::int64_t kLiveFactorMaxDen = core::limits::kRatioTerm.hi;
// L'écart d'une paire mesurée qui donne une période : de 125 à 5 images par seconde. Hors de là (seek, pause,
// image en double), la paire ne dit rien de la cadence.
constexpr std::int64_t kPairGapMinUs = 8'000;
constexpr std::int64_t kPairGapMaxUs = 200'000;

}  // namespace

void UpscaleStage::analyze_source(const DecodedFrame& frame) {
  if (!frame_stats_ || !setup_.live) return;
  gpu::KernelOps& ops = *setup_.backend->kernels();
  // Les analyses finies passent aux scènes, dans l'ordre ; une analyse refusée est jetée et la suivante
  // continue.
  while (!stats_pending_.empty()) {
    auto taken = ops.frame_stats_take(stats_pending_.front().ticket);
    if (!taken) {
      stats_pending_.erase(stats_pending_.begin());
      continue;
    }
    if (!*taken) break;
    useful_area_ = ((*taken)->right - (*taken)->left) * ((*taken)->bottom - (*taken)->top);
    setup_.live->scenes().push(stats_pending_.front().pts_us, **taken, stats_pending_.front().transfer);
    stats_pending_.erase(stats_pending_.begin());
  }
  // Deux relectures attendent encore : la plus ancienne est abandonnée. Son image manque aux scènes ; le
  // limiteur de flashs l'apprend (A42-18), sinon une image claire perdue effaçait deux transitions.
  if (stats_pending_.size() >= gpu::kPendingReadbacks) {
    setup_.live->scenes().missed(stats_pending_.front().pts_us);
    if (stats_dropped_++ == 0)
      log_channel("gpu")->info(
          "analyse des images : une relecture en retard abandonnée (les suivantes ne "
          "sont plus journalisées)");
    stats_pending_.erase(stats_pending_.begin());
  }
  const kernels::FrameTransfer transfer = frame.color.transfer == Transfer::Pq ? kernels::FrameTransfer::Pq
                                          : frame.color.transfer == Transfer::Hlg
                                              ? kernels::FrameTransfer::Hlg
                                              : kernels::FrameTransfer::Sdr;
  auto ticket = ops.frame_stats_start(rgb_[current_]->tensor(), transfer);
  if (!ticket) {
    frame_stats_ = false;
    stats_pending_.clear();
    return;
  }
  stats_pending_.push_back({.ticket = *ticket, .pts_us = frame.pts_us, .transfer = transfer});
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
std::expected<std::uint32_t, Refusal> UpscaleStage::outputs(const DecodedFrame& frame) {
  AJP_TRACE("upscale.outputs");
  if (frame.width != setup_.source_width || frame.height != setup_.source_height) {
    return refuse(RefusalCode::SourceResized,
                  std::format("{}x{} (la chaîne était en {}x{})", frame.width, frame.height,
                              setup_.source_width, setup_.source_height));
  }
  gpu::KernelOps& ops = *setup_.backend->kernels();
  use_lane(gpu::Lane::Measure);  // ingestion, mesures de la paire, suivante en avance
  next_source();
  frame_id_ = frame.id;
  rife_in_pair_ = false;  // les images de rife_in() sont celles de la paire d'avant
  half_.in_pair = false;
  gpu::IngestSpec spec;
  if (setup_.live) spec.reduce = setup_.live->reduction();
  const bool ingested = looks_ahead() && take_ahead(frame, spec);  // déjà, derrière la précédente
  if (drba()) {
    if (auto ok = drba_arrive(frame, spec); !ok) return std::unexpected(std::move(ok.error()));
  } else if (!ingested) {
    auto debanded = ingest_source(frame, spec, rgb_[current_]->tensor());
    if (!debanded) return std::unexpected(std::move(debanded.error()));
    debanded_ = *debanded;
  }
  analyze_source(frame);
  if (small_[current_] && !ingested && !std::exchange(drba_.small_taken, false)) {  // réduite par Area
    if (auto ok = ops.ingest(frame, gpu::IngestSpec{}, small_[current_]->tensor()); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  Factor factor = interpolates() ? setup_.rife_factor : Factor{};
  if (interpolates() && setup_.live) factor = setup_.live->factor_now();
  const bool hidden = setup_.live != nullptr && setup_.live->hidden();  // P4.2 : rien n'est vu
  if (hidden) factor = Factor{};
  // La grille de sortie de la paire (core/cadence.h), refaite quand le facteur change : la paire
  // suivante repart alors de cette image.
  std::optional<core::CadenceStep> step;
  // mode Rapide : pas de facteur, la grille de la paire vient de l'horloge prédite et du temps de service.
  const bool free_mode = interpolates() && setup_.live && setup_.live->free_active() && !hidden;
  const bool steady = free_mode && setup_.live->steady_active();
  // Mouvement fort anticipé : la mesure du milieu de la paire d'avant, relue avant la grille de celle-ci,
  // dit si elle sera lourde (RIFE à chaque instant) ; la grille libre planifie alors son écart en
  // conséquence.
  heavy_pair_ = false;
  if (free_mode && !steady && by_motion()) {
    if (auto ok = settle_motion(); !ok) return std::unexpected(std::move(ok.error()));
    heavy_pair_ = strong_motion_;
  }
  if (steady) {
    cadence_.reset();
    free_grid_.reset();
    step = steady_grid_.push(core::stamp_of(frame), setup_.live->steady_factor());
  } else if (free_mode) {
    cadence_.reset();
    step = free_grid_.push(core::stamp_of(frame), free_inputs(frame.epoch));
  } else if (factor.num > factor.den &&
             (valid_factor(factor) || (setup_.live != nullptr && factor.den <= kLiveFactorMaxDen &&
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
  if (!steady) steady_grid_.reset();
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
      // A09-06 : une image sans date (kNoPts) ne se soustrait pas, la différence déborderait.
      const std::int64_t period =
          std::max<std::int64_t>(pts_gap_us(previous_->pts_us, frame.pts_us), 0) * factor.den / factor.num;
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
                  .gap_us = previous_ ? pts_gap_us(previous_->pts_us, frame.pts_us) : 0,
                  .useful_area = useful_area_};
  if (!facts.same_epoch) forget_motion();  // début ou saut : le mouvement d'avant ne dit plus rien
  std::int64_t lead_us = 0;                // attente des mesures de la paire, avant sa soumission
  // DistilDRBA mesure aussi une paire « source seule » (facteur 1) : sans cela, la paire suivante n'aurait
  // pas les mesures du côté d'avant.
  // Rapide à pas constant : chaque paire est mesurée, à N = 1 aussi (les coupes changent N).
  if ((facts.factor > 1 || drba() || steady) && facts.same_epoch) {
    const auto asked = std::chrono::steady_clock::now();
    auto scores = drba() ? drba_scores(frame) : pair_scores();
    // Les dernières attentes seulement : le journal de fin les résume, la mémoire reste bornée.
    const auto waited =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - asked)
            .count();
    keep_scores_wait(waited);
    lead_us = waited;
    if (!scores) return std::unexpected(std::move(scores.error()));
    facts.scores = *scores;
    judge_cuts(frame, facts);
    count_measured(facts);
    mark_pair(frame, facts);
  }
  // Smooth : une tenue de deux à six sources fondue en un mouvement (upscale_smooth.cpp).
  smooth_advance(frame, facts, on_grid);
  if (auto ok = smooth_decide(frame, spec, facts, on_grid); !ok)
    return std::unexpected(std::move(ok.error()));
  // L'image seule si l'interpolation ne tourne pas ; la grille repart alors d'elle.
  outputs_ = 1;
  intermediates_ = 0;
  middles_done_ = 0;
  middle_rank_ = 0;
  instants_[0] = Instant{.t = 1.0F, .pts_us = frame.pts_us};
  const bool runs = on_grid && (smooth_first_ || rife_runs(facts));
  // P6.15 : une paire sans passe (dessin tenu, coupe, hors grille) sépare la suivante de la dernière passe
  // du milieu ; ses blocs grossiers, vieux de plusieurs sources, ne se reprennent pas (A09-13).
  if (!runs) warm_from_ = nullptr;
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
           .period_us = gap >= kPairGapMinUs && gap <= kPairGapMaxUs
                            ? static_cast<std::int64_t>(static_cast<double>(gap) / play_rate)
                            : 0});
    } else {
      setup_.live->auto_alive(at_us);
    }
  }
  // Coupe ou dessin tenu : les instants de la paire manquent à dessein, pas faute de temps.
  uninterpolated_ = on_grid && !runs && facts.factor > 1 && facts.scores &&
                    (pair_cut(facts) || kernels::is_held(facts.scores->worst_block));
  resync_grids(on_grid && !runs, free_mode, steady);
  if (runs && step) {
    outputs_ = 0;
    std::size_t skipped = 0;
    for (const core::OutputInstant& at : step->instants()) {
      // Smooth, première paire : la source tenue elle-même devient un instant du mouvement vers b.
      const bool source = at.is_source() && !smooth_first_;
      if (!source && skipped < late) {  // déjà en retard : sauté
        ++skipped;
        continue;
      }
      double t = static_cast<double>(at.t.num) / static_cast<double>(at.t.den);
      if (smooth_first_ || (smooth_second_ && !source)) t = smooth_hold_u(t, smooth_rank_, smooth_hold_);
      instants_[outputs_++] =
          Instant{.t = source ? 1.0F : static_cast<float>(t),
                  .pts_us = at.is_source() ? frame.pts_us : at.pts_us.value_or(frame.pts_us)};
      if (!source) ++intermediates_;
    }
    ++interpolated_;
    if (setup_.live) setup_.live->count_interpolated();
  }
  plan_gaps(frame, runs && step ? &*step : nullptr, late, free_mode && !steady && facts.same_epoch);
  // Garder les poses : la paire précédente compte si elle précède vraiment celle-ci ; une paire
  // tenue (dessin identique) n'a pas bougé. Le mouvement de cette paire va dans l'autre tampon.
  before_ = facts.same_epoch ? now_ : PairMotion::Unknown;
  const bool held = held_pair(facts);  // la zone qui bouge des autres paires est comptée (P6.6, P6.18)
  now_ = held ? PairMotion::Still : PairMotion::Unknown;
  // Dessin tenu : l'image agrandie de la précédente sert à celle-ci, sans passe de SPAN. Pas plus
  // de kMaxSpanReuse de suite, comptées depuis la dernière vraie passe : un fondu lent, tenu paire
  // par paire, ne fige pas l'image.
  reuse_span_ = reuses_span(held);
  spans_reused_in_row_ = reuse_span_ ? spans_reused_in_row_ + 1 : 0;
  pick_span_buffer();
  // Après un dessin tenu, b paraît à 1/2 sans passe ni warp (poses_copy) : une seule image écrite.
  copy_pair_ = runs && setup_.skip_held && by_motion() && setup_.live && setup_.live->keep_poses() &&
               before_ == PairMotion::Still && intermediates_ > 0;
  copy_written_ = false;
  b_shown_ = false;
  late_fate_now_ = (setup_.late_fate || steady) && free_mode && !copy_pair_ && setup_.clock != nullptr;
  // Dessin tenu qui reprend l'agrandissement de la précédente, dont l'image est à l'écran : la source
  // est bit à bit celle qui est affichée, ni écrite ni présentée.
  held_same_ = setup_.skip_held && runs == false && outputs_ == 1 && reuse_span_ && shown_source_;
  const int written = copy_pair_ ? 1 : held_same_ ? 0 : static_cast<int>(outputs_);
  // Une seule entrée par paire (P2.10) : une seconde, au même identifiant, masquait le drapeau heavy.
  const bool free_counted = free_mode && facts.same_epoch;
  if (auto_measured || free_counted)
    expect_pair(frame.id.value, written, lead_us, free_counted, steady, runs);
  if (free_counted) count_free(facts, written, steady ? !on_grid : free_grid_.starved());
  if (intermediates_ > 0) mid_ ^= 1U;
  if (intermediates_ > 0 && !copy_pair_ && (setup_.rife != nullptr || by_motion())) {
    if (auto ok = prepare_rife_input(); !ok) return std::unexpected(std::move(ok.error()));
  }
  if (drba()) {
    // Mesurée et sans coupe, la paire laisse a comme image d'avant à la paire suivante.
    drba_.continuous = facts.scores.has_value() && !pair_cut(facts);
    if (auto ok = drba_next(frame, spec, facts.same_epoch); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  spanned_now_ = false;  // SPAN de cette image : à sa première utilisation (span_current)
  // Avant RIFE et SPAN de cette image : la suivante, si elle attend, ingérée et mesurée.
  ahead_.ticket = 0;
  if (looks_ahead()) {
    if (auto ok = look_ahead(frame, spec); !ok) return std::unexpected(std::move(ok.error()));
  }
  return run_early().transform([this] { return outputs_; });  // passes RIFE sans slot (upscale_early.cpp)
}

// Ce que la grille libre sait de l'instant : l'horloge maître prédite à la fin du calcul (retard vidéo mesuré
// compris), l'écart minimal de l'écran et le temps de service mesuré, en temps de contenu.
core::FreeInputs UpscaleStage::free_inputs(std::uint32_t epoch) const {
  core::FreeInputs inputs;
  const std::int64_t at = host_now_us();
  const auto [step_real, gap_real] = setup_.live->free_step(at, heavy_pair_);
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
  if (running) {
    // Écran à fréquence fixe : la grille des vsyncs en temps de contenu, calée sur la dernière phase mesurée.
    if (const auto grid = setup_.live->vsync_grid(at); grid.period_us > 0) {
      inputs.vsync_period_us = media(grid.period_us);
      inputs.vsync_origin_us = now.media_at(grid.anchor_host_us);
    }
  }
  return inputs;
}

bool UpscaleStage::passed(const DecodedFrame& frame, const DecodedFrame& next) {
  if (setup_.clock == nullptr || frame.pts_us == kNoPts || next.pts_us == kNoPts) return false;
  const ClockSnapshot now = setup_.clock->snapshot();
  if (!now.valid || now.epoch != frame.epoch || now.rate <= 0.0) return false;
  // L'image d'après sera passée quand celle-ci serait prête : celle-ci ne paraîtrait jamais.
  if (next.pts_us > ready_media_us(now, host_now_us(), 0, latency_us())) return false;
  // La suite repart de l'image d'après, comme au début d'une époque : ni paire ni grille de cadence ne
  // passent par-dessus l'image sautée. DistilDRBA la garde comme image d'avant (keep_skipped).
  previous_.reset();
  cadence_.reset();
  free_grid_.reset();
  steady_grid_.reset();
  drba_.continuous = false;
  keep_skipped(frame);
  ++skipped_;
  if (setup_.live) setup_.live->count_skipped();
  return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::expected<RingStage::Output, Refusal> UpscaleStage::produce(const DecodedFrame& frame,
                                                                std::uint32_t output, std::uint32_t slot,
                                                                std::uint64_t wait_value,
                                                                std::uint64_t signal_value) {
  AJP_TRACE("upscale.produce");
  const gpu::Tensor& now = rgb_[current_]->tensor();
  const Instant& instant = instants_[std::min<std::size_t>(output, instants_.size() - 1)];
  if (late_fate_now_ && instant_late(frame, output)) ++late_computed_;  // entre fate et ici : ne doit pas
  const gpu::RingWrite sync{.wait_value = wait_value, .signal_value = signal_value};
  const bool last = output + 1 >= outputs_;
  const bool b_early = copy_pair_ && instant.t < 1.0F;  // b écrite à un instant intermédiaire
  if (instant.t >= 1.0F || b_early) {                   // l'image elle-même
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
    shown_source_ = true;
    previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
    return source_output(b_early ? instant.pts_us : frame.pts_us, frame.id);
  }
  // Image intermédiaire, entre la précédente et celle-ci (rife_runs demande la précédente).
  if (!previous_) return refuse(RefusalCode::SequenceOrder, "RIFE : pas d'image précédente");
  middle_rank_ = middles_done_++;
  Output made{.pts_us = instant.pts_us,
              .left = previous_->id,
              .right = smooth_first_ ? smooth_b_id_ : frame.id,
              .debanded = debanded_};
  drba_held_ = false;
  shown_source_ = false;
  use_lane(gpu::Lane::Interpolate);
  next_middle();  // l'autre tampon : SPAN ou l'écriture du ring lit peut-être encore le précédent
  const gpu::RingTarget ring{.slot = &setup_.slots[slot], .sync = sync};
  if (warps_spanned()) {
    const Target out = ring_direct_ ? Target{.ring = &ring} : Target{.tensor = &middle_out()};
    if (smooth_first_) {
      if (auto ok = span_smooth_b(); !ok) return std::unexpected(std::move(ok.error()));
    }
    const gpu::Tensor& b_spanned = smooth_first_ ? spanned_[span_ahead_buf_]->tensor() : spanned(current_);
    if (auto ok = drba_motion() ? interpolate_drba_motion(output, true, out)
                                : interpolate_by_motion(output, spanned(current_ ^ 1U), b_spanned, out);
        !ok) {
      return std::unexpected(std::move(ok.error()));
    }
    use_lane(gpu::Lane::Upscale);
    if (out.tensor != nullptr) {
      if (auto ok = write_slot(*out.tensor, slot, sync); !ok) return std::unexpected(std::move(ok.error()));
    }
    if (last) previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
    made.repeated = drba_held_;
    made.graded = pair_graded();
    return held_b(made, frame, output);
  }
  if (by_motion() && ring_direct_) {  // sans SPAN : le warp écrit la sortie elle-même
    if (auto ok = interpolate_by_motion(output, rgb_[current_ ^ 1U]->tensor(),
                                        smooth_first_ ? smooth_b_rgb() : now, Target{.ring = &ring});
        !ok)
      return std::unexpected(std::move(ok.error()));
    use_lane(gpu::Lane::Upscale);
    output_has_source_ = false;
    if (last) previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
    made.repeated = drba_held_;
    return held_b(made, frame, output);
  }
  if (by_motion() || drba()) {
    auto ok = drba_motion() ? interpolate_drba_motion(output, false, Target{.tensor = &middle()})
              : drba()
                  ? interpolate_drba(output)
                  : interpolate_by_motion(output, rgb_[current_ ^ 1U]->tensor(),
                                          smooth_first_ ? smooth_b_rgb() : now, Target{.tensor = &middle()});
    if (!ok) return std::unexpected(std::move(ok.error()));
  } else if (auto ok = interpolate_image(instant.t); !ok) {
    return std::unexpected(std::move(ok.error()));
  }
  use_lane(gpu::Lane::Upscale);
  if (auto ok = finish(middle(), slot, sync); !ok) return std::unexpected(std::move(ok.error()));
  output_has_source_ = false;
  // ×1,5 : l'image n'est pas sur la grille de cette paire, elle sert seulement de suivante.
  if (last) previous_ = Previous{.pts_us = frame.pts_us, .id = frame.id, .epoch = frame.epoch};
  made.repeated = drba_held_;
  return held_b(made, frame, output);
}

}  // namespace ajp::app::play
