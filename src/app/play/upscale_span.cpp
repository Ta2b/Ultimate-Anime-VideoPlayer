// L'étage gpu (upscale_stage.h) : SPAN, son tampon repris sur un dessin tenu et l'écriture de
// l'emplacement de l'anneau. Sorti d'upscale_stage.cpp (C09-09).
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

#include "app/play/links.h"
#include "app/play/upscale_stage.h"
#include "core/log.h"
#include "core/trace.h"
#include "kernels/ref/deband.h"
#include "kernels/spec/resample.h"

namespace kirari::app::play {
namespace {

// Le plan des tuiles de l'image (SpanReuse::tile_plan), donné au moteur juste avant son SPAN, puis oublié.
// Faux : l'image se calcule entière (pas de plan, ou le moteur l'a refusé). previous nul : la sortie tient
// déjà celle d'avant (output_, sur place).
bool send_tiles(SpanReuse& reuse, gpu::Engine& engine, const gpu::Tensor* previous) {
  const TilePlan plan = std::exchange(reuse.tile_plan, TilePlan{});
  if (!plan.tiled) return false;
  if (engine.next_span_tiles(plan.core, plan.shift_x, plan.shift_y, previous)) return true;
  reuse.tiles.refused();
  return false;
}

// Le plan de l'image ne peut pas servir (sa sortie d'avant n'est plus là) : l'image est entière, comptée
// comme un refus.
void drop_plan(SpanReuse& reuse) {
  if (std::exchange(reuse.tile_plan, TilePlan{}).tiled) reuse.tiles.refused();
}

// --span-tiles-check : l'image tout juste calculée par tuiles (tiled) se calcule aussi entière, et l'écart
// des deux va au bilan des tuiles. Un refus de place ou de mesure éteint le contrôle, pas la lecture.
std::expected<void, Refusal> check_tiles(SpanReuse& reuse, gpu::Backend& backend, gpu::Engine& engine,
                                         std::span<const gpu::Tensor> inputs, const gpu::Tensor& tiled) {
  if (!reuse.check || !reuse.tiles.check_due()) return {};
  const auto stop = [&reuse](const Refusal& why) {
    log_channel("gpu")->info("SPAN par tuiles : contrôle arrêté ({})", why.detail);
    reuse.check = false;
  };
  if (!reuse.check_output) {
    auto made = backend.alloc_tensor(tiled.type, tiled.shape);
    if (!made) {
      stop(made.error());
      return {};
    }
    reuse.check_output = std::move(*made);
  }
  std::array outputs{reuse.check_output->tensor()};
  if (auto ok = backend.enqueue(engine, inputs, outputs); !ok) return ok;
  auto scores = backend.kernels()->scores_now(tiled, reuse.check_output->tensor());
  if (!scores) {
    stop(scores.error());
    return {};
  }
  reuse.tiles.checked(scores->worst_block, scores->scene_score * 255.0);
  return {};
}

// Ce que l'étage sait de la sortie d'avant et de l'entrée de SPAN, pour les tuiles qui bougent.
struct TileContext {
  bool before_warp = false;       // SPAN avant le warp : la sortie d'avant est dans l'autre tampon
  bool previous_spanned = false;  // ce tampon tient vraiment le SPAN de la source d'avant
  bool previous_graded = false;   // et le grade avant le warp y est passé, sur place
  bool ungraded_kept = false;     // sa sortie telle que SPAN l'a rendue est copiée (SpanReuse::ungraded)
  bool output_has_source = false;
  std::uint32_t intermediates = 0;
  bool debanded = false;  // l'entrée de SPAN de cette image
  gpu::Reduction reduce;  // et son filtre de réduction
};

// La recette de l'entrée de SPAN (TileFacts::input_key) : FNV-1a de ce qui la fait.
std::uint64_t input_recipe(bool debanded, const gpu::Reduction& reduce) {
  std::uint64_t key = 1469598103934665603ULL;
  const auto mix = [&key](std::uint64_t value) { key = (key ^ value) * 1099511628211ULL; };
  mix(debanded ? 1U : 0U);
  mix(static_cast<std::uint64_t>(reduce.kind));
  mix(std::bit_cast<std::uint32_t>(reduce.b));
  mix(std::bit_cast<std::uint32_t>(reduce.c));
  return key;
}

// Tuiles qui bougent (upscale_tiles.h) : en lecture seulement (l'export rejoue la chaîne au bit près, la
// carte arrive quand elle peut), ni TemporalFix (la carte ne voit pas sa sortie), ni Smooth (SPAN fait en
// avance), ni plaque de décor (layer_stage.h : elle réécrit la sortie, ses pixels ne se reprennent pas d'une
// image à l'autre).
bool tiles_fit(const UpscaleSetup& setup) {
  const bool playing = setup.clock != nullptr && !setup.yuv_slots;
  return playing && setup.temporal_fix == nullptr && !setup.smooth_drawings && setup.layers.plate == nullptr;
}

// Avant le warp, la sortie d'avant est dans l'autre tampon, si la source d'avant a vraiment eu son SPAN et
// que le grade ne l'a pas réécrite, ou qu'il l'a été depuis sa copie d'avant le grade (keep_ungraded) ; sinon
// dans output_, si aucune image créée ne l'a écrasée.
TileStale output_stale(const TileContext& at) {
  if (!at.before_warp)
    return at.output_has_source && at.intermediates == 0 ? TileStale::None : TileStale::Overwritten;
  if (!at.previous_spanned) return TileStale::Overwritten;
  return at.previous_graded && !at.ungraded_kept ? TileStale::Graded : TileStale::None;
}

void tile_facts(ReuseFacts& rf, const UpscaleSetup& setup, const TileContext& at, const PairFacts& facts) {
  rf.tiles_stale = tiles_fit(setup) ? output_stale(at) : TileStale::Chain;
  rf.can_shift = at.before_warp;
  rf.scene = facts.scene.has_value() ? &*facts.scene : nullptr;
  rf.spread_px = at.debanded ? static_cast<int>(std::ceil(kernels::ref::kDebandRadius)) : 0;
  rf.input_key = input_recipe(at.debanded, at.reduce);
}

// La sortie d'avant telle que SPAN l'a rendue, d'où l'image se calcule par tuiles : le tampon buf de la
// précédente, ou sa copie d'avant le grade quand le grade avant le warp l'a réécrit ; nul sans elle.
const gpu::Tensor* ungraded_previous(const SpanReuse& reuse, const gpu::Tensor& previous, std::uint32_t buf,
                                     bool graded) {
  if (!graded) return &previous;
  return reuse.ungraded && reuse.ungraded_of == buf ? &reuse.ungraded->tensor() : nullptr;
}

// Avant le grade sur place du tampon buf, sa sortie de SPAN telle quelle est copiée dans reuse.ungraded : la
// suivante se calcule par tuiles depuis elle (copie exacte, même taille). Un refus de place ou de copie
// éteint la copie, donc les tuiles sous le grade, pas la lecture.
std::expected<void, Refusal> keep_ungraded(gpu::Backend& backend, const gpu::Tensor& spanned,
                                           std::uint32_t buf, SpanReuse& reuse) {
  KIRARI_TRACE("span.ungraded");
  const auto stop = [&reuse](const Refusal& why) {
    log_channel("gpu")->info(
        "SPAN par tuiles : copie d'avant le grade refusée ({}) ; images entières sous le grade", why.detail);
    reuse.ungraded_ok = false;
  };
  if (!reuse.ungraded) {
    auto tensor = backend.alloc_tensor(spanned.type, spanned.shape);
    if (!tensor) {
      stop(tensor.error());
      return {};
    }
    reuse.ungraded = std::move(*tensor);
  }
  auto copied = backend.kernels()->resample(spanned, kernels::Filter::Area, reuse.ungraded->tensor());
  if (copied) {
    reuse.ungraded_of = buf;
    return {};
  }
  if (copied.error().code != RefusalCode::FormatUnsupported) return copied;
  stop(copied.error());
  return {};
}

// Le moteur oublie la demande de tuiles d'un appel qui n'a pas abouti : elle ne vaut que pour lui.
void drop_tiles(gpu::Engine& engine) {
  static_cast<void>(engine.next_span_tiles({}, 0, 0, nullptr));
}

}  // namespace

namespace {

// L'écart entre l'ancre et l'image en cours ; nullopt si le backend ne sait pas le mesurer (la reprise
// retombe alors à deux de suite). waited_us : l'attente hôte de la relecture.
std::optional<AnchorGap> measure_anchor(gpu::KernelOps& ops, const gpu::Tensor& anchor,
                                        const gpu::Tensor& now, SpanReuse& reuse, std::int64_t& waited_us) {
  KIRARI_TRACE("span.anchor");
  const auto asked = std::chrono::steady_clock::now();
  auto scores = ops.scores_now(anchor, now);
  waited_us =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - asked).count();
  if (scores) return AnchorGap{.worst_block = scores->worst_block};
  log_channel("gpu")->info("SPAN repris : pas de mesure contre l'ancre ({}) ; deux de suite au plus",
                           scores.error().detail);
  reuse.anchor_measure = false;
  return std::nullopt;
}

// Début d'une série : l'ancre (la précédente, telle que la paire la mesure) est copiée avant que son tampon
// ne reçoive une autre image ; la copie suit sur la voie en cours, sans attente. Un refus de place ou de
// copie éteint la mesure contre l'ancre (deux de suite au plus), pas la lecture.
std::expected<void, Refusal> keep_anchor(gpu::Backend& backend, const gpu::Tensor& previous,
                                         SpanReuse& reuse) {
  const auto stop = [&reuse](const Refusal& why) {
    log_channel("gpu")->info("SPAN repris : ancre refusée ({}) ; deux de suite au plus", why.detail);
    reuse.anchor_measure = false;
  };
  if (!reuse.anchor) {
    auto tensor = backend.alloc_tensor(previous.type, previous.shape);
    if (!tensor) {
      stop(tensor.error());
      return {};
    }
    reuse.anchor = std::move(*tensor);
  }
  auto copied = backend.kernels()->resample(previous, kernels::Filter::Area, reuse.anchor->tensor());
  if (copied) {
    reuse.anchor_kept();
    return {};
  }
  if (copied.error().code != RefusalCode::FormatUnsupported) return copied;
  stop(copied.error());
  return {};
}

}  // namespace

std::expected<bool, Refusal> UpscaleStage::reuses_span(const DecodedFrame& frame, const PairFacts& facts,
                                                       bool held, const gpu::Reduction& reduce) {
  if (setup_.engine == nullptr) return false;
  // Smooth, warp après SPAN : chaque copie de la tenue reprend le SPAN de a, quelle que soit sa longueur ;
  // aucune n'est montrée telle quelle, et l'autre tampon tient celui de b (span_smooth_b).
  if (smooth_first_ && warps_spanned()) {
    reuse_.force();
    return true;
  }
  const bool pair_measured = facts.scores.has_value();
  ReuseFacts rf{.same_epoch = facts.same_epoch,
                .corrupt = frame.corrupt,
                .has_output = warps_spanned() || output_has_source_ || (setup_.skip_held && shown_source_),
                .pair_held = !pair_measured || held};
  tile_facts(rf, setup_,
             TileContext{.before_warp = warps_spanned(),
                         .previous_spanned = reuse_.spanned_is(previous_),
                         .previous_graded = span_graded_[span_of_[current_ ^ 1U]],
                         .ungraded_kept = reuse_.ungraded_of == span_of_[current_ ^ 1U],
                         .output_has_source = output_has_source_,
                         .intermediates = intermediates_,
                         .debanded = debanded_,
                         .reduce = reduce},
             facts);
  const bool candidate = rf.same_epoch && !rf.corrupt && rf.has_output && rf.pair_held;
  if (pair_measured && rf.same_epoch) reuse_.note_pair(facts.scores->worst_block);
  const int in_row = reuse_.in_row();
  // Les images de la mesure de la paire : la copie réduite de RIFE quand elle existe, où le seuil des dessins
  // tenus est réglé (A09-18) ; à la taille de travail, le bruit de compression couperait les séries.
  const gpu::Tensor& previous = rife_source(current_ ^ 1U);
  // L'ancre est la précédente au début d'une série, sa copie ensuite, si elle est de cette série (une série
  // imposée par Smooth n'en copie pas). Sans mesure de la paire (SPAN seul), elle se mesure dès la première.
  const bool anchored = in_row == 0 || reuse_.anchor_ready();
  if (candidate && reuse_.anchor_measure && reuse_.wants_anchor(pair_measured) && anchored) {
    std::int64_t waited = 0;
    rf.anchor = measure_anchor(*setup_.backend->kernels(), in_row == 0 ? previous : reuse_.anchor->tensor(),
                               rife_source(current_), reuse_, waited);
    keep_scores_wait(waited);
  }
  // Sans mesure du tout (SPAN seul, mesure refusée) : rien ne dit que l'image est tenue.
  if (!pair_measured && !rf.anchor) rf.pair_held = false;
  const bool reused = reuse_.decide(rf) == ReuseKind::Whole;  // Tiles : SPAN, plan dans reuse_.tile_plan
  if (reused && in_row == 0 && reuse_.anchor_measure) {
    if (auto ok = keep_anchor(*setup_.backend, previous, reuse_); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  return reused;
}

void UpscaleStage::pick_span_buffer() {
  // Choisi au premier SPAN, le tampon restait celui d'il y a deux images, qui après un dessin tenu est
  // celui de a : le premier instant de la paire mélangeait a à elle-même, warpée par le mouvement de la
  // paire (contours doublés, masque de RIFE en trame).
  span_of_[current_] = reuse_span_ ? span_of_[current_ ^ 1U] : span_of_[current_ ^ 1U] ^ 1U;
}

std::expected<void, Refusal> UpscaleStage::span_current() {
  KIRARI_TRACE("span");
  if (spanned_now_) return {};
  spanned_now_ = true;
  // Le tampon est choisi à l'arrivée de l'image : celui de la précédente pour un dessin tenu, sinon
  // l'autre, car celui de la précédente reste le bord a des warps de la paire.
  if (reuse_span_) {
    reuse_.spanned = frame_id_;  // le tampon de l'ancre tient la sortie de cette image
    return {};
  }
  // Smooth : le SPAN de cette image a été fait en avance, dans ce même tampon (span_smooth_b).
  if (std::exchange(span_ahead_, std::nullopt) == frame_id_ && span_ahead_buf_ == span_of_[current_])
    return {};
  // Sur la voie de SPAN, au milieu du travail de la paire : la voie de l'appelant reprend après.
  const gpu::Lane back = lane_;
  use_lane(gpu::Lane::Upscale);
  if (setup_.engine == nullptr) {  // copies gradées (graded_copies_) : la source copiée, puis gradée
    auto ok = setup_.backend->kernels()->resample(rgb_[current_]->tensor(), kernels::Filter::Area,
                                                  spanned(current_));
    if (ok) ok = grade_spanned(span_of_[current_]);
    use_lane(back);
    return ok;
  }
  const std::array inputs{rgb_[current_]->tensor()};
  std::array results{spanned(current_)};
  const std::uint32_t before = span_of_[current_ ^ 1U];
  if (reuse_.ungraded_of == span_of_[current_]) reuse_.ungraded_of.reset();  // ce tampon va changer
  const gpu::Tensor* previous =
      ungraded_previous(reuse_, spanned(current_ ^ 1U), before, span_graded_[before]);
  if (previous == nullptr) drop_plan(reuse_);
  const bool tiled = previous != nullptr && send_tiles(reuse_, *setup_.engine, previous);
  auto ok = blame_stage(setup_.backend->enqueue(*setup_.engine, inputs, results), kSpanRefusalKey);
  if (!ok && tiled) drop_tiles(*setup_.engine);
  if (ok && tiled) ok = check_tiles(reuse_, *setup_.backend, *setup_.engine, inputs, results[0]);
  reuse_.spanned = ok ? std::optional(frame_id_) : std::nullopt;
  // La plaque de décor (layer_stage.h), avant le grade : le décor de b repris de l'image clé.
  if (ok)
    ok = layers_.after_span(*setup_.backend, setup_.layers, inputs[0], results[0],
                            setup_.warp == kernels::RifeWarp::CatmullRom);
  if (ok) ok = grade_spanned(span_of_[current_]);
  use_lane(back);
  return ok;
}

std::expected<void, Refusal> UpscaleStage::grade_spanned(std::uint32_t buf) {
  span_graded_[buf] = false;
  if (reuse_.ungraded_of == buf) reuse_.ungraded_of.reset();
  if (!setup_.grader || setup_.yuv_slots) return {};
  KIRARI_TRACE("grade avant le warp");
  // La copie d'avant le grade, si les tuiles peuvent servir et que le grade a écrit l'image d'avant : un
  // grader qui ne grade pas (réglage, écran) ne coûte pas de copie.
  if (setup_.engine != nullptr && setup_.scene_map && reuse_.tiles.active() && reuse_.ungraded_ok &&
      span_graded_[buf ^ 1U] && tiles_fit(setup_)) {
    if (auto ok = keep_ungraded(*setup_.backend, spanned_[buf]->tensor(), buf, reuse_); !ok) return ok;
  }
  const GradedSource source{.epoch = head_.epoch,
                            .pts_us = head_.pts_us,
                            .effects = setup_.live
                                           ? setup_.live->effect_book().for_source(head_.epoch, head_.pts_us)
                                           : std::nullopt};
  auto graded = setup_.grader->grade(*setup_.backend->kernels(), spanned_[buf]->tensor(), debanded_, source);
  if (!graded) {
    log_channel("gpu")->warn("grade avant le warp : {} {} ; l'écran reprend le grade",
                             name(graded.error().code), graded.error().detail);
    setup_.grader.reset();
    return {};
  }
  span_graded_[buf] = *graded;
  return {};
}

bool UpscaleStage::source_graded() const {
  return warps_spanned() && span_graded_[span_of_[current_]];
}

RingStage::Output UpscaleStage::source_output(std::int64_t pts_us, SourceId id) const {
  return Output{.pts_us = pts_us,
                .left = id,
                .right = id,
                .uninterpolated = uninterpolated_,
                .debanded = debanded_,
                .graded = source_graded()};
}

bool UpscaleStage::pair_graded() const {
  return span_graded_[span_of_[current_ ^ 1U]] &&
         span_graded_[smooth_first_ ? span_ahead_buf_ : span_of_[current_]];
}

std::expected<void, Refusal> UpscaleStage::finish(const gpu::Tensor& rgb, std::uint32_t slot,
                                                  const gpu::RingWrite& sync) {
  if (setup_.engine == nullptr) return write_slot(rgb, slot, sync);
  const std::array inputs{rgb};
  std::array outputs{output_->tensor()};
  // Tuiles qui bougent : la source de l'image seulement (output_ tient celle d'avant), par enqueue (la queue
  // n'écrit que les tuiles calculées : l'emplacement reçoit ensuite output_ entière).
  const gpu::Tensor& now = rgb_[current_]->tensor();
  if (rgb.handle == now.handle && rgb.offset == now.offset && send_tiles(reuse_, *setup_.engine, nullptr)) {
    auto ok = setup_.backend->enqueue(*setup_.engine, inputs, outputs);
    if (!ok) drop_tiles(*setup_.engine);
    if (ok) ok = check_tiles(reuse_, *setup_.backend, *setup_.engine, inputs, outputs[0]);
    if (!ok) return blame_stage(std::move(ok), kSpanRefusalKey);
    return write_slot(output_->tensor(), slot, sync);
  }
  // Sans export YUV : SPAN et l'écriture de l'emplacement en un appel (étape 3.4b), que le backend sait
  // fondre (le SPAN FP8 écrit l'anneau depuis son épilogue) ; la sortie est écrite aussi (reuse_span_).
  if (!setup_.yuv_slots) {
    return blame_stage(
        setup_.backend->enqueue_ring(*setup_.engine, inputs, outputs,
                                     gpu::RingTarget{.slot = &setup_.slots[slot], .sync = sync}),
        kSpanRefusalKey);
  }
  if (auto ok = setup_.backend->enqueue(*setup_.engine, inputs, outputs); !ok)
    return blame_stage(std::move(ok), kSpanRefusalKey);
  return write_slot(output_->tensor(), slot, sync);
}

std::expected<void, Refusal> UpscaleStage::write_slot(const gpu::Tensor& image, std::uint32_t slot,
                                                      const gpu::RingWrite& sync) {
  gpu::KernelOps& ops = *setup_.backend->kernels();
  if (setup_.yuv_slots) return ops.write_yuv(image, setup_.yuv_slots->target(slot, sync.signal_value));
  return ops.write_ring(image, setup_.slots[slot], sync);
}

}  // namespace kirari::app::play
