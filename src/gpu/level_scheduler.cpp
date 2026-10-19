#include "gpu/level_scheduler.h"

#include <algorithm>
#include <format>
#include <string>
#include <utility>

namespace ajp::gpu {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// La timeline tient au moins les paires en attente et leur anticipation.
static_assert(LevelScheduler::kMaxPending + core::drawings::kSmoothPreload + 2 <= core::drawings::kWindow);

}  // namespace

std::expected<LevelScheduler, Refusal> LevelScheduler::create(const core::InterpLevel& level,
                                                              std::uint32_t index) {
  if (index >= kMaxLevels) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("niveau {} hors de [0, {}[", index, kMaxLevels));
  }
  if (level.policy == core::Policy::Quality && level.pacing != core::Pacing::Source) {
    return refuse(RefusalCode::PacingForbidden,
                  std::format("niveau {} : Quality impose le rythme source", index));
  }
  auto cadence = core::CadenceAccumulator::create(level.factor);
  if (!cadence) return std::unexpected(std::move(cadence.error()));
  return LevelScheduler(level, index, *cadence);
}

std::expected<std::span<const LevelOutput>, Refusal> LevelScheduler::push(const LevelInput& input) {
  output_count_ = 0;
  const core::CadenceStep step =
      cadence_.push({.id = input.id, .pts_us = input.pts_us, .epoch = input.epoch});
  if (step.event == core::GridEvent::Stale) return std::span<const LevelOutput>{};
  epoch_ = std::max(epoch_, input.epoch);

  Pending pending{.origin = step.event != core::GridEvent::Pair,
                  .left = step.left,
                  .right = input.id,
                  .epoch = input.epoch};
  for (const core::OutputInstant& instant : step.instants()) pending.instants[pending.count++] = instant;

  // Start et NewEpoch ouvrent une séquence : les paires de l'ancienne époque sont jetées.
  // Les autres origines (horodatage invalide ou qui recule) gardent la timeline : seule la
  // grille repart, le classement des dessins continue.
  const bool starts_sequence =
      step.event == core::GridEvent::Start || step.event == core::GridEvent::NewEpoch;
  if (starts_sequence) {
    pending_count_ = 0;
    last_.reset();
  }
  auto ok = classify(input, starts_sequence);
  if (ok) ok = enqueue(pending);
  if (ok) ok = drain();
  if (!ok) {
    reset();
    return std::unexpected(std::move(ok.error()));
  }
  update_live();
  return std::span<const LevelOutput>{outputs_.data(), output_count_};
}

std::span<const LevelOutput> LevelScheduler::finish() {
  output_count_ = 0;
  timeline_.finish();
  if (!drain()) reset();
  update_live();
  return {outputs_.data(), output_count_};
}

std::expected<void, Refusal> LevelScheduler::flush(std::uint32_t epoch) {
  if (auto ok = cadence_.flush(epoch); !ok) return ok;
  epoch_ = epoch;
  pending_count_ = 0;
  last_.reset();
  live_count_ = 0;
  return {};
}

std::expected<void, Refusal> LevelScheduler::classify(const LevelInput& input, bool starts_sequence) {
  Record record{
      .id = input.id, .position = next_position_, .first = input.id, .first_position = next_position_};
  if (starts_sequence) {
    timeline_.start(input.id);
  } else {
    auto drawing = timeline_.push(input.id, input.measure);
    if (!drawing) return std::unexpected(std::move(drawing.error()));
    record.measure = input.measure;
    if (drawing->scene_cut) ++counters_.scene_cuts;
    if (drawing->first != input.id) {
      const Record* previous = at(next_position_ - 1);
      record.first = drawing->first;
      record.first_position = previous != nullptr ? previous->first_position : next_position_;
    }
  }
  records_[next_position_ % records_.size()] = record;
  ++next_position_;
  return {};
}

std::expected<void, Refusal> LevelScheduler::enqueue(const Pending& pending) {
  if (pending_count_ == kMaxPending) {
    return refuse(
        RefusalCode::SequenceOrder,
        std::format("niveau {} : plus de {} paires en attente d'anticipation", index_, kMaxPending));
  }
  pending_[(pending_head_ + pending_count_) % kMaxPending] = pending;
  ++pending_count_;
  return {};
}

std::expected<void, Refusal> LevelScheduler::drain() {
  while (pending_count_ > 0) {
    const Pending& front = pending_[pending_head_];
    if (!front.origin && !timeline_.ready(front.left, level_.pacing)) break;
    const Pending current = front;
    pending_head_ = (pending_head_ + 1) % kMaxPending;
    --pending_count_;
    if (auto ok = decide(current); !ok) return ok;
  }
  return {};
}

std::expected<void, Refusal> LevelScheduler::decide(const Pending& pending) {
  if (pending.origin) {
    const Record* right = find(pending.right);
    if (right == nullptr)
      return refuse(RefusalCode::SourceUnknown, std::format("origine {}", pending.right.value));
    emit({.pts_us = pending.instants[0].pts_us,
          .epoch = pending.epoch,
          .decision = {.show = core::Show::Source, .a = pending.right},
          .pair_left = pending.right,
          .pair_right = pending.right},
         *right);
    return {};
  }
  const Record* left = find(pending.left);
  const Record* right = find(pending.right);
  if (left == nullptr || right == nullptr) {
    return refuse(RefusalCode::SourceUnknown,
                  std::format("paire ({}, {})", pending.left.value, pending.right.value));
  }
  for (std::size_t i = 0; i < pending.count; ++i) {
    const core::OutputInstant& instant = pending.instants[i];
    auto decision = timeline_.decide(level_.policy, level_.pacing, pending.left, instant.t);
    if (!decision) return std::unexpected(std::move(decision.error()));
    emit({.pts_us = instant.pts_us,
          .epoch = pending.epoch,
          .decision = *decision,
          .pair_left = pending.left,
          .pair_right = pending.right,
          .t = instant.t,
          .scene_cut = !instant.is_source() && right->measure.scene_cut},
         *left);
  }
  return {};
}

void LevelScheduler::emit(LevelOutput output, const Record& left) {
  const Content content = content_of(output.decision, left);
  output.id = derived_id(index_, serial_++);
  output.measure = last_ ? synthesize(*last_, content) : core::PairMeasure{};
  last_ = content;
  ++counters_.outputs;
  if (output.decision.show == core::Show::Interpolate) ++counters_.interpolated;
  if (output.decision.show == core::Show::Repeated) ++counters_.repeated;
  outputs_[output_count_++] = output;
}

void LevelScheduler::reset() {
  // La prochaine entrée redevient l'origine. epoch_ est la plus récente que l'accumulateur
  // ait vue : flush() ne peut pas la refuser.
  const auto flushed = cadence_.flush(epoch_);
  (void)flushed;
  pending_count_ = 0;
  last_.reset();
  live_count_ = 0;
}

void LevelScheduler::update_live() {
  live_count_ = 0;
  if (next_position_ == 0) return;
  // Sans paire en attente, la dernière entrée sera la gauche de la paire suivante.
  std::uint64_t start = next_position_ - 1;
  if (pending_count_ > 0) {
    const Pending& front = pending_[pending_head_];
    if (const Record* r = find(front.origin ? front.right : front.left); r != nullptr) start = r->position;
  }
  for (std::uint64_t p = start; p < next_position_ && live_count_ < kMaxLive; ++p) {
    if (const Record* r = at(p); r != nullptr) live_[live_count_++] = r->id;
  }
  // Smooth cite la première source du dessin de la paire, peut-être bien avant elle.
  if (level_.pacing == core::Pacing::Smooth && live_count_ < kMaxLive) {
    if (const Record* r = at(start); r != nullptr && r->first_position < start)
      live_[live_count_++] = r->first;
  }
}

const LevelScheduler::Record* LevelScheduler::find(SourceId id) const {
  const std::uint64_t kept = std::min<std::uint64_t>(next_position_, records_.size());
  for (std::uint64_t back = 1; back <= kept; ++back) {
    const Record& r = records_[(next_position_ - back) % records_.size()];
    if (r.id == id) return &r;
  }
  return nullptr;
}

const LevelScheduler::Record* LevelScheduler::at(std::uint64_t position) const {
  if (position >= next_position_ || next_position_ - position > records_.size()) return nullptr;
  return &records_[position % records_.size()];
}

LevelScheduler::Content LevelScheduler::content_of(const core::InstantDecision& decision,
                                                   const Record& left) const {
  // La première source du dessin peut être sortie de la fenêtre : sa position vient de la gauche.
  const auto position = [&](SourceId id) {
    if (id == left.first) return left.first_position;
    const Record* r = find(id);
    return r != nullptr ? r->position : left.position;
  };
  if (decision.show == core::Show::Interpolate) {
    return {.lo = position(decision.a), .hi = position(decision.b), .single = false};
  }
  const std::uint64_t p = position(decision.a);
  return {.lo = p, .hi = p, .single = true};
}

core::PairMeasure LevelScheduler::synthesize(const Content& previous, const Content& current) const {
  if (previous.single && current.single && previous.lo == current.lo) {
    return {.worst_block_diff = 0.0, .scene_cut = false, .identical = true};
  }
  // Entre deux sorties, l'image change au plus comme les entrées qu'elles couvrent : le pire
  // écart des entrées de ]lo, hi], et une coupe si l'une d'elles en ouvre une.
  const std::uint64_t lo = std::min(previous.lo, current.lo);
  const std::uint64_t hi = std::max(previous.hi, current.hi);
  double worst = 0.0;
  bool cut = false;
  bool identical = previous.single && current.single;
  for (std::uint64_t p = lo + 1; p <= hi; ++p) {
    const Record* r = at(p);
    if (r == nullptr) {
      // Sortie de la fenêtre : au milieu d'une longue tenue, donc un dessin tenu.
      worst = std::max(worst, core::drawings::kHeldThreshold);
      identical = false;
      if (hi - p >= records_.size()) p = hi - records_.size();  // saute au début de la fenêtre
      continue;
    }
    worst = std::max(worst, r->measure.worst_block_diff);
    cut = cut || r->measure.scene_cut;
    identical = identical && r->measure.identical;
  }
  if (cut) return {.worst_block_diff = worst, .scene_cut = true, .identical = false};
  return {.worst_block_diff = worst, .scene_cut = false, .identical = identical && worst == 0.0};
}

}  // namespace ajp::gpu
