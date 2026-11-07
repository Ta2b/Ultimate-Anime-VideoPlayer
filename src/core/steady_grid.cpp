#include "core/steady_grid.h"

#include <format>
#include <tuple>
#include <utility>

namespace ajp::core {
namespace {

bool usable(const Ratio& r) {
  return r.den > 0 && r.num > r.den && r.num <= std::int64_t{kMaxInstantsPerPair} * r.den;
}

}  // namespace

CadenceStep SteadyGrid::alone(const SourceStamp& source, GridEvent event) const {
  CadenceStep step(event, event == GridEvent::Pair && left_ ? left_->id : SourceId{}, source.id);
  step.add({.t = {1, 1}, .pts_us = source.pts_us});
  return step;
}

CadenceStep SteadyGrid::push(const SourceStamp& source, Ratio factor) {
  if (source.epoch < epoch_) return CadenceStep(GridEvent::Stale, {}, source.id);
  if (source.epoch > epoch_) {
    epoch_ = source.epoch;
    left_.reset();
  }
  const Ratio wanted = usable(factor) ? reduced(factor) : Ratio{1, 1};
  // L'événement de la source, comme CadenceAccumulator::push.
  GridEvent event = GridEvent::Pair;
  if (!left_) {
    event = started_ ? GridEvent::NewEpoch : GridEvent::Start;
  } else if (!source.pts_us || !left_->pts_us) {
    event = GridEvent::InvalidPts;
  } else if (*source.pts_us <= *left_->pts_us) {
    event = GridEvent::PtsNotIncreasing;
  }
  started_ = true;

  if (!usable(wanted)) {
    grid_.reset();
    factor_ = Ratio{1, 1};
    CadenceStep step = alone(source, event);
    left_ = source;
    return step;
  }
  if (!grid_) {
    // Au-dessus de 1 depuis N = 1 (ou au départ) : la grille repart de la source gauche, déjà émise.
    auto made = CadenceAccumulator::create(wanted);
    grid_.emplace(*made);
    std::ignore = grid_->flush(epoch_);  // un refus laisse la grille telle quelle, comme avant
    if (left_ && event == GridEvent::Pair) (void)grid_->push(*left_);
  } else if (!(wanted == factor_)) {
    std::ignore = grid_->retarget(wanted);  // un refus garde le pas précédent, comme avant
  }
  factor_ = wanted;
  left_ = source;
  return grid_->push(source);
}

void SteadyGrid::resync() {
  if (grid_) grid_->resync();
}

void SteadyGrid::reset() {
  left_.reset();
  if (grid_) std::ignore = grid_->flush(epoch_);
}

std::expected<void, Refusal> SteadyGrid::flush(std::uint32_t epoch) {
  if (epoch < epoch_) {
    return std::unexpected(
        Refusal{RefusalCode::SequenceOrder, std::format("époque {} antérieure à {}", epoch, epoch_)});
  }
  epoch_ = epoch;
  reset();
  return {};
}

}  // namespace ajp::core
