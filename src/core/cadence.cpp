// Repris de the-database/mpv@5afefa76 video/filter/vf_animejanai.c:1705-1756 et
// 1795-1855 (accumulateur rife_acc, rife_num, rife_den), :402-421 (flush) et
// :592-611 (réduction du facteur). Mêmes instants ; les horodatages sont
// calculés en entiers au lieu de doubles. Comportement fixé par
// tests/unit/core/cadence_test.cpp.
#include "core/cadence.h"

#include <format>
#include <numeric>
#include <string>
#include <utility>

namespace ajp::core {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// left + (right - left)·acc/num, arrondi au plus proche, sans débordement :
// l'écart passe en non signé (right > left), puis se divise avant de se multiplier.
std::int64_t between(std::int64_t left, std::int64_t right, std::int64_t acc, std::int64_t num) {
  const std::uint64_t delta = static_cast<std::uint64_t>(right) - static_cast<std::uint64_t>(left);
  const auto a = static_cast<std::uint64_t>(acc);
  const auto n = static_cast<std::uint64_t>(num);
  const std::uint64_t offset = (delta / n) * a + ((delta % n) * a + n / 2) / n;
  return static_cast<std::int64_t>(static_cast<std::uint64_t>(left) + offset);
}

}  // namespace

Ratio reduced(Ratio ratio) {
  const std::int64_t sign = ratio.den < 0 ? -1 : 1;
  const std::int64_t g = std::gcd(ratio.num, ratio.den);
  if (g == 0) return ratio;
  return {sign * ratio.num / g, sign * ratio.den / g};
}

std::expected<CadenceAccumulator, Refusal> CadenceAccumulator::create(Ratio factor) {
  const std::string where = std::format("facteur {}/{}", factor.num, factor.den);
  if (!limits::kRatioTerm.contains(factor.num) || !limits::kRatioTerm.contains(factor.den)) {
    return refuse(RefusalCode::FieldOutOfRange, where + " : terme hors de [1, 100000]");
  }
  if (factor.num <= factor.den || factor.num > std::int64_t{kMaxInstantsPerPair} * factor.den) {
    return refuse(RefusalCode::FieldOutOfRange, where + " : hors de 1 < num/den <= 10");
  }
  const Ratio r = reduced(factor);
  return CadenceAccumulator(r.num, r.den);
}

CadenceStep CadenceAccumulator::origin(const SourceStamp& source, GridEvent event) {
  // La source est l'origine de la grille : le point suivant tombe à den/num dans la paire suivante.
  acc_ = den_;
  left_ = source;
  started_ = true;
  CadenceStep step(event, {}, source.id);
  step.add({.t = {num_, num_}, .pts_us = source.pts_us});
  return step;
}

CadenceStep CadenceAccumulator::push(const SourceStamp& source) {
  if (source.epoch < epoch_) return CadenceStep(GridEvent::Stale, {}, source.id);
  if (source.epoch > epoch_) {
    // Une image d'une époque plus récente vaut un Flush, quel que soit l'ordre d'arrivée.
    epoch_ = source.epoch;
    left_.reset();
  }
  if (!left_) return origin(source, started_ ? GridEvent::NewEpoch : GridEvent::Start);
  if (!source.pts_us || !left_->pts_us) return origin(source, GridEvent::InvalidPts);
  const std::int64_t left_pts = *left_->pts_us;
  const std::int64_t right_pts = *source.pts_us;
  if (right_pts <= left_pts) return origin(source, GridEvent::PtsNotIncreasing);

  // Les points de grille dus dans la paire : acc, acc + den… tant que <= num.
  // acc == num est la source droite elle-même.
  CadenceStep step(GridEvent::Pair, left_->id, source.id);
  for (; acc_ <= num_; acc_ += den_) {
    step.add({.t = {acc_, num_}, .pts_us = between(left_pts, right_pts, acc_, num_)});
  }
  acc_ -= num_;  // se recale dans la paire suivante
  left_ = source;
  return step;
}

void CadenceAccumulator::resync() {
  acc_ = den_;
}

std::expected<void, Refusal> CadenceAccumulator::flush(std::uint32_t epoch) {
  if (epoch < epoch_) {
    return refuse(RefusalCode::SequenceOrder, std::format("époque {} antérieure à {}", epoch, epoch_));
  }
  epoch_ = epoch;
  left_.reset();
  acc_ = den_;
  return {};
}

}  // namespace ajp::core
