#include "core/free_grid.h"

#include <algorithm>
#include <format>
#include <string>
#include <utility>

namespace ajp::core {

CadenceStep FreeGrid::origin(const SourceStamp& source, GridEvent event) {
  left_ = source;
  started_ = true;
  last_emitted_ = source.pts_us;
  next_due_.reset();
  CadenceStep step(event, {}, source.id);
  step.add({.t = {1, 1}, .pts_us = source.pts_us});
  return step;
}

std::expected<FreeGrid::Pair, CadenceStep> FreeGrid::pair_with(const SourceStamp& source) {
  if (source.epoch < epoch_) return std::unexpected(CadenceStep(GridEvent::Stale, {}, source.id));
  if (source.epoch > epoch_) {
    epoch_ = source.epoch;
    left_.reset();
  }
  if (!left_) return std::unexpected(origin(source, started_ ? GridEvent::NewEpoch : GridEvent::Start));
  if (!source.pts_us || !left_->pts_us) return std::unexpected(origin(source, GridEvent::InvalidPts));
  const std::int64_t g = *left_->pts_us;
  const std::int64_t d = *source.pts_us;
  if (d <= g) return std::unexpected(origin(source, GridEvent::PtsNotIncreasing));
  return Pair{.left = left_->id, .g = g, .d = d};
}

CadenceStep FreeGrid::push(const SourceStamp& source, const FreeInputs& inputs) {
  starved_ = false;
  const std::expected<Pair, CadenceStep> pair = pair_with(source);
  if (!pair) return pair.error();
  const std::int64_t g = pair->g;
  const std::int64_t d = pair->d;

  const std::int64_t gap = d - g;
  const std::int64_t vsync = std::max<std::int64_t>(inputs.vsync_period_us, 0);
  const std::int64_t min_gap = std::max<std::int64_t>({inputs.min_gap_us, vsync, 1});
  // Sans mesure, deux instants par paire ; jamais plus de kMaxInstantsPerPair dans une paire.
  const std::size_t limit = std::clamp<std::size_t>(inputs.max_instants, 1, kMaxInstantsPerPair);
  const std::int64_t wanted = inputs.step_us > 0 ? inputs.step_us : gap / 2;
  const std::int64_t step = std::max(
      {wanted, min_gap, (gap + static_cast<std::int64_t>(limit) - 1) / static_cast<std::int64_t>(limit)});

  // Jamais avant l'horloge prédite (l'image serait passée en paraissant), jamais plus près que l'écart
  // minimal du dernier instant émis, jamais avant la paire : les instants d'avant sont abandonnés sans
  // calcul. Au-delà, le pas : la phase continue d'une paire à l'autre.
  // Sur écran fixe, l'instant monte au vsync suivant (jamais entre deux) ; `base` est l'instant idéal, avant
  // ce calage : le pas moyen reste celui du GPU et l'image tenue l'est un vsync de plus quand il le faut. Un
  // écart minimal de « une période et quelques µs » (arrondis) ne doit pas sauter un vsync de plus.
  const std::int64_t spacing = vsync > 0 ? min_gap - vsync / 8 : min_gap;
  const auto on_vsync = [&](std::int64_t x) {
    if (vsync <= 0) return x;
    const std::int64_t r = ((x - inputs.vsync_origin_us) % vsync + vsync) % vsync;
    return r == 0 ? x : x + (vsync - r);
  };
  std::int64_t base = std::max(
      {next_due_.value_or(INT64_MIN), last_emitted_.value_or(INT64_MIN) + spacing, inputs.ready_us, g + 1});
  std::int64_t t = on_vsync(base);
  // Le même premier instant sans l'échéance reportée : ce que la paire aurait eu si le pas ne l'avait pas
  // repoussée.
  const std::int64_t unheld =
      on_vsync(std::max({last_emitted_.value_or(INT64_MIN) + spacing, inputs.ready_us, g + 1}));
  CadenceStep out(GridEvent::Pair, pair->left, source.id);
  std::size_t emitted = 0;
  std::size_t interpolated = 0;
  // Un instant à moins d'un quart de pas de la source droite lui est ramené : l'image est la source elle-même
  // (aucune passe d'interpolation), décalée de quelques millisecondes au plus. Sur écran fixe (P3.37), le
  // vsync à moins d'une demi-période avant la source la montre telle quelle, à l'heure du vsync.
  const std::int64_t snap = vsync > 0 ? vsync / 2 : step / 4;
  bool source_shown = false;
  while (emitted < limit && t <= d) {
    const bool on_source = d - t < snap;
    const std::int64_t at = on_source && vsync <= 0 ? d : t;
    out.add({.t = on_source ? Ratio{1, 1} : Ratio{at - g, gap}, .pts_us = at});
    last_emitted_ = at;
    ++emitted;
    if (!on_source) ++interpolated;
    base = vsync > 0 ? std::max(base + step, at + spacing) : at + step;
    t = on_vsync(base);
    if (on_source) {
      source_shown = true;
      break;
    }
  }
  // B12-04 : sur écran fixe, le vsync à moins d'une demi-période après la source la montre aussi telle
  // quelle. Sans cela, la paire suivante la recalculait par le réseau à t presque nul : une inférence de plus
  // et une image légèrement adoucie au lieu du dessin d'origine.
  if (vsync > 0 && !source_shown && emitted > 0 && emitted < limit && t > d && t - d < snap) {
    out.add({.t = {1, 1}, .pts_us = t});  // à l'heure du vsync, comme une source rattrapée avant lui
    last_emitted_ = t;
    ++emitted;
    base = std::max(base + step, t + spacing);
  }
  if (emitted == 0) {
    // Le GPU ne tient pas une image dans cette paire : la source seule, comme sans interpolation. Sur écran
    // fixe, au vsync le plus proche de son pts (jamais plus près du dernier instant que l'écart minimal) :
    // hors de la grille, elle pouvait viser le vblank de l'image d'avant, et l'une des deux manquait.
    const std::int64_t at =
        vsync > 0 ? on_vsync(std::max(d - vsync / 2, last_emitted_.value_or(INT64_MIN) + spacing)) : d;
    out.add({.t = {1, 1}, .pts_us = at});
    last_emitted_ = at;
    base = std::max(base, at + min_gap);
  }
  // Affamée par le pas : la source seule, alors qu'un instant tenait à un écart minimal avant elle.
  starved_ = interpolated == 0 && inputs.step_us > 0 && unheld <= d - min_gap;
  next_due_ = base;
  left_ = source;
  return out;
}

void FreeGrid::resync() {
  if (left_ && left_->pts_us) {
    last_emitted_ = left_->pts_us;
    next_due_.reset();
  }
}

void FreeGrid::reset() {
  left_.reset();
  next_due_.reset();
  last_emitted_.reset();
}

std::expected<void, Refusal> FreeGrid::flush(std::uint32_t epoch) {
  if (epoch < epoch_) {
    return std::unexpected(
        Refusal{RefusalCode::SequenceOrder, std::format("époque {} antérieure à {}", epoch, epoch_)});
  }
  epoch_ = epoch;
  reset();
  return {};
}

}  // namespace ajp::core
