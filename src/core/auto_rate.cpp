#include "core/auto_rate.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/pacing.h"

namespace ajp::core {
namespace {

constexpr std::int64_t kNever = std::numeric_limits<std::int64_t>::min() / 4;

}  // namespace

int auto_granularity(int wanted) {
  return wanted <= 1 ? 1 : wanted == 2 ? 2 : kFineGranularity;
}

int auto_ceiling(double display_hz, double source_fps, int granularity) {
  granularity = auto_granularity(granularity);
  // Hors de [5, 240] i/s : une cadence fausse (VFR, base de temps) ; l'interpolation reste à ×1.
  if (!(display_hz > 0.0) || !(source_fps >= 5.0)  // NaN comme 0 : écran ou cadence inconnus
      || source_fps < 5.0 || source_fps > 240.0)
    return granularity;
  int best = granularity;
  for (int step = 2 * granularity; step <= kAutoMaxStep * granularity; ++step) {
    if (source_fps * step / granularity <= display_hz * 1.01) best = step;
  }
  return best;
}

std::vector<int> locked_steps(double display_hz, double source_fps, int granularity) {
  granularity = auto_granularity(granularity);
  if (!(display_hz > 0.0) || !(source_fps >= 5.0) || source_fps > 240.0) return {};
  std::vector<int> out{granularity};
  for (int step = granularity + 1; step <= kAutoMaxStep * granularity; ++step) {
    const double vsyncs = display_hz * granularity / (source_fps * step);
    if (vsyncs < 0.99) break;
    if (std::abs(vsyncs - std::round(vsyncs)) <= 0.01 * vsyncs) out.push_back(step);
  }
  return out;
}

namespace {

// Les paliers calés d'un écran fixe et leur facteur exact, du plus bas au plus haut ; rien si l'écran ou la
// cadence sont inconnus. cap_hz > 0 : aucune cadence de sortie au-dessus.
void lock_to_display(AutoLimits& out, double display_hz, double source_fps, double cap_hz) {
  if (!(display_hz > 0.0) || !(source_fps >= 5.0) || source_fps > 240.0) return;
  constexpr double kLowest = 1.49;  // ×1,5 à 1 % près
  out.locked.push_back(out.granularity);
  out.exact.push_back(Ratio{1, 1});
  for (int vsyncs = pacing::kMaxVsyncsPerFrame; vsyncs >= 1; --vsyncs) {
    const double target = display_hz / (static_cast<double>(vsyncs) * source_fps);
    if (target < kLowest || target > static_cast<double>(kAutoMaxStep)) continue;
    if (cap_hz > 0.0 && target * source_fps > cap_hz * 1.01) continue;
    const std::optional<Ratio> factor = lock_factor(target);
    if (!factor) continue;
    const int step = static_cast<int>(std::lround(factor->value() * out.granularity));
    if (step <= out.granularity) continue;
    if (step == out.locked.back()) {  // deux facteurs pour un même arrondi : le plus fluide
      out.exact.back() = *factor;
      continue;
    }
    out.locked.push_back(step);
    out.exact.push_back(*factor);
  }
}

}  // namespace

AutoLimits auto_limits(double display_hz, double source_fps, bool vrr, int level_factor, double cap_hz) {
  AutoLimits out;
  out.granularity = vrr ? kFineGranularity : 2;
  const double limit_hz = cap_hz > 0.0 && display_hz > 0.0 ? std::min(display_hz, cap_hz) : display_hz;
  out.ceiling = auto_ceiling(limit_hz, source_fps, out.granularity);
  if (!vrr) {
    lock_to_display(out, display_hz, source_fps, cap_hz);
    // Un facteur exact s'arrondit parfois au-dessus du plafond en demi-paliers (×6,88 : 14 pour 13).
    if (!out.locked.empty()) out.ceiling = std::max(out.ceiling, out.locked.back());
  }
  if (level_factor >= 2) {
    const int screen = out.ceiling;
    out.ceiling = std::min(out.ceiling, level_factor * out.granularity + (out.granularity == 2 ? 1 : 0));
    // A37-14 : à fréquence fixe, le premier palier calé au-dessus du plafond du niveau reste permis s'il ne
    // dépasse ce niveau que d'un multiplicateur (×5 pour ×4 à 120 et 240 Hz, en 23,976 i/s) : sans lui, le
    // Stable tombe au palier d'en dessous (×2,5). Le coût mesuré redescend ensuite si ×5 ne tient pas.
    const auto above = std::ranges::upper_bound(out.locked, out.ceiling);
    if (above != out.locked.end() && *above <= std::min(screen, (level_factor + 1) * out.granularity))
      out.ceiling = *above;
  }
  return out;
}

int AutoRate::centi() const {
  if (exact_.size() != locked_.size() || exact_.empty()) return step_ * 100 / granularity_;
  return static_cast<int>(std::lround(factor().value() * 100.0));
}

Ratio AutoRate::factor() const {
  if (exact_.size() == locked_.size()) {
    const auto at = std::ranges::find(locked_, step_);
    if (at != locked_.end()) return exact_[static_cast<std::size_t>(at - locked_.begin())];
  }
  return Ratio{step_, granularity_};
}

int AutoRate::snap_down(int step) const {
  if (locked_.empty()) return step;
  int best = granularity_;
  for (const int s : locked_)
    if (s <= step) best = std::max(best, s);
  return best;
}

int AutoRate::snap_up(int step) const {
  if (locked_.empty()) return step;
  for (const int s : locked_)
    if (s >= step) return s;
  return 0;
}

void AutoRate::set_locked(std::vector<int> steps, std::vector<Ratio> exact) {
  if (exact.size() != steps.size()) exact.clear();
  if (exact.empty()) std::ranges::sort(steps);  // avec leurs facteurs, les paliers viennent déjà en ordre
  if (steps == locked_ && exact == exact_) return;
  exact_ = std::move(exact);
  if (steps == locked_) return;
  locked_ = std::move(steps);
  const int snapped = snap_down(step_);
  if (snapped != step_) {
    step_ = snapped;
    reason_ = AutoReason::Ceiling;
    ++changes_;
    clean_ = 0;
    probing_ = false;
  }
}

std::string_view auto_reason_text(AutoReason reason) {
  switch (reason) {
    case AutoReason::Start:
      return "départ";
    case AutoReason::Memory:
      return "départ mémorisé";
    case AutoReason::Rise:
      return "scène allégée";
    case AutoReason::Late:
      return "paires en retard";
    case AutoReason::Missed:
      return "images manquées";
    case AutoReason::Predicted:
      return "coût au-dessus de la période";
    case AutoReason::Watchdog:
      return "plus aucune paire";
    case AutoReason::Ceiling:
      return "plafond de l'écran";
    case AutoReason::Rearm:
      return "réarmement au plancher";
  }
  return {};
}

AutoRate::AutoRate(int ceiling, int granularity)
    : granularity_(auto_granularity(granularity)),
      ceiling_(std::clamp(ceiling, granularity_, kAutoMaxStep * granularity_)),
      step_(std::min(2 * granularity_, ceiling_)) {
  late_at_.fill(kNever);
  miss_.fill({kNever, 0});
}

void AutoRate::set_ceiling(int ceiling, int granularity) {
  granularity = auto_granularity(granularity);
  if (granularity != granularity_) {
    // D'une unité à l'autre, l'arrondi est vers le bas : jamais un palier plus haut que celui qui tenait.
    // Les coûts se comptent en images créées, ils passent tels quels.
    const int from = granularity_;
    const auto convert = [&](int value) { return value > 0 ? std::max(value * granularity / from, 1) : 0; };
    step_ = std::max(convert(step_), granularity);
    cap_ = convert(cap_);
    hint_ = convert(hint_);
    fails_.fill({});
    granularity_ = granularity;
    clean_ = 0;
    probing_ = false;
  }
  set_ceiling(ceiling);
}

void AutoRate::set_ceiling(int ceiling) {
  ceiling_ = std::clamp(ceiling, granularity_, kAutoMaxStep * granularity_);
  if (cap_ > ceiling_) cap_ = 0;
  if (step_ > ceiling_) {
    step_ = snap_down(ceiling_);
    reason_ = AutoReason::Ceiling;
    ++changes_;
    clean_ = 0;
    probing_ = false;
  }
}

void AutoRate::seed(int remembered_centi, std::int64_t now_us) {
  hint_ = std::clamp(remembered_centi * granularity_ / 100, granularity_, ceiling_);
  step_ = snap_down(std::clamp(hint_ - (fine() ? std::max(2, hint_ * 8 / 100) : 1), granularity_, ceiling_));
  reason_ = AutoReason::Memory;
  last_change_us_ = now_us - kMinRiseGapUs;
  beat_us_ = now_us;
}

void AutoRate::cost(int images, std::int64_t cost_us, std::int64_t now_us) {
  model_.add(images, cost_us, now_us);
}

double AutoRate::penalty(int step, std::int64_t now_us) const {
  const Fail& f = fails_[static_cast<std::size_t>(std::clamp(step, 0, kAutoMaxStep * kFineGranularity + 1))];
  if (f.extra <= 0.0) return 1.0;
  const double age = static_cast<double>(std::max<std::int64_t>(now_us - f.at_us, 0));
  return 1.0 + f.extra * std::exp2(-age / static_cast<double>(kHalfLifeUs));
}

double AutoRate::model_us(int step, std::int64_t now_us) const {
  const std::optional<CostLine> line = model_.fit(now_us);
  if (!line) return -1.0;
  return (line->a + line->b * static_cast<double>(step) / granularity_) * model_.spike(now_us);
}

double AutoRate::cap_strength(std::int64_t now_us) const {
  if (cap_ <= 0) return 0.0;
  const double age = static_cast<double>(std::max<std::int64_t>(now_us - cap_at_us_, 0));
  return std::exp2(-age / static_cast<double>(kHalfLifeUs));
}

int AutoRate::learned_cap(std::int64_t now_us) const {
  return cap_strength(now_us) >= 0.25 ? cap_ : 0;
}

int AutoRate::predicted_step(std::int64_t now_us) const {
  if (period_us_ <= 0) return 0;
  const std::optional<CostLine> line = model_.fit(now_us);
  if (!line) return 0;
  const double bump = model_.spike(now_us);
  const bool capped = cap_strength(now_us) >= 0.25;
  int best = granularity_;
  for (int s = granularity_; s <= ceiling_; ++s) {
    const double margin = capped && s > cap_ ? kCapMargin : kMargin;
    const double t = (line->a + line->b * static_cast<double>(s) / granularity_) * bump * penalty(s, now_us);
    if (t <= static_cast<double>(period_us_) * (1.0 - margin)) best = s;
  }
  return best;
}

int AutoRate::late_in_window(std::int64_t now_us) const {
  return static_cast<int>(
      std::ranges::count_if(late_at_, [&](std::int64_t at) { return now_us - at <= kLateWindowUs; }));
}

int AutoRate::missed_in_window(std::int64_t now_us) const {
  int sum = 0;
  for (const auto& [at, count] : miss_) {
    if (now_us - at <= kMissWindowUs) sum += count;
  }
  return sum;
}

void AutoRate::changed(AutoReason reason, std::int64_t now_us) {
  reason_ = reason;
  ++changes_;
  last_change_us_ = now_us;
  hold_until_us_ = now_us + kHoldUs;
  clean_ = 0;
  late_at_.fill(kNever);
  miss_.fill({kNever, 0});
}

int AutoRate::drop_for(int predicted) const {
  if (!fine()) return predicted > 0 ? std::clamp(step_ - predicted, 1, 2) : 1;
  // Au pas fin : de 8 % à 25 % du palier, vers ce que la prédiction tient.
  const int low = std::max(1, step_ * 8 / 100);
  const int high = std::max(low, step_ / 4);
  return std::clamp(predicted > 0 ? step_ - predicted : 0, low, high);
}

void AutoRate::descend(AutoReason reason, std::int64_t now_us, int steps) {
  if (step_ <= granularity_) {  // le plancher : rien à descendre, la mesure repart
    // Pas de temps de repos ici : des images manquées qui continuent (trous du contenu) le repousseraient
    // sans fin et la montée ne reviendrait jamais.
    clean_ = 0;
    late_at_.fill(kNever);
    miss_.fill({kNever, 0});
    return;
  }
  if (probing_ &&
      (reason == AutoReason::Late || reason == AutoReason::Missed || reason == AutoReason::Predicted)) {
    // Le palier qui vient d'être tenté échoue : sa prédiction passe pour trop optimiste un moment, et
    // le palier d'en dessous devient le plafond appris, qui s'estompe.
    Fail& f = fails_[static_cast<std::size_t>(step_)];
    f.extra = std::min(f.extra * std::exp2(-static_cast<double>(now_us - f.at_us) / kHalfLifeUs) + 0.12, 0.6);
    f.at_us = now_us;
    cap_ = step_ - 1;
    cap_at_us_ = now_us;
  }
  step_ = snap_down(std::max(granularity_, step_ - std::max(steps, 1)));
  probing_ = false;
  changed(reason, now_us);
}

void AutoRate::try_rise(std::int64_t now_us) {
  if (step_ >= ceiling_ || pairs_ < kMinPairs || clean_ < kRisePairs || now_us < hold_until_us_ ||
      now_us - last_change_us_ < kMinRiseGapUs || missed_in_window(now_us) > 0)
    return;
  const int predicted = predicted_step(now_us);
  int target = step_;
  if (predicted > 0) {
    target = predicted;
    if (target <= step_ &&
        hint_ > step_) {  // le dernier palier tenu pour cette vidéo : on y revient s'il y a de l'air
      const double t = model_us(step_, now_us);
      if (t >= 0.0 && t < 0.9 * static_cast<double>(period_us_)) target = hint_;
    }
  } else {  // aucune mesure de coût : comme avant, lentement, et plus lentement sous un plafond appris
    const std::uint32_t needed = cap_strength(now_us) >= 0.25 ? kBlindRisePairs * 4 : kBlindRisePairs;
    if (clean_ >= needed) target = step_ + 1;
  }
  int gain = target - step_;
  if (fine()) {
    // Bande morte de 6 % ; en régime un changement toutes les 5 s au plus, de la moitié de l'écart (1 à 12 %
    // du palier) ; loin sous le palier prédit (scène allégée) : les deux tiers de l'écart, jusqu'à 40 %, dès
    // la seconde suivante.
    if (gain < std::max(1, static_cast<int>(step_ * kDeadBand))) return;
    const bool far = static_cast<double>(gain) >= kFarRatio * static_cast<double>(step_);
    if (!far && now_us - last_change_us_ < kSteadyGapUs) return;
    gain = far ? std::clamp(gain * 2 / 3, 1, std::max(1, step_ * 40 / 100))
               : std::clamp((gain + 1) / 2, 1, std::max(1, step_ * 12 / 100));
  } else if (!locked_.empty()) {
    // Le palier calé suivant, s'il tient sous la prédiction (ou à l'aveugle, après les paires propres).
    const int next = snap_up(step_ + 1);
    if (next == 0 || next > ceiling_ || (predicted > 0 && next > target) || target <= step_) return;
    gain = next - step_;
  } else {
    gain = std::min(gain, 1);
  }
  if (gain < 1) return;
  step_ = std::min(step_ + gain, ceiling_);
  probing_ = true;
  probe_until_us_ = now_us + kProbeUs;
  changed(AutoReason::Rise, now_us);
}

int AutoRate::pair(const PairOutcome& outcome) {
  const std::int64_t now = outcome.now_us;
  beat_us_ = now;
  mute_ = false;
  if (outcome.period_us > 0) period_us_ = outcome.period_us;
  // Faite au palier d'avant, ou dans le temps de repos d'un changement : elle ne dit rien du palier d'ici.
  if (outcome.made_step != step_ || now < hold_until_us_) return step_;
  ++pairs_;
  if (outcome.late > 0) {
    clean_ = 0;
    late_at_[late_next_++ % late_at_.size()] = now;
    if (late_in_window(now) >= kLatePairs) {
      const int predicted = predicted_step(now);
      descend(AutoReason::Late, now, drop_for(predicted));
    }
    return step_;
  }
  ++clean_;
  if (probing_ && now >= probe_until_us_) {  // le palier tenté tient
    probing_ = false;
    fails_[static_cast<std::size_t>(step_)] = {};
    if (cap_ > 0 && step_ > cap_) cap_ = 0;
  }
  // Le coût mesuré dépasse la période de la source : ce palier ne se tient pas, même sans retard encore.
  if (period_us_ > 0 && step_ > granularity_ && (!fine() || now - last_change_us_ >= kMinRiseGapUs)) {
    const double t = model_us(step_, now) * penalty(step_, now);
    if (t > static_cast<double>(period_us_)) {
      const int predicted = predicted_step(now);
      descend(AutoReason::Predicted, now, drop_for(predicted));
      return step_;
    }
  }
  try_rise(now);
  return step_;
}

int AutoRate::missed(std::int64_t now_us, std::int64_t delta) {
  if (delta <= 0 || now_us < hold_until_us_) return step_;
  miss_[miss_next_++ % miss_.size()] = {now_us, static_cast<int>(std::min<std::int64_t>(delta, 1000))};
  if (missed_in_window(now_us) >= kMissLimit) descend(AutoReason::Missed, now_us);
  return step_;
}

int AutoRate::tick(std::int64_t now_us, bool playing) {
  if (!playing) {  // pause, fenêtre cachée : aucune paire n'est attendue
    beat_us_ = now_us;
    return step_;
  }
  if (now_us - beat_us_ >= kWatchdogUs) {
    beat_us_ = now_us;
    pairs_ = 0;  // plus de montée avant que les paires reviennent
    mute_ = true;
    descend(AutoReason::Watchdog, now_us);
  }
  // Réarmement : au plancher alors que l'écran permet plus, la régulation ne reste pas figée. Les montées
  // normales veulent des paires propres, aucune image manquée et un plafond appris estompé ; si l'une de ces
  // conditions ne se lève jamais, un cran d'essai toutes les kFloorEscapeUs. S'il échoue, la descente
  // habituelle le reprend.
  if (step_ <= granularity_ && ceiling_ > granularity_ && !mute_) {
    if (floor_since_us_ == 0) floor_since_us_ = now_us;
    if (now_us - floor_since_us_ >= kFloorEscapeUs && now_us - last_change_us_ >= kFloorEscapeUs) {
      // Aucun palier permis au-dessus du plancher (écran fixe sans palier calé, ou le premier au-dessus
      // du plafond) : rien à tenter, ni changement compté ni mesures ignorées (A12-05).
      const int up = snap_up(granularity_ + (fine() ? std::max(2, granularity_ * 8 / 100) : 1));
      const int target = locked_.empty() ? std::min(up, ceiling_) : up;
      if (target > granularity_ && target <= ceiling_) {
        step_ = target;
        probing_ = true;
        probe_until_us_ = now_us + kProbeUs;
        changed(AutoReason::Rearm, now_us);
      }
      floor_since_us_ = 0;
    }
  } else {
    floor_since_us_ = 0;
  }
  return step_;
}

void AutoRate::scene_cut(std::int64_t now_us) {
  if (cap_ > 0) cap_at_us_ -= kHalfLifeUs;  // la moitié de sa force
  model_.fade(now_us);
  for (Fail& f : fails_) f.extra *= 0.5;
}

void AutoRate::settle(std::int64_t now_us) {
  hold_until_us_ = std::max(hold_until_us_, now_us + kSettleUs);
  pairs_ = 0;
  clean_ = 0;
  beat_us_ = now_us;
  probing_ = false;
  late_at_.fill(kNever);
  miss_.fill({kNever, 0});
}

AutoInfo AutoRate::info(std::int64_t now_us) const {
  AutoInfo out;
  out.step = step_;
  out.centi = centi();
  out.granularity = granularity_;
  out.ceiling = ceiling_;
  out.predicted = predicted_step(now_us);
  out.cap = learned_cap(now_us);
  out.reason = reason_;
  out.changes = changes_;
  const double t = model_us(step_, now_us);
  if (period_us_ > 0 && t >= 0.0) {
    out.load_pct = static_cast<int>(std::lround(t * 100.0 / static_cast<double>(period_us_)));
    out.heavy = out.load_pct >= 85;
  }
  return out;
}

}  // namespace ajp::core
