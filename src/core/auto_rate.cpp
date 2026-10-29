#include "core/auto_rate.h"

#include <algorithm>
#include <cmath>
#include <limits>

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
  if (display_hz <= 0.0 || source_fps < 5.0 || source_fps > 240.0) return granularity;
  int best = granularity;
  for (int step = 2 * granularity; step <= kAutoMaxStep * granularity; ++step) {
    if (source_fps * step / granularity <= display_hz * 1.01) best = step;
  }
  return best;
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
    step_ = ceiling_;
    reason_ = AutoReason::Ceiling;
    ++changes_;
    clean_ = 0;
    probing_ = false;
  }
}

void AutoRate::seed(int remembered_centi, std::int64_t now_us) {
  hint_ = std::clamp(remembered_centi * granularity_ / 100, granularity_, ceiling_);
  step_ = std::clamp(hint_ - (fine() ? std::max(2, hint_ * 8 / 100) : 1), granularity_, ceiling_);
  reason_ = AutoReason::Memory;
  last_change_us_ = now_us - kMinRiseGapUs;
  beat_us_ = now_us;
}

double AutoRate::weight_of(const ImageCost& c, std::int64_t now_us) const {
  if (c.weight <= 0.0) return 0.0;
  const double age = static_cast<double>(std::max<std::int64_t>(now_us - c.at_us, 0));
  return c.weight * std::exp2(-age / static_cast<double>(kHalfLifeUs));
}

void AutoRate::cost(int images, std::int64_t cost_us, std::int64_t now_us) {
  if (cost_us <= 0 || cost_us > 5'000'000) return;
  ImageCost& c = costs_[static_cast<std::size_t>(std::clamp(images, 1, kMaxImages))];
  double w = weight_of(c, now_us);
  const double value = static_cast<double>(cost_us);
  if (w < 0.5) {  // une autre scène : la moyenne d'avant ne vaut plus
    c.ewma_us = value;
    c.filled = 0;
    c.next = 0;
    w = 0.0;
  } else {
    c.ewma_us += std::max(0.15, 1.0 / (w + 1.0)) * (value - c.ewma_us);
  }
  c.weight = std::min(w + 1.0, 24.0);
  c.at_us = now_us;
  c.ring[c.next] = static_cast<float>(value);
  c.next = (c.next + 1) % c.ring.size();
  c.filled = std::min(c.filled + 1, c.ring.size());
}

std::optional<AutoRate::Line> AutoRate::fit(std::int64_t now_us) const {
  double w_sum = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  for (int n = 1; n <= kMaxImages; ++n) {
    const ImageCost& c = costs_[static_cast<std::size_t>(n)];
    const double w = std::min(weight_of(c, now_us), 12.0);
    if (w < 2.0) continue;
    const double x = n;
    w_sum += w;
    sx += w * x;
    sy += w * c.ewma_us;
    sxx += w * x * x;
    sxy += w * x * c.ewma_us;
  }
  if (w_sum < 3.0) return std::nullopt;
  const double spread = w_sum * sxx - sx * sx;
  if (spread < 1e-6 * w_sum * w_sum) {
    // Un seul nombre d'images vu : le mouvement pèse 40 % du coût, le reste se répartit par image.
    const double y = sy / w_sum;
    const double x = sx / w_sum;
    return Line{.a = 0.4 * y, .b = 0.6 * y / x};
  }
  double b = (w_sum * sxy - sx * sy) / spread;
  double a = (sy - b * sx) / w_sum;
  if (b < 0.0) {
    b = 0.0;
    a = sy / w_sum;
  } else if (a < 0.0) {
    a = 0.0;
    b = sxy / sxx;
  }
  return Line{.a = a, .b = b};
}

double AutoRate::spike(std::int64_t now_us) const {
  const ImageCost* best = &costs_[1];
  for (const ImageCost& e : costs_) {
    if (weight_of(e, now_us) > weight_of(*best, now_us)) best = &e;
  }
  const ImageCost& c = *best;
  if (weight_of(c, now_us) < 1.0 || c.filled < 6 || c.ewma_us <= 0.0) return 1.0;
  std::array<float, 16> sorted = c.ring;
  const auto end = sorted.begin() + static_cast<std::ptrdiff_t>(c.filled);
  const auto p90 = sorted.begin() + static_cast<std::ptrdiff_t>((c.filled - 1) * 9 / 10);
  std::nth_element(sorted.begin(), p90, end);
  return std::clamp(static_cast<double>(*p90) / c.ewma_us, 1.0, 1.5);
}

double AutoRate::penalty(int step, std::int64_t now_us) const {
  const Fail& f = fails_[static_cast<std::size_t>(std::clamp(step, 0, kAutoMaxStep * kFineGranularity + 1))];
  if (f.extra <= 0.0) return 1.0;
  const double age = static_cast<double>(std::max<std::int64_t>(now_us - f.at_us, 0));
  return 1.0 + f.extra * std::exp2(-age / static_cast<double>(kHalfLifeUs));
}

double AutoRate::model_us(int step, std::int64_t now_us) const {
  const std::optional<Line> line = fit(now_us);
  if (!line) return -1.0;
  return (line->a + line->b * static_cast<double>(step) / granularity_) * spike(now_us);
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
  const std::optional<Line> line = fit(now_us);
  if (!line) return 0;
  const double bump = spike(now_us);
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
  hold_until_us_ = now_us + (fine() ? kFineHoldUs : kHoldUs);
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
    hold_until_us_ = now_us + kHoldUs;
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
  step_ = std::max(granularity_, step_ - std::max(steps, 1));
  probing_ = false;
  changed(reason, now_us);
}

void AutoRate::try_rise(std::int64_t now_us) {
  const std::int64_t min_gap = fine() ? kFineGapUs : kMinRiseGapUs;
  if (step_ >= ceiling_ || pairs_ < kMinPairs || clean_ < (fine() ? kFineRisePairs : kRisePairs) ||
      now_us < hold_until_us_ || now_us - last_change_us_ < min_gap || missed_in_window(now_us) > 0)
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
    // Au pas fin, sans mesure fraîche (la cadence est tombée à ×1 : aucune image créée, donc aucun coût
    // relevé), on sonde vite : un cinquième du palier toutes les 6 paires propres, sans plafond appris.
    const std::uint32_t needed =
        fine() ? kFineRisePairs * 2 : (cap_strength(now_us) >= 0.25 ? kBlindRisePairs * 4 : kBlindRisePairs);
    if (clean_ >= needed) target = step_ + (fine() ? std::max(2, step_ / 5) : 1);
  }
  int gain = target - step_;
  if (fine()) {
    // Pas de bande morte : un pas de cadence (~1 i/s) suffit. La montée rejoint la cible par un lissage
    // léger proportionnel au temps écoulé depuis le dernier changement ; jamais plus que l'écart.
    const double seconds = static_cast<double>(now_us - last_change_us_) / 1e6;
    const double alpha = std::clamp(seconds / kFineSmoothS, 0.1, 1.0);
    gain = std::min(gain, std::max(1, static_cast<int>(std::lround(static_cast<double>(gain) * alpha))));
    // Pas de saut brutal dans l'inconnu : la montée gagne au plus 40 % du palier par seconde (la prédiction
    // vient de mesures de moins en moins fraîches quand la cadence a beaucoup baissé).
    const double slope_seconds = std::min(seconds, 1.0);
    gain = std::min(
        gain,
        std::max(1, static_cast<int>(std::lround(static_cast<double>(step_) * kFineSlope * slope_seconds))));
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
  if (period_us_ > 0 && step_ > granularity_ && (!fine() || now - last_change_us_ >= kFineGapUs)) {
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
    descend(AutoReason::Watchdog, now_us);
  }
  return step_;
}

void AutoRate::scene_cut(std::int64_t now_us) {
  if (cap_ > 0) cap_at_us_ -= kHalfLifeUs;  // la moitié de sa force
  for (ImageCost& c : costs_) {
    const double w = weight_of(c, now_us);
    c.weight = w * 0.5;
    c.at_us = now_us;
  }
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
