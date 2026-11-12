// La session et l'écran : période et délai d'affichage, calage de la vitesse sur la fréquence.
#include <algorithm>
#include <cmath>

#include "app/play/session.h"
#include "core/pacing.h"

namespace ajp::app::play {
namespace {

// Période d'écran supposée tant qu'aucune présentation rapprochée ne la borne : 60 Hz.
constexpr std::int64_t kAssumedPeriodUs = 16'667;
// Une acquisition qui attend plus d'un quart de période a trouvé la file de l'écran pleine ; la part de ces
// attentes se lisse sur une dizaine de présentations. La file est tenue pour pleine au-dessus de 0,6 et
// redevient libre sous 0,4.
constexpr double kQueueFillSmoothing = 0.1;
constexpr double kQueueFullAbove = 0.6;
constexpr double kQueueFreeBelow = 0.4;
// B13-09 : une période mesurée ne descend pas sous celle d'un écran de 500 Hz ; une présentation rapprochée
// (interface, présentation refaite) la faisait tomber à quelques centaines de µs pour 240 présentations.
constexpr std::int64_t kShortestPeriodUs = 2'000;

}  // namespace

// Mesurée faute de mieux : la présentation est à la demande, chaque intervalle est donc un
// multiple de la période (six à 24 i/s sur 144 Hz, des centaines avant la première image) et le
// plus court des derniers la borne. La médiane rendait la cadence du contenu : l'image suivante
// devenait due une période d'image trop tôt et les images partaient par paires.
std::int64_t Session::display_period() const {
  if (config_.display_period_us > 0) return config_.display_period_us;
  const std::int64_t shortest = stats_.present_interval.percentile(0);
  return shortest > 0 ? std::clamp(shortest, kShortestPeriodUs, kAssumedPeriodUs) : kAssumedPeriodUs;
}

// Imposé par le réglage, ou estimé : la file de l'écran a de la place, l'image part après son
// rendu et paraît au vsync suivant (une demi-période plus tard en moyenne) ; file pleine, les
// depth − 1 images déjà en file passent d'abord.
std::int64_t Session::display_delay_us(std::int64_t period) const {
  if (config_.video_latency_us > 0) return config_.video_latency_us;
  // P3.45 : en fenêtré, le compositeur garde l'image une trame de plus avant l'écran.
  const std::int64_t compositor = composited_ ? period : 0;
  if (queue_full_)
    return std::int64_t{std::max(config_.swapchain_depth - 1, 0)} * period + period / 2 + compositor;
  return stats_.render.median() + period / 2 + compositor;
}

void Session::note_queue_wait(std::int64_t acquire_us, std::int64_t period) {
  if (period <= 0) return;
  const double waited = acquire_us > std::max(period / 4, kAcquireBlockedUs) ? 1.0 : 0.0;
  queue_fill_ += kQueueFillSmoothing * (waited - queue_fill_);
  if (queue_fill_ > kQueueFullAbove) queue_full_ = true;
  if (queue_fill_ < kQueueFreeBelow) queue_full_ = false;
}

void Session::set_display(std::int64_t period_us, double hz) {
  // La fréquence mesurée par le retour de présentation (6.8) : 164,78 Hz et non les 165 annoncés, l'écart
  // de 0,13 % dépassant la borne de la synchro (pacing.h). Même mode à 2 % près ; à 0,002 Hz près, la
  // fréquence d'avant reste (pas de vitesse redemandée à chaque relevé).
  if (const auto grid = screen_.vblank_grid(); grid && grid->period_us > 0.0 && hz > 0.0) {
    const double measured = 1e6 / grid->period_us;
    if (std::abs(measured - hz) < hz * 0.02) {
      const bool same = config_.display_hz > 0.0 && std::abs(measured - config_.display_hz) < 0.002;
      hz = same ? config_.display_hz : measured;
      period_us = same ? config_.display_period_us : std::llround(grid->period_us);
    }
  }
  if (period_us == config_.display_period_us && hz == config_.display_hz) return;
  // Un autre mode ou un autre écran (pas le même à la mesure près) : la phase repart.
  if (config_.display_hz <= 0.0 || std::abs(hz - config_.display_hz) > config_.display_hz * 0.001)
    phase_trim_.reset();
  config_.display_period_us = period_us;
  config_.display_hz = hz;
  follow_display();
}

void Session::follow_display() {
  double speed = 1.0;
  std::optional<double> lock;
  if (config_.display_sync && config_.extra != nullptr) {
    const double hz = config_.display_hz > 0.0        ? config_.display_hz
                      : config_.display_period_us > 0 ? 1e6 / static_cast<double>(config_.display_period_us)
                                                      : 0.0;
    lock = core::vsync_lock(hz, config_.extra->output_fps());
  }
  // Calé sur la fréquence : la phase rattrape ce que la mesure de la fréquence laisse (A12-06). Sans
  // calage, la correction de phase repart de zéro.
  speed_locked_ = lock.has_value();
  if (lock) speed = *lock + phase_trim_.trim();
  if (!lock) phase_trim_.reset();
  if (speed == speed_) return;
  if (stages_.set_speed(speed)) {  // boîte audio pleine : redemandé au tour suivant
    speed_ = speed;
    osd_at_us_.reset();
  }
}

}  // namespace ajp::app::play
