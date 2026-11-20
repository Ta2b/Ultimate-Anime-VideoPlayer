#include "app/play/cadence_watch.h"

#include <cmath>
#include <format>

#include "app/play/frame_stats.h"
#include "app/play/live_settings.h"

namespace kirari::app::play {

void ReadySeen::note(const Key& key, std::int64_t tick_us) {
  for (const Entry& e : entries_) {
    if (e.tick_us != 0 && e.key == key) return;  // la première fois seulement
  }
  entries_[next_] = Entry{.key = key, .tick_us = tick_us};
  next_ = (next_ + 1) % entries_.size();
}

bool ReadySeen::before(const Key& key, std::int64_t tick_us) const {
  for (const Entry& e : entries_) {
    if (e.tick_us != 0 && e.key == key) return e.tick_us < tick_us;
  }
  return false;  // inconnue : comptée comme avant, une perte du GPU
}

std::int64_t CadenceWatch::take_losses(const LossCounts& totals) {
  const LossCounts delta{.missing = totals.missing - seen_.missing,
                         .dropped = totals.dropped - seen_.dropped,
                         .dropped_ready = totals.dropped_ready - seen_.dropped_ready};
  seen_ = totals;
  const std::int64_t lost = pressure_losses(delta, options_.display_loss_pressure);
  display_losses_ += std::max<std::int64_t>(delta.dropped_ready, 0);
  gpu_losses_ += pressure_losses(delta, false);
  return lost;
}

void CadenceWatch::sample(bool active, const core::FreeInfo& info, std::int64_t min_gap_us) {
  if (!active || info.step_us <= 0 || info.fps <= 0.0) return;
  ++samples_;
  min_gap_us_ = min_gap_us;
  fps_sum_ += info.fps;
  step_sum_ += static_cast<double>(info.step_us);
  service_sum_ += static_cast<double>(info.service_us);
  pressure_sum_ += info.pressure;
  pressure_max_ = std::max(pressure_max_, info.pressure);
  const double service = static_cast<double>(info.service_us) * core::FreeRate::kMargin;
  if (info.step_us <= min_gap_us) {
    ++screen_bound_;
  } else if (service >= static_cast<double>(min_gap_us)) {
    ++service_bound_;
  } else {
    ++pressure_bound_;
  }
}

void CadenceWatch::poll(LiveSettings& live, const FrameStats& shown, std::int64_t now_us) {
  const std::int64_t lost =
      take_losses({.missing = shown.missing, .dropped = shown.dropped, .dropped_ready = shown.dropped_ready});
  const bool active = live.free_active();
  if (active) live.free_missed(now_us, lost);
  sample(active, live.free_info(now_us), live.free_step(now_us).second);
  motion(shown.moving_frames, shown.moving_us);
}

std::string CadenceWatch::summary() const {
  std::string text = std::format(
      "cadence haute : présentation {}, marge VRR {:.1f} %, pertes de l'affichage {} ({} la "
      "contre-pression), "
      "pertes du GPU {}",
      name(options_.swap_mode), options_.vrr_margin_pct, display_losses_,
      options_.display_loss_pressure ? "comptées dans" : "hors de", gpu_losses_);
  if (moving_us_ > 0) {
    text += std::format(" ; en mouvement (hors coupes et dessins tenus) {:.1f} i/s",
                        static_cast<double>(moving_frames_) * 1e6 / static_cast<double>(moving_us_));
  }
  if (samples_ == 0) return text + " ; mode Rapide inactif";
  const double n = samples_;
  const auto share = [n](int count) { return static_cast<int>(std::lround(100.0 * count / n)); };
  text += std::format(
      " ; mode Rapide : {:.1f} i/s planifiées en moyenne, écart minimal {:.2f} ms, écart planifié moyen {:.2f} ms, "
      "service moyen {:.2f} ms, contre-pression moyenne {:.2f} (max {:.2f}) ; borné par l'écran {} %, "
      "par le temps de service {} %, par la contre-pression {} %",
      fps_sum_ / n, static_cast<double>(min_gap_us_) / 1000.0, step_sum_ / n / 1000.0,
      service_sum_ / n / 1000.0, pressure_sum_ / n, pressure_max_, share(screen_bound_),
      share(service_bound_), share(pressure_bound_));
  return text;
}

}  // namespace kirari::app::play
