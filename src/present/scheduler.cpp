#include "present/scheduler.h"

#include <algorithm>
#include <cmath>
#include <format>

#include "core/pacing.h"

namespace ajp::present {

namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

bool known(std::int64_t pts) {
  return pts != kNoPts;
}

}  // namespace

std::expected<SchedulerConfig, Refusal> make_scheduler_config(
    const core::DisplayInfo& display, const std::optional<core::CadenceChoice>& cadence,
    std::uint32_t generation) {
  SchedulerConfig config{.generation = generation};
  if (cadence) config.speed = cadence->speed;
  if (display.vrr) {
    config.mode = PaceMode::Vrr;
  } else if (cadence && cadence->grid == core::Grid::Vsync) {
    if (!(cadence->output_fps > 0.0)) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("cadence calée sans fps de sortie ({})", cadence->output_fps));
    }
    config.mode = PaceMode::VsyncGrid;
    config.vsyncs_per_frame = cadence->vsyncs_per_frame;
    config.frame_period_us = 1e6 / cadence->output_fps;
  }
  if (auto checked = PresentScheduler::create(config); !checked) return std::unexpected(checked.error());
  return config;
}

std::expected<PresentScheduler, Refusal> PresentScheduler::create(const SchedulerConfig& config) {
  if (!scheduler_limits::kSpeed.contains(config.speed)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("vitesse de lecture {} hors de [0,25, 4]", config.speed));
  }
  if (config.mode == PaceMode::VsyncGrid) {
    if (config.vsyncs_per_frame < 1 || config.vsyncs_per_frame > core::pacing::kMaxVsyncsPerFrame) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("{} vsyncs par image hors de [1, {}]", config.vsyncs_per_frame,
                                core::pacing::kMaxVsyncsPerFrame));
    }
    if (!scheduler_limits::kFramePeriodUs.contains(config.frame_period_us)) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("durée d'image {} µs hors de [100, 10000000]", config.frame_period_us));
    }
  }
  return PresentScheduler(config);
}

std::expected<void, Refusal> PresentScheduler::push(const RingFrame& frame) {
  if (held() >= ReleaseList::kCapacity) {
    // Seulement si app ne lit plus les décisions : app doit rendre ce slot elle-même. Accepter
    // l'image ferait déborder releases_ au prochain Flush (une file entière y part d'un coup).
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("slot {} refusé : l'ordonnanceur tient déjà {} images", frame.slot,
                              ReleaseList::kCapacity));
  }
  // D'un autre ring ou d'une époque passée : jamais montrée, mais rendue (core/ring.h).
  if (frame.generation != config_.generation || frame.epoch < epoch_) {
    ++stats_.stale;
    releases_.add(frame, false);
    return {};
  }
  if (frame.epoch > epoch_) {
    if (auto flushed = flush(frame.epoch); !flushed) return flushed;
  }
  if (known(frame.pts_us) && known(last_pushed_pts_) && frame.pts_us <= last_pushed_pts_) {
    releases_.add(frame, false);
    return refuse(RefusalCode::SequenceOrder,
                  std::format("image du slot {} à {} µs après {} µs dans l'époque {}", frame.slot,
                              frame.pts_us, last_pushed_pts_, frame.epoch));
  }
  if (queue_.full()) {
    releases_.add(frame, false);
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("file de présentation pleine ({} images) : slot {} rendu",
                              FrameQueue::kCapacity, frame.slot));
  }
  queue_.push_back(frame);
  if (known(frame.pts_us)) {
    if (known(last_pushed_pts_)) last_interval_us_ = frame.pts_us - last_pushed_pts_;
    last_pushed_pts_ = frame.pts_us;
  }
  return {};
}

std::expected<void, Refusal> PresentScheduler::flush(std::uint32_t epoch) {
  if (epoch < epoch_) {
    return refuse(RefusalCode::SequenceOrder,
                  std::format("Flush vers l'époque {} après l'époque {}", epoch, epoch_));
  }
  epoch_ = epoch;
  const std::size_t queued = queue_.size();
  for (std::size_t i = 0; i < queued; ++i) {
    const RingFrame frame = queue_.pop_front();
    if (frame.epoch < epoch_) {
      ++stats_.stale;
      releases_.add(frame, false);
    } else {
      queue_.push_back(frame);
    }
  }
  anchored_ = false;
  step_pending_ = false;
  last_pushed_pts_ = kNoPts;
  last_interval_us_ = 0;
  return {};
}

void PresentScheduler::set_generation(std::uint32_t generation) {
  stats_.stale += queue_.size();
  config_.generation = generation;
  while (!queue_.empty()) releases_.add(queue_.pop_front(), false);
  if (current_) releases_.add(*current_, true);
  current_.reset();
  anchored_ = false;
  step_pending_ = false;
  last_pushed_pts_ = kNoPts;
  last_interval_us_ = 0;
}

double PresentScheduler::tolerance(const VsyncTick& tick) const {
  switch (config_.mode) {
    case PaceMode::VsyncGrid:
      return vsync_step() / 2.0;
    case PaceMode::FixedTimestamps:
      return tick.period_us * config_.speed / 2.0;
    case PaceMode::Vrr:
      return 0.0;
  }
  return 0.0;
}

double PresentScheduler::clock_rate(const VsyncTick& tick) const {
  if (paused_) return 0.0;
  if (config_.mode == PaceMode::VsyncGrid && tick.period_us > 0.0) {
    return vsync_step() / tick.period_us;
  }
  return config_.speed;
}

double PresentScheduler::advance_clock(const VsyncTick& tick, const ClockSnapshot& master) {
  std::int64_t vsyncs = tick.index - last_index_;
  if (vsyncs < 0) vsyncs = 1;
  const auto host_elapsed = static_cast<double>(std::max<std::int64_t>(0, tick.host_us - last_host_us_));
  master_valid_ = config_.mode == PaceMode::Vrr && master.valid && master.epoch == epoch_;
  if (master_valid_) {
    clock_ = static_cast<double>(master.media_at(tick.host_us));
  } else if (anchored_ && !paused_) {
    if (config_.mode == PaceMode::VsyncGrid) {
      clock_ += static_cast<double>(vsyncs) * vsync_step();
    } else if (config_.mode == PaceMode::FixedTimestamps && tick.period_us > 0.0) {
      clock_ += static_cast<double>(vsyncs) * tick.period_us * config_.speed;
    } else {
      clock_ += host_elapsed * config_.speed;
    }
  }
  last_index_ = tick.index;
  last_host_us_ = std::max(last_host_us_, tick.host_us);
  return clock_;
}

void PresentScheduler::show(const RingFrame& frame) {
  if (current_) releases_.add(*current_, true);
  current_ = frame;
  ++stats_.shown;
}

void PresentScheduler::anchor(const RingFrame& frame) {
  anchored_ = true;
  if (known(frame.pts_us) && !master_valid_) clock_ = static_cast<double>(frame.pts_us);
}

bool PresentScheduler::pick_due(double target, double tolerance) {
  std::size_t due = 0;
  for (std::size_t i = 0; i < queue_.size(); ++i) {
    const RingFrame& frame = queue_.at(i);
    if (known(frame.pts_us) && static_cast<double>(frame.pts_us) > target + tolerance) break;
    due = i + 1;
  }
  if (due == 0) {
    count_underflow(target, tolerance);
    return false;
  }
  for (std::size_t i = 0; i + 1 < due; ++i) {
    ++stats_.skipped;
    releases_.add(queue_.pop_front(), false);
  }
  const RingFrame frame = queue_.pop_front();
  // En retard : passé de plus d'une demi-vsync (écran fixe) ou d'une demi-image (Vrr, quand
  // l'écart entre images est connu).
  const bool vrr = config_.mode == PaceMode::Vrr;
  const double late_after = vrr ? static_cast<double>(last_interval_us_) / 2.0 : tolerance;
  if (known(frame.pts_us) && (!vrr || last_interval_us_ > 0) &&
      target - static_cast<double>(frame.pts_us) > late_after) {
    ++stats_.late;
  }
  show(frame);
  return true;
}

void PresentScheduler::count_underflow(double target, double tolerance) {
  // Une image en file mais pas encore due n'est pas un manque.
  if (!queue_.empty() || !current_ || !known(current_->pts_us)) return;
  const double interval =
      config_.mode == PaceMode::VsyncGrid ? config_.frame_period_us : static_cast<double>(last_interval_us_);
  if (interval <= 0.0) return;
  if (static_cast<double>(current_->pts_us) + interval <= target + tolerance) ++stats_.repeated;
}

std::optional<std::int64_t> PresentScheduler::next_due(const ClockSnapshot& master) const {
  if (config_.mode != PaceMode::Vrr || queue_.empty() || paused_) return std::nullopt;
  const std::int64_t pts = queue_.at(0).pts_us;
  if (!known(pts)) return last_host_us_;
  if (master_valid_) {
    if (master.rate <= 0.0) return std::nullopt;
    return master.host_us + std::llround(static_cast<double>(pts - master.media_us) / master.rate);
  }
  if (!anchored_) return last_host_us_;
  return last_host_us_ + std::llround((static_cast<double>(pts) - clock_) / config_.speed);
}

PresentDecision PresentScheduler::on_vsync(const VsyncTick& tick, const ClockSnapshot& master) {
  releases_.forget_delivered();
  const double target = advance_clock(tick, master);
  bool changed = false;
  if (!anchored_) {
    // Première image de l'époque : montrée tout de suite, elle fixe l'origine de l'horloge.
    if (!queue_.empty()) {
      const RingFrame frame = queue_.pop_front();
      anchor(frame);
      show(frame);
      changed = true;
    }
  } else if (paused_) {
    if (step_pending_ && !queue_.empty()) {
      const RingFrame frame = queue_.pop_front();
      if (known(frame.pts_us)) clock_ = static_cast<double>(frame.pts_us);
      show(frame);
      step_pending_ = false;
      changed = true;
    }
  } else {
    changed = pick_due(target, tolerance(tick));
  }

  PresentDecision decision{.frame = current_, .changed = changed, .releases = releases_.deliver()};
  if (config_.mode != PaceMode::Vrr) {
    decision.clock = ClockSnapshot{
        .media_us = std::llround(clock_),
        .host_us = tick.host_us,
        .rate = clock_rate(tick),
        .epoch = epoch_,
        .valid = anchored_,
    };
  }
  decision.next_due_host_us = next_due(master);
  return decision;
}

}  // namespace ajp::present
