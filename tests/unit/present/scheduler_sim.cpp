#include "scheduler_sim.h"

#include <algorithm>
#include <cmath>

#include <catch2/catch_test_macros.hpp>

namespace ajp::present::test {

std::int64_t source_pts(double fps, std::int64_t index) {
  return std::llround(static_cast<double>(index) * 1e6 / fps);
}

Simulation::Simulation(const SchedulerConfig& config, const SimOptions& options)
    : scheduler_(PresentScheduler::create(config).value()),
      cadence_(core::CadenceAccumulator::create(options.factor == core::Ratio{1, 1} ? core::Ratio{2, 1}
                                                                                    : options.factor)
                   .value()),
      options_(options),
      busy_(static_cast<std::size_t>(options.ring_slots), false),
      released_value_(static_cast<std::size_t>(options.ring_slots), 0) {}

Simulation::Instant Simulation::next_instant() {
  const bool passthrough = options_.factor == core::Ratio{1, 1};
  while (instants_.empty()) {
    const std::int64_t i = next_source_++;
    const core::SourceStamp source{.id = SourceId{static_cast<std::uint32_t>(i)},
                                   .pts_us = source_pts(options_.source_fps, i),
                                   .epoch = epoch_};
    if (passthrough) {
      instants_.push_back({.pts = *source.pts_us, .left = source.id, .right = source.id, .t = {1, 1}});
      continue;
    }
    const core::CadenceStep step = cadence_.push(source);
    const SourceId left = step.event == core::GridEvent::Pair ? step.left : step.right;
    for (const core::OutputInstant& instant : step.instants()) {
      instants_.push_back(
          {.pts = instant.pts_us.value_or(kNoPts), .left = left, .right = step.right, .t = instant.t});
    }
  }
  const Instant next = instants_.front();
  instants_.pop_front();
  return next;
}

void Simulation::produce(std::int64_t index) {
  for (;;) {
    const auto free = std::find(busy_.begin(), busy_.end(), false);
    if (free == busy_.end()) return;
    if (delay_) {
      if (const auto earliest = delay_(produced_); earliest && index < *earliest) return;
    }
    const auto slot = static_cast<std::uint32_t>(free - busy_.begin());
    const Instant instant = next_instant();
    const std::uint64_t value = released_value_[slot] + 1;
    const RingFrame frame{.slot = slot,
                          .generation = options_.generation,
                          .pts_us = instant.pts,
                          .left = instant.left,
                          .right = instant.right,
                          .t_num = instant.t.num,
                          .t_den = instant.t.den,
                          .epoch = epoch_,
                          .ready_value = value};
    REQUIRE(scheduler_.push(frame));
    busy_[slot] = true;
    frame_of_[{slot, value}] = produced_;
    pts_.push_back(instant.pts);
    epochs_.push_back(epoch_);
    ++produced_;
  }
}

const VsyncRecord& Simulation::tick(std::int64_t index, std::int64_t host_us) {
  produce(index);
  const ClockSnapshot master = master_ ? master_(host_us) : ClockSnapshot{};
  const PresentDecision decision =
      scheduler_.on_vsync({.host_us = host_us, .index = index, .period_us = options_.period_us}, master);
  for (const FrameRelease& release : decision.releases) {
    const std::uint32_t slot = release.free.slot;
    REQUIRE(release.free.generation == options_.generation);
    REQUIRE(slot < busy_.size());
    REQUIRE(busy_[slot]);
    // ready_value + 1 : la valeur que gpu attendra avant de réécrire le slot.
    REQUIRE(release.free.semaphore_value == released_value_[slot] + 2);
    const std::int64_t frame = frame_of_.at({slot, released_value_[slot] + 1});
    const bool shown = std::any_of(records_.begin(), records_.end(),
                                   [frame](const VsyncRecord& r) { return r.frame == frame; });
    REQUIRE(release.displayed == shown);
    busy_[slot] = false;
    released_value_[slot] = release.free.semaphore_value;
    ++(release.displayed ? releases_displayed_ : releases_unseen_);
  }
  VsyncRecord record{.host_us = host_us, .changed = decision.changed, .clock = decision.clock};
  if (decision.frame) {
    REQUIRE(busy_[decision.frame->slot]);
    record.frame = frame_of_.at({decision.frame->slot, decision.frame->ready_value});
  }
  records_.push_back(record);
  return records_.back();
}

void Simulation::run(int vsyncs) {
  for (int i = 0; i < vsyncs; ++i) {
    const std::int64_t index = next_index_++;
    tick(index, std::llround(static_cast<double>(index) * options_.period_us));
  }
}

void Simulation::seek(std::uint32_t epoch, std::int64_t source_index) {
  REQUIRE(scheduler_.flush(epoch));
  if (!(options_.factor == core::Ratio{1, 1})) REQUIRE(cadence_.flush(epoch));
  epoch_ = epoch;
  instants_.clear();
  next_source_ = source_index;
}

int Simulation::busy_slots() const {
  return static_cast<int>(std::count(busy_.begin(), busy_.end(), true));
}

std::vector<std::pair<std::int64_t, int>> Simulation::runs(std::size_t from, std::size_t to) const {
  std::vector<std::pair<std::int64_t, int>> out;
  for (std::size_t i = from; i < std::min(to, records_.size()); ++i) {
    const auto& frame = records_[i].frame;
    if (!frame) continue;
    if (!out.empty() && out.back().first == *frame) {
      ++out.back().second;
    } else {
      out.emplace_back(*frame, 1);
    }
  }
  return out;
}

}  // namespace ajp::present::test
