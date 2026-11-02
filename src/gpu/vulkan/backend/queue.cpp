#include "gpu/vulkan/backend/queue.h"

#include <algorithm>
#include <format>
#include <utility>

namespace ajp::gpu::vk {
namespace {

// Lu par toute soumission, écrit par celle dont les accès sont inconnus : elle attend tout, et
// tout ce qui suit l'attend.
constexpr std::uintptr_t kUnknownAccess = 2;

}  // namespace

std::expected<BackendQueue, Refusal> BackendQueue::create(VkContext& context,
                                                          VkExternalSemaphoreHandleTypeFlags exports,
                                                          bool lanes) {
  // Les voies sur les files : l'agrandissement sur la principale (0), l'interpolation sur la
  // deuxième, la mesure sur la troisième ou, à défaut, avec l'agrandissement.
  const std::size_t files = lanes ? std::min<std::size_t>(context.queue_lanes(), kLanes) : 1;
  std::vector<LaneState> states;
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    auto timeline = TimelineSemaphore::create(context, 0, lane == kMain ? exports : 0);
    if (!timeline) return std::unexpected(timeline.error());
    std::size_t file = 0;
    if (lane == lane_index(Lane::Interpolate) && files >= 2) file = 1;
    if (lane == lane_index(Lane::Measure) && files >= 3) file = 2;
    states.push_back(LaneState{.timeline = std::move(*timeline), .file = file});
  }
  return BackendQueue(context, std::move(states), static_cast<int>(files));
}

void BackendQueue::issued(std::size_t lane) {
  if (files_ == 1) return;
  open_.push_back({.ticket = ++tickets_, .lane = lane, .value = lanes_[lane].value});
}

TimelinePoint BackendQueue::at(std::uint64_t ticket) const {
  if (files_ == 1) return lanes_[kMain].timeline.at(ticket);
  const auto found = std::ranges::lower_bound(open_, ticket, {}, &Ticket::ticket);
  if (found == open_.end() || found->ticket != ticket) return lanes_[kMain].timeline.at(0);  // fini
  return lanes_[found->lane].timeline.at(found->value);
}

std::expected<std::uint64_t, Refusal> BackendQueue::completed() {
  if (files_ == 1) return lanes_[kMain].timeline.value();
  std::array<std::uint64_t, kLanes> reached{};
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    auto value = lanes_[lane].timeline.value();
    if (!value) return std::unexpected(value.error());
    reached[lane] = *value;
  }
  while (!open_.empty() && open_.front().value <= reached[open_.front().lane]) open_.pop_front();
  return open_.empty() ? tickets_ : open_.front().ticket - 1;
}

TimelinePoint BackendQueue::mark_point(std::size_t lane, std::uint64_t mark) const {
  const LaneState& state = lanes_[lane];
  const std::size_t slot = mark % kMarks;
  // Rang gardé : sa valeur. Plus ancien que la place (écrasé) : la valeur, plus récente, qui l'a
  // remplacé. Jamais soumis (soumission refusée) : la dernière valeur de la voie.
  std::uint64_t value = state.last_mark_value;
  if (state.marks[slot] == mark || (state.marks[slot] > mark && state.marks[slot] != 0))
    value = state.values[slot];
  return state.timeline.at(value);
}

void BackendQueue::keep_mark(std::size_t lane, std::uint64_t mark) {
  LaneState& state = lanes_[lane];
  state.marks[mark % kMarks] = mark;
  state.values[mark % kMarks] = state.value;
  state.last_mark_value = state.value;
}

void BackendQueue::add_waits(const LaneHazards::Op& op) {
  for (const LaneHazards::Wait& wait : op.waits) {
    const TimelinePoint point = mark_point(lane_index(wait.lane), wait.mark);
    if (point.value > 0) waits_.push_back(point);
  }
}

void BackendQueue::wait(const TimelinePoint& point) {
  if (files_ == 1) {
    lanes_[kMain].pending.push_back(point);
    return;
  }
  for (LaneState& state : lanes_) state.pending.push_back(point);
}

Done BackendQueue::submit(Recording& recording, std::span<const TimelinePoint> extra_waits,
                          std::span<const TimelinePoint> extra_signals) {
  LaneState& state = lanes_[current_];
  waits_.assign(state.pending.begin(), state.pending.end());
  if (state.value > 0) waits_.push_back(state.timeline.at(state.value));
  std::array<std::uint64_t, 2> marks{};
  std::size_t mark_count = 0;
  if (files_ > 1) {
    const Lane lane = static_cast<Lane>(current_);
    if (recording.unknown_access()) {
      const LaneHazards::Op all = hazards_.join(lane);
      add_waits(all);
      marks[mark_count++] = all.mark;
      const std::array<std::uintptr_t, 1> unknown{kUnknownAccess};
      const LaneHazards::Op op = hazards_.access(lane, {}, unknown);
      add_waits(op);
      marks[mark_count++] = op.mark;
    } else {
      reads_.assign(recording.reads().begin(), recording.reads().end());
      reads_.push_back(kUnknownAccess);
      const LaneHazards::Op op = hazards_.access(lane, reads_, recording.writes());
      add_waits(op);
      marks[mark_count++] = op.mark;
    }
  }
  waits_.insert(waits_.end(), extra_waits.begin(), extra_waits.end());
  signals_.assign(1, state.timeline.at(state.value + 1));
  signals_.insert(signals_.end(), extra_signals.begin(), extra_signals.end());
  if (auto ok = context_->enqueue(recording, waits_, signals_, state.file); !ok) return ok;
  ++state.value;
  issued(current_);
  for (std::size_t i = 0; i < mark_count; ++i) keep_mark(current_, marks[i]);
  state.pending.clear();
  // Un sémaphore d'ailleurs (slot du ring, image décodée) est attendu hors du lot : il part.
  return extra_signals.empty() ? Done{} : context_->flush();
}

std::expected<TimelinePoint, Refusal> BackendQueue::joined() {
  LaneState& state = lanes_[current_];
  if (files_ == 1) return last();
  const LaneHazards::Op op = hazards_.join(static_cast<Lane>(current_));
  waits_.clear();
  add_waits(op);
  if (waits_.empty() && state.pending.empty()) {
    keep_mark(current_, op.mark);
    return last();
  }
  waits_.insert(waits_.end(), state.pending.begin(), state.pending.end());
  if (state.value > 0) waits_.push_back(state.timeline.at(state.value));
  signals_.assign(1, state.timeline.at(state.value + 1));
  if (auto ok = context_->enqueue_join(waits_, signals_, state.file); !ok) return std::unexpected(ok.error());
  ++state.value;
  issued(current_);
  keep_mark(current_, op.mark);
  state.pending.clear();
  return last();
}

std::optional<BackendQueue::Step> BackendQueue::external_step() {
  if (!lanes_[current_].pending.empty()) return std::nullopt;
  return host_step();
}

BackendQueue::Step BackendQueue::host_step() {
  LaneState& state = lanes_[current_];
  const Step step{.wait = state.value, .signal = state.value + 1};
  ++state.value;
  return step;
}

Done BackendQueue::wait_elsewhere(std::uint64_t value, std::chrono::nanoseconds timeout) const {
  if (value == 0) return {};
  const VkSemaphore semaphore = lanes_[current_].timeline.handle();
  const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                 .semaphoreCount = 1,
                                 .pSemaphores = &semaphore,
                                 .pValues = &value};
  const auto nanoseconds = static_cast<std::uint64_t>(timeout.count() < 0 ? 0 : timeout.count());
  return check(context_->fn().vkWaitSemaphores(context_->device(), &wait, nanoseconds),
               std::format("vkWaitSemaphores(valeur {}, autre fil)", value));
}

}  // namespace ajp::gpu::vk
