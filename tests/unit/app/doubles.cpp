#include "doubles.h"

#include <format>
#include <utility>
#include <variant>

namespace ajp::app::play::test {

bool FakeStages::flush(std::uint32_t epoch) {
  log.push_back(std::format("flush {}", epoch));
  return true;
}

bool FakeStages::seek(const cmd::Seek& seek) {
  log.push_back(std::format("seek {}{}", seek.target_us, seek.exact ? "" : " rapide"));
  return true;
}

bool FakeStages::play(std::uint32_t epoch, std::int64_t media_us) {
  log.push_back(std::format("play {} {}", epoch, media_us));
  return true;
}

bool FakeStages::pause() {
  log.emplace_back("pause");
  return true;
}

bool FakeStages::slot_free(const SlotFree& free) {
  freed.push_back(free);
  return true;
}

bool FakeStages::request_subtitles(const subs::RenderRequest& request) {
  requests.push_back(request);
  return true;
}

std::expected<ScreenTiming, Refusal> FakeScreen::show(const ShowRequest& request) {
  now_us += period_us;
  shown.push_back(
      Shown{.frame = request.frame ? std::optional(*request.frame) : std::nullopt,
            .subtitles_pts = request.subtitles ? std::optional(request.subtitles->pts_us) : std::nullopt,
            .osd = {request.osd.begin(), request.osd.end()}});
  return ScreenTiming{.render_us = 1000, .present_us = now_us};
}

SlotFree FakeScreen::retire(const RingFrame& frame) {
  retired.push_back(frame);
  return SlotFree{
      .slot = frame.slot, .semaphore_value = frame.ready_value + 1, .generation = frame.generation};
}

RingFrame ring_frame(std::uint32_t slot, std::int64_t pts_us, std::uint32_t epoch,
                     std::uint64_t ready_value) {
  return RingFrame{.slot = slot,
                   .pts_us = pts_us,
                   .left = SourceId{static_cast<std::uint32_t>(pts_us / 1000 + 1)},
                   .right = SourceId{static_cast<std::uint32_t>(pts_us / 1000 + 1)},
                   .epoch = epoch,
                   .ready_value = ready_value};
}

std::expected<std::unique_ptr<VideoStage>, Refusal> PassthroughStage::start(VideoStageInit init,
                                                                            std::uint32_t slots) {
  return std::make_unique<PassthroughStage>(std::move(init), slots);
}

PassthroughStage::PassthroughStage(VideoStageInit init, std::uint32_t slots)
    : init_(std::move(init)),
      inbox_(StageInbox::Bounds{kMaxFrames, slots, 16, 4}),
      released_(slots, 0),
      free_(slots, true) {
  auto thread = start_thread("gpu-doublure", init_.report,
                             [this](const std::stop_token& stop, const ThreadReport&) { run(stop); });
  if (thread) thread_ = std::move(*thread);
}

bool PassthroughStage::try_push(DecodedFrame&& frame) {
  if (in_flight_.fetch_add(1) >= kMaxFrames || !inbox_.try_push(std::move(frame))) {
    --in_flight_;
    return false;
  }
  return true;
}

void PassthroughStage::run(const std::stop_token& stop) {
  while (auto message = inbox_.wait(stop)) {
    bool quit = false;
    std::visit(
        [&](auto& m) {
          using M = std::decay_t<decltype(m)>;
          if constexpr (std::is_same_v<M, DecodedFrame>) {
            pending_.push_back(std::move(m));
          } else if constexpr (std::is_same_v<M, SlotFree>) {
            if (m.generation == 0 && m.slot < free_.size()) {
              free_[m.slot] = true;
              released_[m.slot] = m.semaphore_value;
            }
          } else if constexpr (std::is_same_v<M, Flush>) {
            epoch_ = m.epoch;
          } else {
            quit = std::holds_alternative<cmd::Quit>(m);
          }
        },
        *message);
    if (quit) return;
    emit();
  }
}

void PassthroughStage::emit() {
  while (!pending_.empty()) {
    if (pending_.front().epoch < epoch_) {  // d'avant le Flush : jetée
      pending_.pop_front();
      --in_flight_;
      init_.credit();
      continue;
    }
    std::uint32_t slot = 0;
    while (slot < free_.size() && !free_[slot]) ++slot;
    if (slot == free_.size()) return;  // attend un SlotFree
    const DecodedFrame& frame = pending_.front();
    const RingFrame ring{.slot = slot,
                         .pts_us = frame.pts_us,
                         .left = frame.id,
                         .right = frame.id,
                         .epoch = frame.epoch,
                         .ready_value = released_[slot] + 1};
    if (!init_.ring->try_push(ring)) return;  // boîte de main pleine : au prochain message
    free_[slot] = false;
    pending_.pop_front();
    --in_flight_;
    init_.credit();
  }
}

}  // namespace ajp::app::play::test
