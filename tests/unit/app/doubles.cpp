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

bool FakeStages::audio(const Command& command) {
  if (const auto* volume = std::get_if<cmd::SetVolume>(&command)) {
    log.push_back(std::format("volume {:.2f}", volume->gain));
  } else if (const auto* delay = std::get_if<cmd::SetAudioDelay>(&command)) {
    log.push_back(std::format("délai audio {}", delay->delay_us));
  }
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
            .osd = {request.osd.begin(), request.osd.end()},
            .ui = request.ui != nullptr});
  return ScreenTiming{.render_us = 1000, .present_us = now_us, .queue_wait_us = queue_wait_us};
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
  std::unique_ptr<PassthroughStage> stage(new PassthroughStage(std::move(init), slots));
  if (auto started = stage->RingStage::start("gpu-doublure"); !started) {
    return std::unexpected(std::move(started.error()));
  }
  return stage;
}

}  // namespace ajp::app::play::test
