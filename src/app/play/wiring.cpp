#include "app/play/wiring.h"

#include <utility>

namespace ajp::app::play {

ClockSnapshot HostClock::snapshot() const {
  const std::scoped_lock lock(mutex_);
  return snapshot_;
}

void HostClock::start(std::uint32_t epoch, std::int64_t media_us, std::int64_t host_us) {
  const std::scoped_lock lock(mutex_);
  snapshot_ = {.media_us = media_us, .host_us = host_us, .rate = 1.0, .epoch = epoch, .valid = true};
}

void HostClock::pause(std::int64_t host_us) {
  const std::scoped_lock lock(mutex_);
  if (!snapshot_.valid) return;
  snapshot_ = {.media_us = snapshot_.media_at(host_us),
               .host_us = host_us,
               .rate = 0.0,
               .epoch = snapshot_.epoch,
               .valid = true};
}

void HostClock::flush(std::uint32_t epoch) {
  const std::scoped_lock lock(mutex_);
  snapshot_ = {.epoch = epoch, .valid = false};
}

void MainMail::end_of_stream(std::uint32_t epoch) {
  if (audio_ != nullptr) audio_->post(audio::EndOfStream{.epoch = epoch});
  inbox_.try_push(StreamEnded{.epoch = epoch});
}

void MainMail::failed(const Refusal& refusal) {
  // Un seek refusé laisse la lecture où elle était ; tout autre refus arrête la source.
  report("source", refusal, refusal.code != RefusalCode::SeekFailed);
}

ThreadReport MainMail::reporter(std::string stage) {
  return ThreadReport([this, stage = std::move(stage)](const Refusal& refusal) {
    report(stage, refusal, refusal.code == RefusalCode::ThreadFailed);
  });
}

void MainMail::report(std::string stage, const Refusal& refusal, bool fatal) {
  inbox_.try_push(StageRefusal{.stage = std::move(stage), .refusal = refusal, .fatal = fatal});
}

bool ThreadStages::flush(std::uint32_t epoch) {
  flushed_epoch_ = epoch;
  bool sent = links_.source->try_push(Flush{.epoch = epoch});
  sent = links_.video->post(Flush{.epoch = epoch}) && sent;
  if (links_.audio != nullptr) sent = links_.audio->post(Flush{.epoch = epoch}) && sent;
  if (links_.host_clock != nullptr) links_.host_clock->flush(epoch);
  if (links_.subs) sent = links_.subs->try_push(Flush{.epoch = epoch}) && sent;
  return sent;
}

bool ThreadStages::seek(const cmd::Seek& seek) {
  if (links_.audio_bridge != nullptr) {
    links_.audio_bridge->expect_seek(flushed_epoch_,
                                     seek.exact ? std::optional(seek.target_us) : std::nullopt);
  }
  return links_.source->try_push(Command{seek});
}

bool ThreadStages::play(std::uint32_t epoch, std::int64_t media_us) {
  if (links_.audio != nullptr) return links_.audio->post(Command{cmd::Play{}});
  links_.host_clock->start(epoch, media_us, host_now_us());
  return true;
}

bool ThreadStages::pause() {
  if (links_.audio != nullptr) return links_.audio->post(Command{cmd::Pause{}});
  links_.host_clock->pause(host_now_us());
  return true;
}

bool ThreadStages::request_subtitles(const subs::RenderRequest& request) {
  return links_.subs && links_.subs->try_push(subs::RenderRequest{request});
}

void ThreadStages::quit() {
  links_.source->try_push(Command{cmd::Quit{}});
  links_.video->post(Command{cmd::Quit{}});
  if (links_.audio != nullptr) links_.audio->post(Command{cmd::Quit{}});
  if (links_.subs) links_.subs->try_push(Command{cmd::Quit{}});
}

}  // namespace ajp::app::play
