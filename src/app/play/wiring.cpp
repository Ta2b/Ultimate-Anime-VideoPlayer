#include "app/play/wiring.h"

#include <utility>

namespace ajp::app::play {

ClockSnapshot HostClock::snapshot() const {
  const std::scoped_lock lock(mutex_);
  return snapshot_;
}

void HostClock::start(std::uint32_t epoch, std::int64_t media_us, std::int64_t host_us, bool keep_position) {
  const std::scoped_lock lock(mutex_);
  if (keep_position && snapshot_.valid && snapshot_.epoch == epoch) media_us = snapshot_.media_at(host_us);
  snapshot_ = {.media_us = media_us, .host_us = host_us, .rate = rate_, .epoch = epoch, .valid = true};
}

void HostClock::set_rate(double rate, std::int64_t host_us) {
  const std::scoped_lock lock(mutex_);
  rate_ = rate;
  if (!snapshot_.valid || snapshot_.rate == 0.0) return;  // arrêtée : la vitesse vaut au départ
  snapshot_ = {.media_us = snapshot_.media_at(host_us),
               .host_us = host_us,
               .rate = rate,
               .epoch = snapshot_.epoch,
               .valid = true};
}

void HostClock::adopt(const ClockSnapshot& from, bool running, std::int64_t host_us) {
  const std::scoped_lock lock(mutex_);
  if (!from.valid) {
    snapshot_ = {.epoch = from.epoch, .valid = false};
    return;
  }
  snapshot_ = {.media_us = from.media_at(host_us),
               .host_us = host_us,
               .rate = running ? rate_ : 0.0,
               .epoch = from.epoch,
               .valid = true};
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
  links_.host_clock->flush(epoch);
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

// L'horloge hôte suit play et pause même quand l'audio mène : elle est prête à mener à son tour.
// Elle part avant l'audio, qui s'y cale dès sa reprise quand l'écran mène.
bool ThreadStages::play(std::uint32_t epoch, std::int64_t media_us) {
  playing_ = true;
  const bool host_leads_audio = links_.lead != nullptr && links_.lead->host_leads();
  links_.host_clock->start(epoch, media_us, host_now_us(), host_leads_audio);
  return links_.audio == nullptr || links_.audio->post(Command{cmd::Play{}});
}

bool ThreadStages::pause() {
  playing_ = false;
  links_.host_clock->pause(host_now_us());
  return links_.audio == nullptr || links_.audio->post(Command{cmd::Pause{}});
}

// Vitesse 1 : aucun calage sur l'écran, l'audio mène (s'il y en a). Autre vitesse : l'horloge hôte
// reprend la position de l'audio, à cette vitesse, puis l'audio la suit (audio::SyncTo).
bool ThreadStages::set_speed(double speed) {
  const std::int64_t now = host_now_us();
  const bool host_leads = speed != 1.0;
  if (links_.audio == nullptr || links_.lead == nullptr || links_.lead->host_leads() == host_leads) {
    links_.host_clock->set_rate(speed, now);
    return true;
  }
  if (host_leads) {
    const ClockSnapshot audio = links_.lead->snapshot();  // l'audio, qui mène encore
    links_.host_clock->set_rate(speed, now);
    links_.host_clock->adopt(audio, playing_, now);
    if (!links_.audio->post(audio::SyncTo{.master = links_.host_clock})) return false;
  } else {
    if (!links_.audio->post(audio::SyncTo{})) return false;
    links_.host_clock->set_rate(1.0, now);
  }
  links_.lead->lead_with_host(host_leads);
  return true;
}

bool ThreadStages::request_subtitles(const subs::RenderRequest& request) {
  return links_.subs && links_.subs->try_push(subs::RenderRequest{request});
}

bool ThreadStages::audio(const Command& command) {
  return links_.audio == nullptr || links_.audio->post(Command{command});
}

void ThreadStages::quit() {
  links_.source->try_push(Command{cmd::Quit{}});
  links_.video->post(Command{cmd::Quit{}});
  if (links_.audio != nullptr) links_.audio->post(Command{cmd::Quit{}});
  if (links_.subs) links_.subs->try_push(Command{cmd::Quit{}});
}

}  // namespace ajp::app::play
