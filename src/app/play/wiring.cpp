#include "app/play/wiring.h"

#include <tuple>
#include <utility>

#include "app/play/subtitle_lead.h"
#include "core/log.h"

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
  if (video_ != nullptr) video_->post(StreamEnd{.epoch = epoch});
  if (!inbox_.try_push(StreamEnded{.epoch = epoch}))
    log_channel("app")->warn("end of stream {} lost, inbox full", epoch);
}

void MainMail::failed(const Refusal& refusal) {
  // La source dit elle-même si elle continue (seek refusé, Flush ignoré, suite du fichier
  // illisible traitée comme sa fin) ; tout autre refus l'arrête, et la lecture avec.
  report("source", refusal, !refusal.recoverable);
}

ThreadReport MainMail::reporter(std::string stage) {
  return ThreadReport([this, stage = std::move(stage)](const Refusal& refusal) {
    report(stage, refusal, refusal.code == RefusalCode::ThreadFailed || is_fatal_stage_key(refusal.key));
  });
}

void MainMail::report(std::string stage, const Refusal& refusal, bool fatal) {
  // Un refus fatal a son emplacement réservé (P2.25) : il arrête la lecture même derrière 64 refus.
  if (!inbox_.try_push(StageRefusal{.stage = stage, .refusal = refusal, .fatal = fatal}, fatal))
    log_channel("app")->warn("{}: refusal lost, inbox full: {}", stage, refusal.detail);
}

bool ThreadStages::flush(std::uint32_t epoch) {
  flushed_epoch_ = epoch;
  // L'audio a une nouvelle chance à cette époque, sauf si l'écran mène. C18-08 : rien n'est oublié avant que
  // l'audio n'ait reçu son ordre ; un envoi refusé (boîte pleine) se refait au Flush suivant.
  if (audio_lost_) {
    const bool give_back = !host_must_lead(speed_, false, audio_delay_us_) && links_.audio != nullptr &&
                           links_.lead != nullptr && links_.lead->host_leads();
    if (give_back) {
      if (links_.audio->post(audio::SyncTo{})) {
        links_.host_clock->set_rate(1.0, host_now_us());
        links_.lead->lead_with_host(false);
        audio_lost_ = false;
        master_pending_ = false;
      }
    } else if (!master_pending_ || links_.audio == nullptr ||
               links_.audio->post(audio::SyncTo{.master = links_.host_clock})) {
      audio_lost_ = false;  // l'écran mène : l'audio suit l'horloge hôte, qui le lui a dit
      master_pending_ = false;
    }
  }
  bool sent = links_.source->try_push(Flush{.epoch = epoch});
  sent = links_.video->post(Flush{.epoch = epoch}) && sent;
  if (links_.audio != nullptr) sent = links_.audio->post(Flush{.epoch = epoch}) && sent;
  links_.host_clock->flush(epoch);
  if (links_.subs) sent = links_.subs->try_push(Flush{.epoch = epoch}) && sent;
  return sent;
}

bool ThreadStages::seek(const cmd::Seek& seek) {
  if (links_.subs_lead != nullptr) links_.subs_lead->seek(seek.target_us);
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
  const bool host_leads = host_must_lead(speed, audio_lost_, audio_delay_us_);
  if (links_.audio == nullptr || links_.lead == nullptr || links_.lead->host_leads() == host_leads) {
    links_.host_clock->set_rate(speed, now);
    speed_ = speed;
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
  speed_ = speed;
  return true;
}

bool ThreadStages::audio_lost(std::uint32_t epoch, std::int64_t media_us) {
  if (links_.audio == nullptr || links_.lead == nullptr || links_.lead->host_leads() || !playing_)
    return false;
  links_.host_clock->start(epoch, media_us, host_now_us());
  // L'audio, s'il revient, se cale sur l'horloge hôte (silence ou rognage) au lieu de la mener. L'image ne
  // s'arrête pas avec l'audio : l'hôte mène tout de suite. C18-08 : un ordre refusé se refait au Flush
  // suivant.
  master_pending_ = !links_.audio->post(audio::SyncTo{.master = links_.host_clock});
  links_.lead->lead_with_host(true);
  audio_lost_ = true;
  return true;
}

bool ThreadStages::request_subtitles(const subs::RenderRequest& request) {
  if (links_.subs_lead != nullptr) links_.subs_lead->follow(request.pts_us, request.delay_us);
  return links_.subs && links_.subs->try_push(subs::RenderRequest{request});
}

bool ThreadStages::audio(const Command& command) {
  if (const auto* delay = std::get_if<cmd::SetAudioDelay>(&command)) {
    const bool before = host_must_lead(speed_, audio_lost_, audio_delay_us_);
    audio_delay_us_ = delay->delay_us;
    // Le délai change qui doit mener : le passage se fait comme celui d'un changement de vitesse, avant que
    // le délai n'arrive à l'audio (il se recale alors sur l'horloge qui mène).
    if (host_must_lead(speed_, audio_lost_, audio_delay_us_) != before) set_speed(speed_);
  }
  return links_.audio == nullptr || links_.audio->post(Command{command});
}

void ThreadStages::quit() {
  std::ignore = links_.source->try_push(Command{cmd::Quit{}});  // boîte pleine : l'arrêt du jthread suffit
  links_.video->post(Command{cmd::Quit{}});
  if (links_.audio != nullptr) links_.audio->post(Command{cmd::Quit{}});
  if (links_.subs) std::ignore = links_.subs->try_push(Command{cmd::Quit{}});
}

}  // namespace ajp::app::play
