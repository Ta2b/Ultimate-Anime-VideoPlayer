#include "app/play/session.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <type_traits>
#include <utility>
#include <variant>

#include "app/play/i18n.h"
#include "core/frame.h"
#include "core/pacing.h"
#include "core/vrr_floor.h"

namespace ajp::app::play {
namespace {

// Fin de flux : plus rien n'arrive depuis ce délai, la lecture est finie.
constexpr std::int64_t kEndGraceUs = 1'000'000;
// Un seek dont une boîte a refusé un message est rejoué après ce délai.
constexpr std::int64_t kSeekRetryUs = 50'000;
// Un seek que la source a refusé sans retrouver de position (partage muet, MED-06) : rejoué après ce délai,
// doublé à chaque échec jusqu'à kSeekLostMaxUs, tant que l'époque n'a pas d'image (A13-08).
constexpr std::int64_t kSeekLostRetryUs = 2'000'000;
constexpr std::int64_t kSeekLostMaxUs = 16'000'000;
// Des images attendent l'horloge audio depuis 1,5 s (Session::audio_stall_grace_us_) : l'horloge
// hôte prend la main. Après une première perte, le délai est plus court : la piste est
// probablement illisible.
constexpr std::int64_t kAudioStallRepeatUs = 400'000;
constexpr std::size_t kMaxRefusals = 3;
// L'OSD se recompose quatre fois par seconde, pas à chaque vsync.
constexpr std::int64_t kOsdPeriodUs = 250'000;
// Cadence fixe : marge sous la période de l'écran, pour présenter avant le vsync visé.
constexpr std::int64_t kCadenceSlackUs = 500;
// À l'amorce, la première image attend ses sous-titres au plus ce temps.
constexpr std::int64_t kPrerollSubtitleWaitUs = 100'000;
constexpr std::int64_t kAssumedFramePeriodUs = 41'667;  // une image de la source à 24 i/s, faute de mieux

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

}  // namespace

Session::Session(SessionConfig config, Stages& stages, Screen& screen, const MasterClock& clock)
    : config_(std::move(config)),
      stages_(stages),
      screen_(screen),
      clock_(clock),
      show_stats_(config_.stats) {
  late_.reserve(kMaxRingSlots);
  unwritten_.reserve(kMaxRingSlots);
  unmeasured_.reserve(kMaxRingSlots);
}

void Session::begin(std::int64_t /*host_now*/) {
  if (config_.start_us > 0) seek_to(config_.start_us, true);
}

void Session::handle(MainInbox::Message&& message) {
  std::visit(Overloaded{[this](RingFrame& frame) { on_frame(frame); },
                        [this](subs::SubtitleFrame& frame) { on_subtitles(std::move(frame)); },
                        [this](const StreamEnded& end) { on_stream_ended(end); },
                        [this](const StageRefusal& refusal) { on_refusal(refusal); }},
             message);
}

void Session::command(const Command& command) {
  std::visit(
      Overloaded{[this](const cmd::Play&) { resume(); },
                 [this](const cmd::Pause&) {
                   if (paused_) return;
                   paused_ = true;
                   send_pause();
                 },
                 [this](const cmd::TogglePause&) {
                   if (paused_) {
                     this->command(cmd::Play{});
                   } else {
                     this->command(cmd::Pause{});
                   }
                 },
                 [this](const cmd::Seek& seek) { seek_to(seek.target_us, seek.exact); },
                 [this](const cmd::SeekRelative& seek) {
                   std::int64_t base = 0;
                   if (pending_seek_us_) {
                     base = *pending_seek_us_;
                   } else if (current_ && current_->pts_us != kNoPts) {
                     base = current_->pts_us;
                   }
                   seek_to(base + seek.delta_us, seek.exact);
                 },
                 [this](const cmd::StepFrame& step) { step_frame(step.direction); },
                 [this](const cmd::Quit&) {
                   finished_ = true;
                   quit_ = true;
                 },
                 [this](const cmd::SetVolume& volume) {
                   if (!stages_.audio(volume)) {
                     on_refusal({.stage = "main",
                                 .refusal = {RefusalCode::PacketBacklog, "volume : boîte audio pleine"}});
                   }
                 },
                 [this](const cmd::SetAudioDelay& delay) {
                   if (!stages_.audio(delay)) {
                     on_refusal(
                         {.stage = "main",
                          .refusal = {RefusalCode::PacketBacklog, "délai audio : boîte audio pleine"}});
                   }
                 },
                 [this](const cmd::ToggleOsd&) {
                   show_stats_.all = !show_stats_.all;
                   osd_at_us_.reset();  // relu au tour suivant
                 },
                 [](const auto&) {}},  // le reste ne concerne pas la tranche verticale
      command);
}

void Session::resume() {
  if (!paused_) return;
  paused_ = false;
  // A13-07 : des pas pris dans la file n'ont pas déplacé l'audio, resté à la pause. La lecture repart par un
  // seek exact sur l'image montrée, comme après un pas en arrière.
  if (std::exchange(stepped_in_queue_, false) && !prerolling_ && current_ && current_->pts_us != kNoPts) {
    seek_to(current_->pts_us, true);
    return;
  }
  if (!prerolling_ && !start_pts_ && current_) send_play(epoch_, current_->pts_us);
}

// Image par image : en pause, un seek exact d'une image de la source. Le seek part de la position visée, donc
// des pressions rapprochées avancent d'autant. P3.42 : vers l'avant, l'image suivante déjà dans la file, sans
// seek.
void Session::step_frame(int direction) {
  if (!paused_) command(cmd::Pause{});
  if (direction > 0 && step_from_queue()) return;
  std::int64_t period = config_.extra != nullptr ? config_.extra->source_period_us() : 0;
  if (period <= 0) period = kAssumedFramePeriodUs;
  // P3.78 : la cible est le milieu de l'image voisine, pas son début : des pts arrondis à la microseconde
  // (23,976 i/s) ne font plus tomber le seek exact deux images plus loin. Une cible en attente est déjà un
  // milieu.
  std::int64_t base = 0;
  if (pending_seek_us_) {
    base = *pending_seek_us_ - period / 2;
  } else if (current_ && current_->pts_us != kNoPts) {
    base = current_->pts_us;
    // A13-05 : le pas d'avant est retombé dans l'image montrée (elle dure plus que la période moyenne, en
    // VFR) ; le suivant part de sa cible, sans quoi rien ne bougerait jamais.
    if (direction > 0 && last_step_ && last_step_->from_pts_us == base)
      base = std::max(base, last_step_->target_us - period / 2);
  }
  const std::int64_t target = std::max<std::int64_t>(base + direction * period + period / 2, 0);
  const std::optional<std::int64_t> from =
      current_ ? std::optional(current_->pts_us) : std::optional<std::int64_t>();
  seek_to(target, true);
  if (direction > 0 && from) last_step_ = StepMark{*from, target};
}

void Session::set_subtitle_delay(std::int64_t delay_us) {
  if (delay_us == config_.subtitle_delay_us) return;
  config_.subtitle_delay_us = delay_us;
  if (current_) request_subtitles(*current_);  // comme la taille : l'image à l'écran redemande les siens
}

void Session::set_subtitle_scale(float scale) {
  if (scale == config_.subtitle_scale) return;
  config_.subtitle_scale = scale;
  // Les réponses gardées sont à l'ancienne taille : l'image à l'écran redemande les siennes (en
  // pause aussi), les suivantes arrivent à la nouvelle.
  if (current_) request_subtitles(*current_);
}

void Session::set_stats(const StatsSettings& stats) {
  show_stats_ = stats;
  osd_at_us_.reset();
}

void Session::set_tracks(std::vector<std::string> tracks) {
  config_.tracks = std::move(tracks);
  osd_at_us_.reset();
}

void Session::set_header(std::vector<std::string> header) {
  config_.header = std::move(header);
  osd_at_us_.reset();
}

void Session::set_output(std::string output) {
  config_.output = std::move(output);
  osd_at_us_.reset();
}

void Session::set_subtitles(bool on) {
  config_.subtitles = on;
  if (!on) subtitles_.clear();
}

void Session::toggle_fullscreen() {
  screen_.set_fullscreen(!screen_.fullscreen());
}

void Session::seek_to(std::int64_t target_us, bool exact) {
  target_us = std::max<std::int64_t>(target_us, 0);
  if (config_.duration_us > 0) target_us = std::min(target_us, config_.duration_us);
  const bool retrying = retry_seek_.has_value();
  retry_seek_.reset();
  clock_stalled_since_us_.reset();
  // Un pas garde sa marque jusqu'à sa première image (A13-05) ; tout autre seek l'oublie.
  last_step_.reset();
  stepped_in_queue_ = false;
  if (seek_lost_at_us_) {
    seek_lost_at_us_.reset();
  } else {
    seek_lost_wait_us_ = kSeekLostRetryUs;
  }
  send_pause();
  screen_.new_scene();
  ++epoch_;
  read_lost_.reset();  // un seek relit le fichier (la source repart, media/source.h)
  start_pts_.reset();
  // Le Seek part même si une boîte a refusé son Flush : la source qui a reçu le sien attend ce
  // Seek pour reprendre (media/source.h). La diffusion se rejoue ensuite, à une époque neuve,
  // tant qu'une boîte a refusé quelque chose.
  const bool flushed = stages_.flush(epoch_);
  const bool sought = stages_.seek(cmd::Seek{.target_us = target_us, .exact = exact});
  if (!flushed || !sought) {
    retry_seek_ = SeekRetry{.target_us = target_us, .exact = exact, .at_us = last_tick_us_ + kSeekRetryUs};
    if (!retrying) {
      on_refusal({.stage = "main",
                  .refusal = {RefusalCode::SeekFailed,
                              std::format("seek vers {} : boîte pleine", format_position(target_us))}});
    }
  }
  for (const RingFrame& frame : queue_.take_stale(epoch_)) {
    retire(frame);
    ++stats_.stale;
  }
  // Rendues, leurs slots repartent : leur écriture ne se mesure plus.
  std::erase_if(unmeasured_, [this](const RingFrame& frame) { return frame.epoch < epoch_; });
  stats_.restart_pacing();
  subtitles_.clear();
  subtitle_requests_ = 0;
  subtitle_waiting_.reset();
  subtitle_pts_.reset();
  held_subtitle_pts_.reset();
  prerolling_ = true;
  preroll_since_us_.reset();
  stream_ended_ = false;
  pending_seek_us_ = target_us;
}

bool Session::step_from_queue() {
  if (pending_seek_us_ || prerolling_ || !current_ || current_->pts_us == kNoPts) return false;
  std::int64_t period = config_.extra != nullptr ? config_.extra->source_period_us() : 0;
  if (period <= 0) period = kAssumedFramePeriodUs;
  const RingFrame shown = *current_;
  // L'image une période source plus loin (les images interpolées d'entre deux sont passées) ; écrite.
  const RingFrame* head = queue_.first_of(epoch_);
  if (head == nullptr || head->pts_us == kNoPts || head->pts_us > shown.pts_us + period + period / 4)
    return false;
  late_.clear();
  const std::optional<RingFrame> next = queue_.pick(shown.pts_us + period, period / 4, late_,
                                                    [this](const RingFrame& f) { return screen_.ready(f); });
  for (const RingFrame& frame : late_) retire(frame);
  late_.clear();
  if (!next) return false;
  retire(shown);
  current_ = next;
  stepped_ = true;           // montrée au prochain tour
  stepped_in_queue_ = true;  // l'audio est resté à la pause : la reprise le recale (A13-07)
  return true;
}

void Session::on_frame(const RingFrame& frame) {
  if (frame.epoch < epoch_) {  // d'avant un seek : jamais montrée
    retire(frame);
    ++stats_.stale;
    return;
  }
  // Une image reçue tient son slot jusqu'à être écrite : jamais plus de kMaxRingSlots en attente.
  if (config_.latency != nullptr && frame.submitted_us > 0 && unmeasured_.size() < kMaxRingSlots)
    unmeasured_.push_back(frame);
  seek_lost_at_us_.reset();  // la source a retrouvé une position : rien à rejouer
  screen_.watch(frame);
  queue_.push(frame);
  newest_ = frame;
  request_subtitles(frame);
}

void Session::on_subtitles(subs::SubtitleFrame&& frame) {
  if (frame.epoch != epoch_) return;  // demande d'avant un seek : l'atlas rentre chez subs
  if (subtitle_requests_ > 0) --subtitle_requests_;
  if (subtitle_waiting_) {
    const RingFrame waiting = *subtitle_waiting_;
    subtitle_waiting_.reset();
    request_subtitles(waiting);
  }
  if (!config_.subtitles) return;  // désactivés depuis la demande
  subtitles_.push(std::move(frame), current_ ? current_->pts_us : kNoPts);
}

void Session::on_refusal(const StageRefusal& refusal) {
  refusals_.push_back(
      std::format("{} : {} {}", refusal.stage, name(refusal.refusal.code), refusal.refusal.detail));
  if (refusals_.size() > kMaxRefusals) refusals_.erase(refusals_.begin());
  osd_at_us_.reset();  // le refus paraît au tour suivant
  if (refusal.refusal.code == RefusalCode::SourceResized) resized_ = true;
  if (refusal.fatal && !resized_ && !failure_)
    failure_ = Refusal{refusal.refusal.code, std::format("{} : {}", refusal.stage, refusal.refusal.detail)};
  if (refusal.fatal || resized_) finished_ = true;
  if (!refusal.fatal && refusal.refusal.recoverable && refusal.refusal.code != RefusalCode::FieldOutOfRange)
    notice_ = refusal.refusal;
  // La source rend une fin de flux après ce refus (media/source.h) : retenu pour on_stream_ended.
  if (!refusal.fatal && refusal.refusal.recoverable && refusal.stage == "source" &&
      (refusal.refusal.code == RefusalCode::DecodeFailed || refusal.refusal.code == RefusalCode::IoFailed)) {
    read_lost_ = Refusal{refusal.refusal.code, std::format("{} : {}", refusal.stage, refusal.refusal.detail)};
  }
  // La source n'a pas retrouvé de position après un seek refusé : rien ne viendra avant un autre seek. Il est
  // rejoué au tour où son délai passe (A13-08).
  if (refusal.stage == "source" && refusal.refusal.code == RefusalCode::SeekFailed && prerolling_ &&
      pending_seek_us_ && queue_.first_of(epoch_) == nullptr)
    seek_lost_at_us_ = last_tick_us_ + seek_lost_wait_us_;
}

void Session::send_play(std::uint32_t epoch, std::int64_t media_us) {
  clock_retry_ =
      stages_.play(epoch, media_us) ? std::nullopt : std::optional(ClockRetry{true, epoch, media_us});
}

void Session::send_pause() {
  clock_retry_ = stages_.pause() ? std::nullopt : std::optional(ClockRetry{false, 0, 0});
}

void Session::retry_pending(std::int64_t host_now) {
  if (retry_seek_ && host_now >= retry_seek_->at_us) seek_to(retry_seek_->target_us, retry_seek_->exact);
  if (seek_lost_at_us_ && host_now >= *seek_lost_at_us_ && prerolling_ && pending_seek_us_ &&
      queue_.first_of(epoch_) == nullptr) {
    seek_lost_wait_us_ = std::min(seek_lost_wait_us_ * 2, kSeekLostMaxUs);
    seek_to(*pending_seek_us_, true);  // garde le délai doublé : seek_lost_at_us_ est encore posé
  }
  retry_clock();
}

// A13-06 : un Play ou une Pause refusés (boîte audio pleine) sont redits à chaque tour, tant qu'ils valent
// encore. Perdue, une pause laissait courir l'horloge audio sous une barre « en pause ».
void Session::retry_clock() {
  if (!clock_retry_) return;
  const ClockRetry retry = *clock_retry_;
  if (retry.play && (paused_ || retry.epoch != epoch_)) {
    clock_retry_.reset();
  } else if (retry.play) {
    send_play(retry.epoch, retry.media_us);
  } else {
    send_pause();
  }
}

void Session::request_subtitles(const RingFrame& frame) {
  if (!config_.subtitles || frame.pts_us == kNoPts) return;
  if (subtitle_requests_ >= kMaxSubtitleRequests) {
    subtitle_waiting_ = frame;
    return;
  }
  const OutputSize size = screen_.size();
  if (size.width <= 0 || size.height <= 0) return;  // fenêtre réduite (P2.18) : rien à placer, rien de refusé
  subtitle_pts_ = frame.pts_us;
  const subs::RenderRequest request{.pts_us = frame.pts_us,
                                    .epoch = frame.epoch,
                                    .delay_us = config_.subtitle_delay_us,
                                    .target = {.width = size.width,
                                               .height = size.height,
                                               .video_width = config_.video_width,
                                               .video_height = config_.video_height,
                                               .sample_aspect = config_.video_sar,
                                               .video_matrix = config_.video_matrix,
                                               .font_scale = config_.subtitle_scale}};
  if (stages_.request_subtitles(request)) ++subtitle_requests_;
}

void Session::retire(const RingFrame& frame) {
  // Pas encore écrite (seek, image d'avant un seek) : la rendre maintenant ferait attendre son
  // écriture à la file de l'affichage. Elle part au premier tour où elle l'est.
  if (!screen_.ready(frame)) {
    unwritten_.push_back(frame);
    return;
  }
  stages_.slot_free(screen_.retire(frame));
}

void Session::retire_written() {
  std::erase_if(unwritten_, [this](const RingFrame& frame) {
    if (!screen_.ready(frame)) return false;
    stages_.slot_free(screen_.retire(frame));
    return true;
  });
}

// Avant tout retour de slot du tour : un slot rendu puis réécrit paraîtrait écrit bien plus tard.
void Session::measure_latency(std::int64_t host_now) {
  if (config_.latency == nullptr) return;
  std::erase_if(unmeasured_, [&](const RingFrame& frame) {
    if (!screen_.ready(frame)) return false;
    // L'heure où le veilleur l'a vue écrite, sinon celle de ce tour (au plus une attente de main plus tard).
    const std::int64_t written = screen_.written_at(frame).value_or(host_now);
    config_.latency->observe(written - frame.submitted_us);
    if (config_.on_written) config_.on_written(frame, written);
    return true;
  });
}

void Session::choose(std::int64_t host_now) {
  if (prerolling_) {
    const RingFrame* head = queue_.first_of(epoch_);
    // Pas encore là, ou pas encore écrite : l'image d'avant le seek reste à l'écran.
    if (head == nullptr || !screen_.ready(*head)) return;
    // L'horloge ne court pas encore : la première image attend ses sous-titres (au plus
    // kPrerollSubtitleWaitUs), pour ne pas être montrée sans eux puis recomposée.
    if (!preroll_since_us_) preroll_since_us_ = host_now;
    if (config_.subtitles && head->pts_us != kNoPts && subtitle_requests_ > 0 &&
        subtitles_.for_pts(head->pts_us) == nullptr &&
        host_now - *preroll_since_us_ < kPrerollSubtitleWaitUs) {
      return;
    }
    preroll_since_us_.reset();
    const std::optional<RingFrame> first = queue_.pop_front();
    if (!first) return;
    if (current_) retire(*current_);
    current_ = first;
    fresh_ = true;
    prerolling_ = false;
    pending_seek_us_.reset();
    last_fresh_host_us_ = host_now;
    start_pts_ = first->pts_us;  // l'horloge part quand elle est rendue (shown_at)
    return;
  }
  const ClockSnapshot now = clock_.snapshot();
  watch_clock(now, host_now);
  if (!now.valid || now.epoch != epoch_) return;  // horloge pas encore partie : l'image reste
  // L'image choisie maintenant paraît après display_delay_us ; elle doit être celle dont le pts
  // est le plus proche de cet instant, à une demi-période près.
  const std::int64_t period = display_period();
  const std::int64_t lead_us = display_delay_us(period) + period / 2;
  late_.clear();
  // Seules les images déjà écrites : l'écran garde la sienne tant que la suivante se calcule, au
  // lieu d'attendre le GPU des moteurs à chaque présentation.
  const std::optional<RingFrame> picked = pick(now, host_now, lead_us);
  for (const RingFrame& frame : late_) {
    retire(frame);
    ++stats_.dropped;
  }
  if (!picked) return;
  if (current_) retire(*current_);
  current_ = picked;
  fresh_ = true;
  last_fresh_host_us_ = host_now;
}

std::optional<RingFrame> Session::pick(const ClockSnapshot& now, std::int64_t host_now,
                                       std::int64_t lead_us) {
  const auto ready = [this](const RingFrame& frame) { return screen_.ready(frame); };
  // L'instant hôte où le pts d'une image tombe sur l'horloge.
  const auto host_of = [&](const RingFrame& frame) {
    return host_now +
           static_cast<std::int64_t>(std::llround(static_cast<double>(frame.pts_us - now.media_at(host_now)) /
                                                  std::max(now.rate, 1e-6)));
  };
  std::optional<RingFrame> picked;
  const std::optional<core::VblankGrid> grid = config_.vsync_index ? screen_.vblank_grid() : std::nullopt;
  if (grid && grid->period_us > 0.0 && now.rate > 0.0) {
    // Le vblank où paraîtra l'image présentée maintenant ; l'image retenue est la dernière dont le vblank
    // (arrondi avec hystérésis près du milieu) n'est pas après lui.
    const double next =
        std::ceil(grid->index_of(static_cast<double>(host_now + display_delay_us(display_period()))));
    picked = queue_.pick_by(
        [&](const RingFrame& frame) {
          return static_cast<double>(
                     vsync_picker_.pick(grid->index_of(static_cast<double>(host_of(frame))))) <= next;
        },
        late_, ready);
    // La position de l'image neuve sur la grille mesurée règle la vitesse calée (core/phase_trim.h) ; la
    // nouvelle vitesse part au tour suivant (follow_display).
    if (speed_locked_ && picked && picked->pts_us != kNoPts)
      phase_trim_.note(grid->index_of(static_cast<double>(host_of(*picked))), host_now);
  } else {
    picked = queue_.pick(now.media_at(host_now), lead_us, late_, ready);
  }
  target_host_us_ =
      config_.vsync_index && picked && picked->pts_us != kNoPts && now.rate > 0.0 ? host_of(*picked) : 0;
  return picked;
}

// Des images attendent une horloge qui ne court pas : l'audio maître n'en fournit plus (piste
// illisible, trou, début tardif, fin avant la vidéo). Passé le délai, l'horloge hôte mène et la
// vidéo repart ; sans cela elle resterait figée, la source bloquée sur l'image tenue.
void Session::watch_clock(const ClockSnapshot& now, std::int64_t host_now) {
  const bool running = now.valid && now.epoch == epoch_ && now.rate != 0.0;
  if (running || paused_ || current_ == std::nullopt || queue_.first_of(epoch_) == nullptr) {
    clock_stalled_since_us_.reset();
    return;
  }
  if (!clock_stalled_since_us_) {
    clock_stalled_since_us_ = host_now;
    return;
  }
  // P3.75 : passé la fin connue de la piste audio, l'horloge hôte mène tout de suite et sans message : ce
  // n'est pas une panne, la vidéo continue seule.
  const bool audio_over = config_.audio_end_us > 0 && current_->pts_us >= config_.audio_end_us - 200'000;
  if (host_now - *clock_stalled_since_us_ < (audio_over ? 0 : audio_stall_grace_us_)) return;
  clock_stalled_since_us_.reset();
  if (!stages_.audio_lost(epoch_, current_->pts_us) || audio_over) return;
  audio_stall_grace_us_ = kAudioStallRepeatUs;  // l'audio a déjà manqué : la prochaine fois, plus vite
  on_refusal({.stage = "main",
              .refusal = {RefusalCode::DeviceUnavailable, i18n::tr("audio missing: the display clock leads")},
              .fatal = false});
}

Session::SubtitleLook Session::look_of(const subs::SubtitleFrame* frame) {
  if (frame == nullptr || !frame->atlas || frame->rects.empty()) return {};
  return SubtitleLook{.generation = frame->atlas->generation, .rects = frame->rects};
}

const subs::SubtitleFrame* Session::subtitles_for(std::int64_t pts_us) {
  const subs::SubtitleFrame* frame = subtitles_.for_pts(pts_us);
  // Une image montrée sans réponse alors que d'autres attendent : le défaut du clignotement.
  if (frame == nullptr && subtitles_.size() > 0) ++stats_.subtitle_gaps;
  return frame;
}

// Une image tenue plus d'une période source : les demandes de sous-titres continuent sur la grille des
// sources, et la lecture suit l'horloge (jamais au-delà de la dernière demande).
void Session::follow_subtitles(std::int64_t host_now) {
  if (fresh_) held_subtitle_pts_.reset();
  if (!config_.subtitles || !current_ || !subtitle_pts_ || config_.extra == nullptr) return;
  const std::int64_t period = config_.extra->source_period_us();
  if (period <= 0) return;
  const ClockSnapshot now = clock_.snapshot();
  if (!now.valid) return;
  const std::int64_t media = now.media_at(host_now);
  int asked = 0;
  const RingFrame shown = *current_;  // request_subtitles ne change pas l'image montrée
  while (subtitle_requests_ < kMaxSubtitleRequests && asked < 4) {
    const std::optional<std::int64_t> last = subtitle_pts_;
    if (!last || media - *last < period) break;
    RingFrame next = shown;
    next.pts_us = *last + period;
    request_subtitles(next);
    if (subtitle_pts_ != next.pts_us) break;  // refusée : on réessaie au tour suivant
    held_subtitle_pts_ = std::max(shown.pts_us, std::min(media, next.pts_us));
    ++asked;
  }
}

std::expected<std::optional<PresentRecord>, Refusal> Session::tick(std::int64_t host_now, const UiFrame& ui) {
  if (auto lost = screen_.device_lost()) return std::unexpected(std::move(*lost));  // relance, pas de gel
  fresh_ = std::exchange(stepped_, false);
  last_tick_us_ = host_now;
  retry_pending(host_now);
  measure_latency(host_now);
  retire_written();
  follow_display();
  choose(host_now);
  follow_subtitles(host_now);
  const bool fresh = fresh_;
  bool osd_changed = false;
  if (!osd_at_us_ || host_now - *osd_at_us_ >= kOsdPeriodUs) {
    std::vector<std::string> lines = osd_lines();
    osd_changed = lines != osd_;
    osd_ = std::move(lines);
    osd_at_us_ = host_now;
  }
  const subs::SubtitleFrame* subtitles =
      current_ ? subtitles_for(held_subtitle_pts_.value_or(current_->pts_us)) : nullptr;
  SubtitleLook look = look_of(subtitles);
  const OutputSize size = screen_.size();
  // Présentation à la demande : rien n'a changé, l'écran garde l'image (et en VRR, il suit le
  // rythme des images au lieu de recomposer à chaque vsync, ce qui laisse le GPU aux moteurs).
  const bool requested = std::exchange(present_requested_, false);
  const bool ui_changed = overlay_due(host_now, ui, osd_changed) || requested;
  std::string why;
  if (fresh) why += 'i';
  if (look != shown_subtitles_) why += 's';
  if (osd_changed) why += 'o';
  if (ui_changed) why += 'u';
  if (size != shown_size_) why += 't';
  // Cadence fixe : la période de l'écran écoulée (à une demi-milliseconde près, pour ne pas
  // manquer le vsync), l'image est représentée telle quelle.
  const bool cadence = config_.steady_cadence && config_.display_period_us > 0 && shown_once_ &&
                       host_now - shown_at_us_ >= config_.display_period_us - kCadenceSlackUs;
  if (cadence && why.empty()) why += 'c';
  const std::int64_t floor_step = floor_step_us_now(host_now);
  const bool floor_due = floor_step > 0 && host_now - shown_at_us_ >= floor_step - kCadenceSlackUs;
  if (floor_due && why.empty()) why += 'f';
  if (shown_once_ && !fresh && !osd_changed && !ui_changed && look == shown_subtitles_ &&
      size == shown_size_ && !cadence && !floor_due) {
    check_end(host_now);
    return std::optional<PresentRecord>{};
  }
  const ShowRequest request{.frame = current_ ? &*current_ : nullptr,
                            .subtitles = subtitles,
                            .osd = osd_,
                            .ui = ui.draw,
                            .ui_bottom = ui.draw != nullptr ? ui.bottom : 0,
                            .ui_top = ui.draw != nullptr ? ui.top : 0,
                            .redraws_expected = config_.steady_cadence || config_.vrr_floor_us > 0 ||
                                                ui.draw != nullptr || !osd_.empty(),
                            .target_us = fresh_ ? target_host_us_ : 0};
  auto timing = screen_.show(request);
  if (!timing) return std::unexpected(std::move(timing.error()));

  // Présentation refusée (fenêtre cachée, swapchain à refaire) : l'image n'est pas à l'écran, elle se
  // représente au tour suivant au lieu d'être tenue pour affichée.
  if (!timing->presented) present_requested_ = true;
  shown_at(timing->presented ? timing->present_us : host_now);
  shown_ui_ = ui.draw != nullptr;
  shown_subtitles_ = std::move(look);
  shown_size_ = size;
  std::optional<PresentRecord> record;
  if (timing->presented) {
    const std::int64_t period = display_period();
    note_queue_wait(timing->queue_wait_us, period);
    const std::int64_t previous = stats_.last_new_present_us;
    stats_.presented(timing->render_us, timing->present_us, fresh);
    if (fresh && current_) pace(*current_, host_now + display_delay_us(period));
    record = PresentRecord{.seq = ++presented_seq_,
                           .host_us = timing->present_us,
                           .pts_us = current_ ? current_->pts_us : kNoPts,
                           .fresh = fresh,
                           .acquire_us = timing->acquire_us,
                           .render_us = timing->render_us,
                           .queue_wait_us = timing->queue_wait_us,
                           .vsync_us = timing->vsync_us,
                           .interval_us = fresh && previous != 0 ? timing->present_us - previous : 0,
                           .dropped = stats_.dropped,
                           .why = std::move(why),
                           .missing = stats_.missing,
                           .lag_us = fresh && stats_.lag.count() > 0 ? stats_.lag.latest() : 0,
                           .uninterpolated = stats_.uninterpolated,
                           .display_gpu_us = timing->display_gpu_us,
                           .present_id = timing->present_id,
                           .feedback_id = timing->feedback_id,
                           .target_us = timing->target_us,
                           .visible_us = timing->visible_us,
                           .vsync_k = timing->vsync_k,
                           .level = std::string(name(timing->level))};
  }
  check_end(host_now);
  return record;
}

std::int64_t Session::floor_step_us_now(std::int64_t host_now) const {
  if (config_.vrr_floor_us <= 0 || !shown_once_ || !current_ || prerolling_) return 0;
  const ClockSnapshot now = clock_.snapshot();
  if (!now.valid || now.epoch != epoch_ || now.rate <= 0.0) return 0;
  // La tenue attendue : le temps déjà passé, plus ce qui reste avant l'image suivante (inconnu : le
  // plancher).
  std::int64_t hold = host_now - shown_at_us_ + config_.vrr_floor_us;
  if (const RingFrame* head = queue_.first_of(epoch_); head != nullptr && head->pts_us != kNoPts) {
    const std::int64_t period = display_period();
    const std::int64_t lead_us = display_delay_us(period) + period / 2;
    const auto media_wait = static_cast<double>(head->pts_us - lead_us - now.media_at(host_now));
    hold = host_now - shown_at_us_ + std::max<std::int64_t>(std::llround(media_wait / now.rate), 0);
  }
  return core::floor_step_us(hold, config_.vrr_floor_us);
}

std::int64_t Session::idle_wait_us(std::int64_t host_now) {
  constexpr std::int64_t kPollUs = 1'000;
  constexpr std::int64_t kIdleUs = 4'000;
  if (prerolling_) return kPollUs;
  const RingFrame* head = queue_.first_of(epoch_);
  if (head == nullptr) return kIdleUs;
  // Pas encore écrite : le veilleur réveille main à son écriture, sinon main la sonde.
  if (!screen_.ready(*head)) return screen_.wakes_on_write() ? kIdleUs : kPollUs;
  const ClockSnapshot now = clock_.snapshot();
  if (!now.valid || now.epoch != epoch_ || now.rate <= 0.0 || head->pts_us == kNoPts) return kPollUs;
  const std::int64_t period = display_period();
  const std::int64_t lead_us = display_delay_us(period) + period / 2;
  const auto media_wait = static_cast<double>(head->pts_us - lead_us - now.media_at(host_now));
  std::int64_t wait = std::llround(media_wait / now.rate) - kCadenceSlackUs;
  // Cadence fixe : l'image se représente à chaque période de l'écran.
  if (config_.steady_cadence && config_.display_period_us > 0 && shown_once_)
    wait = std::min(wait, shown_at_us_ + config_.display_period_us - kCadenceSlackUs - host_now);
  if (const std::int64_t step = floor_step_us_now(host_now); step > 0)
    wait = std::min(wait, shown_at_us_ + step - kCadenceSlackUs - host_now);
  return std::clamp(wait, kPollUs, kIdleUs);
}

std::int64_t Session::pacing_period_us() const {
  // Un écran plus lent que la sortie saute des images par construction : sa période compte.
  const std::int64_t nominal = config_.extra != nullptr ? config_.extra->output_period_us() : 0;
  return nominal > 0 ? std::max(nominal, display_period()) : 0;
}

void Session::pace(const RingFrame& frame, std::int64_t visible_us) {
  // Le retard ne se lit que pendant la lecture : en pause ou avant le départ, l'horloge est figée.
  const ClockSnapshot now = clock_.snapshot();
  const bool running = !paused_ && now.valid && now.epoch == epoch_ && frame.pts_us != kNoPts;
  stats_.paced(frame.pts_us, running ? now.media_at(visible_us) - frame.pts_us : 0, pacing_period_us(),
               frame.uninterpolated);
}

void Session::check_end(std::int64_t host_now) {
  if (playing_since_us_) {
    const std::int64_t played = host_now - *playing_since_us_;
    // --seconds atteint : un arrêt demandé, sans épisode suivant (A13-14).
    if (config_.stop_after_us > 0 && played >= config_.stop_after_us) finished_ = quit_ = true;
    if (config_.fullscreen_after_us > 0 && !fullscreen_done_ && played >= config_.fullscreen_after_us) {
      screen_.set_fullscreen(true);
      fullscreen_done_ = true;
    }
  }
  if (stream_ended_ && !paused_ && queue_.empty() && stages_.idle() &&
      host_now - last_fresh_host_us_ > kEndGraceUs) {
    finished_ = true;
  }
}

void Session::end() {
  // À l'arrêt, l'écran peut attendre les écritures : tout est rendu tout de suite.
  for (const RingFrame& frame : queue_.take_all()) stages_.slot_free(screen_.retire(frame));
  for (const RingFrame& frame : unwritten_) stages_.slot_free(screen_.retire(frame));
  unwritten_.clear();
  unmeasured_.clear();
  if (current_) stages_.slot_free(screen_.retire(*current_));
  current_.reset();
  subtitles_.clear();
  stages_.quit();
}

}  // namespace ajp::app::play
