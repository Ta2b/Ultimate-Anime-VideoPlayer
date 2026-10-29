#include "app/play/session.h"

#include <algorithm>
#include <format>
#include <type_traits>
#include <utility>
#include <variant>

#include "app/play/i18n.h"
#include "core/frame.h"
#include "core/pacing.h"

namespace ajp::app::play {
namespace {

// Fin de flux : plus rien n'arrive depuis ce délai, la lecture est finie.
constexpr std::int64_t kEndGraceUs = 1'000'000;
// Un seek dont une boîte a refusé un message est rejoué après ce délai.
constexpr std::int64_t kSeekRetryUs = 50'000;
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
// Une présentation qui attend plus d'un quart de période a trouvé la file de l'écran pleine ; la
// part de ces attentes se lisse sur une dizaine de présentations, et au-delà de la moitié la file
// est tenue pour pleine.
constexpr double kQueueFillSmoothing = 0.1;
// Période d'écran supposée tant qu'aucune présentation rapprochée ne la borne : 60 Hz.
constexpr std::int64_t kAssumedPeriodUs = 16'667;
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
                        [this](const StreamEnded& end) {
                          if (end.epoch == epoch_) stream_ended_ = true;
                        },
                        [this](const StageRefusal& refusal) { on_refusal(refusal); }},
             message);
}

void Session::command(const Command& command) {
  std::visit(
      Overloaded{
          [this](const cmd::Play&) {
            if (!paused_) return;
            paused_ = false;
            if (!prerolling_ && current_) stages_.play(epoch_, current_->pts_us);
          },
          [this](const cmd::Pause&) {
            if (paused_) return;
            paused_ = true;
            stages_.pause();
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
          [this](const cmd::StepFrame& step) {
            // Image par image : en pause, un seek exact d'une image de la source. Le seek
            // part de la position visée, donc des pressions rapprochées avancent d'autant.
            if (!paused_) this->command(cmd::Pause{});
            std::int64_t base = 0;
            if (pending_seek_us_) {
              base = *pending_seek_us_;
            } else if (current_ && current_->pts_us != kNoPts) {
              base = current_->pts_us;
            }
            std::int64_t period = config_.extra != nullptr ? config_.extra->source_period_us() : 0;
            if (period <= 0) period = kAssumedFramePeriodUs;
            seek_to(std::max<std::int64_t>(base + step.direction * period, 0), true);
          },
          [this](const cmd::Quit&) {
            finished_ = true;
            quit_ = true;
          },
          [this](const cmd::SetVolume& volume) {
            if (!stages_.audio(volume)) {
              on_refusal(
                  {.stage = "main", .refusal = {RefusalCode::PacketBacklog, "volume : boîte audio pleine"}});
            }
          },
          [this](const cmd::SetListening& listening) {
            if (!stages_.audio(listening)) {
              on_refusal(
                  {.stage = "main", .refusal = {RefusalCode::PacketBacklog, "écoute : boîte audio pleine"}});
            }
          },
          [this](const cmd::SetAudioDelay& delay) {
            if (!stages_.audio(delay)) {
              on_refusal({.stage = "main",
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
  stages_.pause();
  screen_.new_scene();
  ++epoch_;
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
  prerolling_ = true;
  preroll_since_us_.reset();
  stream_ended_ = false;
  pending_seek_us_ = target_us;
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
  queue_.push(frame);
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
  if (refusal.fatal) finished_ = true;
}

void Session::request_subtitles(const RingFrame& frame) {
  if (!config_.subtitles || frame.pts_us == kNoPts) return;
  if (subtitle_requests_ >= kMaxSubtitleRequests) {
    subtitle_waiting_ = frame;
    return;
  }
  const OutputSize size = screen_.size();
  const subs::RenderRequest request{.pts_us = frame.pts_us,
                                    .epoch = frame.epoch,
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
    config_.latency->observe(host_now - frame.submitted_us);
    if (config_.on_written) config_.on_written(frame, host_now);
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
    if (!paused_) stages_.play(epoch_, first->pts_us);
    if (!playing_since_us_) playing_since_us_ = host_now;
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
  const std::optional<RingFrame> picked =
      queue_.pick(now.media_at(host_now), lead_us, late_,
                  [this](const RingFrame& frame) { return screen_.ready(frame); });
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
  if (host_now - *clock_stalled_since_us_ < audio_stall_grace_us_) return;
  clock_stalled_since_us_.reset();
  if (!stages_.audio_lost(epoch_, current_->pts_us)) return;
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

std::expected<std::optional<PresentRecord>, Refusal> Session::tick(std::int64_t host_now, const UiFrame& ui) {
  fresh_ = false;
  last_tick_us_ = host_now;
  if (retry_seek_ && host_now >= retry_seek_->at_us) seek_to(retry_seek_->target_us, retry_seek_->exact);
  measure_latency(host_now);
  retire_written();
  follow_display();
  choose(host_now);
  const bool fresh = fresh_;
  bool osd_changed = false;
  if (!osd_at_us_ || host_now - *osd_at_us_ >= kOsdPeriodUs) {
    std::vector<std::string> lines = osd_lines();
    osd_changed = lines != osd_;
    osd_ = std::move(lines);
    osd_at_us_ = host_now;
  }
  const subs::SubtitleFrame* subtitles = current_ ? subtitles_for(current_->pts_us) : nullptr;
  SubtitleLook look = look_of(subtitles);
  const OutputSize size = screen_.size();
  // Présentation à la demande : rien n'a changé, l'écran garde l'image (et en VRR, il suit le
  // rythme des images au lieu de recomposer à chaque vsync, ce qui laisse le GPU aux moteurs).
  const bool ui_changed = ui.changed || (ui.draw != nullptr) != shown_ui_;
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
  if (shown_once_ && !fresh && !osd_changed && !ui_changed && look == shown_subtitles_ &&
      size == shown_size_ && !cadence) {
    check_end(host_now);
    return std::optional<PresentRecord>{};
  }
  const ShowRequest request{.frame = current_ ? &*current_ : nullptr,
                            .subtitles = subtitles,
                            .osd = osd_,
                            .ui = ui.draw,
                            .ui_bottom = ui.draw != nullptr ? ui.bottom : 0,
                            .ui_top = ui.draw != nullptr ? ui.top : 0};
  auto timing = screen_.show(request);
  if (!timing) return std::unexpected(std::move(timing.error()));

  shown_once_ = true;
  shown_at_us_ = timing->presented ? timing->present_us : host_now;
  shown_ui_ = ui.draw != nullptr;
  shown_subtitles_ = std::move(look);
  shown_size_ = size;
  std::optional<PresentRecord> record;
  if (timing->presented) {
    const std::int64_t period = display_period();
    if (period > 0) {
      const double waited = timing->queue_wait_us > period / 4 ? 1.0 : 0.0;
      queue_fill_ += kQueueFillSmoothing * (waited - queue_fill_);
    }
    const std::int64_t previous = stats_.last_new_present_us;
    stats_.presented(timing->render_us, timing->present_us, fresh);
    if (fresh && current_) pace(*current_, host_now + display_delay_us(period));
    record = PresentRecord{.host_us = timing->present_us,
                           .pts_us = current_ ? current_->pts_us : kNoPts,
                           .fresh = fresh,
                           .render_us = timing->render_us,
                           .interval_us = fresh && previous != 0 ? timing->present_us - previous : 0,
                           .dropped = stats_.dropped,
                           .why = std::move(why),
                           .missing = stats_.missing,
                           .lag_us = fresh && stats_.lag.count() > 0 ? stats_.lag.latest() : 0,
                           .uninterpolated = stats_.uninterpolated};
  }
  check_end(host_now);
  return record;
}

std::int64_t Session::pacing_period_us() const {
  // Un écran plus lent que la sortie saute des images par construction : sa période compte.
  const std::int64_t nominal = config_.extra != nullptr ? config_.extra->output_period_us() : 0;
  return nominal > 0 ? std::max(nominal, display_period()) : 0;
}

// Mesurée faute de mieux : la présentation est à la demande, chaque intervalle est donc un
// multiple de la période (six à 24 i/s sur 144 Hz, des centaines avant la première image) et le
// plus court des derniers la borne. La médiane rendait la cadence du contenu : l'image suivante
// devenait due une période d'image trop tôt et les images partaient par paires.
std::int64_t Session::display_period() const {
  if (config_.display_period_us > 0) return config_.display_period_us;
  const std::int64_t shortest = stats_.present_interval.percentile(0);
  return shortest > 0 ? std::min(shortest, kAssumedPeriodUs) : kAssumedPeriodUs;
}

// Imposé par le réglage, ou estimé : la file de l'écran a de la place, l'image part après son
// rendu et paraît au vsync suivant (une demi-période plus tard en moyenne) ; file pleine, les
// depth − 1 images déjà en file passent d'abord.
std::int64_t Session::display_delay_us(std::int64_t period) const {
  if (config_.video_latency_us > 0) return config_.video_latency_us;
  if (queue_fill_ > 0.5) return std::int64_t{std::max(config_.swapchain_depth - 1, 0)} * period + period / 2;
  return stats_.render.median() + period / 2;
}

void Session::set_display(std::int64_t period_us, double hz) {
  if (period_us == config_.display_period_us && hz == config_.display_hz) return;
  config_.display_period_us = period_us;
  config_.display_hz = hz;
  follow_display();
}

void Session::follow_display() {
  double speed = 1.0;
  if (config_.display_sync && config_.extra != nullptr) {
    const double hz = config_.display_hz > 0.0        ? config_.display_hz
                      : config_.display_period_us > 0 ? 1e6 / static_cast<double>(config_.display_period_us)
                                                      : 0.0;
    speed = core::vsync_speed(hz, config_.extra->output_fps());
  }
  if (speed == speed_) return;
  if (stages_.set_speed(speed)) {  // boîte audio pleine : redemandé au tour suivant
    speed_ = speed;
    osd_at_us_.reset();
  }
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
    if (config_.stop_after_us > 0 && played >= config_.stop_after_us) finished_ = true;
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

std::vector<std::string> Session::osd_lines() const {
  const StatsSettings& show = show_stats_;
  std::vector<std::string> lines;
  if (show.all || show.chain) lines.insert(lines.end(), config_.header.begin(), config_.header.end());
  if (show.all || show.tracks) lines.insert(lines.end(), config_.tracks.begin(), config_.tracks.end());
  if ((show.all || show.output) && !config_.output.empty()) lines.push_back(config_.output);
  if (show.all || show.frames) {
    const std::uint64_t skipped = config_.extra != nullptr ? config_.extra->skipped() : 0;
    lines.push_back(i18n::tr_format(
        "frame {} (σ {}) · render {} · skipped {}{}", format_ms(stats_.frame_interval.median()),
        format_ms(static_cast<std::int64_t>(stats_.frame_interval.stddev())),
        format_ms(stats_.render.median()), stats_.dropped,
        skipped > 0 ? std::format(" (+ {} avant calcul)", skipped) : std::string()));
    const std::int64_t period = pacing_period_us();
    std::string synced;
    if (speed_ != 1.0) {
      synced = i18n::tr_format(" · locked to the display {:+.3f} %", (speed_ - 1.0) * 100.0);
      std::ranges::replace(synced, '.', ',');
    }
    lines.push_back(i18n::tr_format(
        "smoothness {}{} fps · missed {}{} · p95 delay {}{}", format_fps(stats_.shown_fps),
        period > 0 ? std::format(" / {}", format_fps(1e6 / static_cast<double>(period))) : std::string(),
        stats_.missing,
        stats_.uninterpolated > 0 ? i18n::tr_format(" (+ {} cuts and held drawings)", stats_.uninterpolated)
                                  : std::string(),
        format_ms(stats_.lag.percentile(95)), synced));
  }
  if ((show.all || show.rife) && config_.extra != nullptr) {
    if (auto rife = config_.extra->rife()) lines.push_back(std::move(*rife));
  }
  if ((show.all || show.memory) && config_.extra != nullptr) {
    if (auto memory = config_.extra->memory()) lines.push_back(std::move(*memory));
  }
  if ((show.all || show.memory) && config_.extra != nullptr) {
    if (auto gpu = config_.extra->gpu()) lines.push_back(std::move(*gpu));
  }
  if (show.all || show.position) {
    const std::int64_t position = current_ ? current_->pts_us : kNoPts;
    lines.push_back(std::format("{} / {}{}", format_position(position), format_position(config_.duration_us),
                                paused_ ? " · pause" : ""));
  }
  // Les refus restent visibles, statistiques ou pas.
  lines.insert(lines.end(), refusals_.begin(), refusals_.end());
  return lines;
}

ControlsView Session::view() const {
  std::int64_t position = kNoPts;
  if (pending_seek_us_) {
    position = *pending_seek_us_;
  } else if (current_) {
    position = current_->pts_us;
  }
  return ControlsView{.position_us = position,
                      .duration_us = config_.duration_us,
                      .paused = paused_,
                      .fullscreen = screen_.fullscreen(),
                      .stats = show_stats_.all,
                      .chapters = config_.chapters};
}

}  // namespace ajp::app::play
