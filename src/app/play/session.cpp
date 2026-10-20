#include "app/play/session.h"

#include <algorithm>
#include <format>
#include <type_traits>
#include <utility>
#include <variant>

#include "core/frame.h"

namespace ajp::app::play {
namespace {

// Fin de flux : plus rien n'arrive depuis ce délai, la lecture est finie.
constexpr std::int64_t kEndGraceUs = 1'000'000;
constexpr std::size_t kMaxRefusals = 3;
// L'OSD se recompose quatre fois par seconde, pas à chaque vsync.
constexpr std::int64_t kOsdPeriodUs = 250'000;
// Cadence fixe : marge sous la période de l'écran, pour présenter avant le vsync visé.
constexpr std::int64_t kCadenceSlackUs = 500;
// À l'amorce, la première image attend ses sous-titres au plus ce temps.
constexpr std::int64_t kPrerollSubtitleWaitUs = 100'000;

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
      Overloaded{[this](const cmd::Play&) {
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
                 [this](const cmd::Quit&) { finished_ = true; },
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
  stages_.pause();
  ++epoch_;
  if (!stages_.flush(epoch_) || !stages_.seek(cmd::Seek{.target_us = target_us, .exact = exact})) {
    on_refusal({.stage = "main",
                .refusal = {RefusalCode::SeekFailed,
                            std::format("seek vers {} : boîte pleine", format_position(target_us))}});
  }
  for (const RingFrame& frame : queue_.take_stale(epoch_)) {
    retire(frame);
    ++stats_.stale;
  }
  stats_.restart_pacing();
  subtitles_.clear();
  subtitle_requests_ = 0;
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
  queue_.push(frame);
  request_subtitles(frame);
}

void Session::on_subtitles(subs::SubtitleFrame&& frame) {
  if (frame.epoch != epoch_) return;  // demande d'avant un seek : l'atlas rentre chez subs
  if (subtitle_requests_ > 0) --subtitle_requests_;
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
  if (!config_.subtitles || frame.pts_us == kNoPts || subtitle_requests_ >= kMaxSubtitleRequests) return;
  const OutputSize size = screen_.size();
  const subs::RenderRequest request{.pts_us = frame.pts_us,
                                    .epoch = frame.epoch,
                                    .target = {.width = size.width,
                                               .height = size.height,
                                               .video_width = config_.video_width,
                                               .video_height = config_.video_height,
                                               .video_matrix = config_.video_matrix,
                                               .font_scale = config_.subtitle_scale}};
  if (stages_.request_subtitles(request)) ++subtitle_requests_;
}

void Session::retire(const RingFrame& frame) {
  stages_.slot_free(screen_.retire(frame));
}

void Session::choose(std::int64_t host_now) {
  if (prerolling_) {
    const RingFrame* head = queue_.first_of(epoch_);
    if (head == nullptr) return;  // l'image d'avant le seek reste à l'écran
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
  if (!now.valid || now.epoch != epoch_) return;  // horloge pas encore partie : l'image reste
  // L'image paraît au rafraîchissement qui suit sa présentation : on vise le milieu de la période
  // de l'écran (connue, ou mesurée en FIFO).
  const std::int64_t lead_us =
      (config_.display_period_us > 0 ? config_.display_period_us : stats_.present_interval.median()) / 2;
  late_.clear();
  const std::optional<RingFrame> picked = queue_.pick(now.media_at(host_now), lead_us, late_);
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
    const std::int64_t previous = stats_.last_new_present_us;
    stats_.presented(timing->render_us, timing->present_us, fresh);
    if (fresh && current_) pace(*current_, timing->present_us);
    record = PresentRecord{.host_us = timing->present_us,
                           .pts_us = current_ ? current_->pts_us : kNoPts,
                           .fresh = fresh,
                           .render_us = timing->render_us,
                           .interval_us = fresh && previous != 0 ? timing->present_us - previous : 0,
                           .dropped = stats_.dropped,
                           .why = std::move(why),
                           .missing = stats_.missing,
                           .lag_us = fresh && stats_.lag.count() > 0 ? stats_.lag.latest() : 0};
  }
  check_end(host_now);
  return record;
}

std::int64_t Session::pacing_period_us() const {
  // Un écran plus lent que la sortie saute des images par construction : sa période compte.
  const std::int64_t nominal = config_.extra != nullptr ? config_.extra->output_period_us() : 0;
  const std::int64_t display =
      config_.display_period_us > 0 ? config_.display_period_us : stats_.present_interval.median();
  return nominal > 0 ? std::max(nominal, display) : 0;
}

void Session::pace(const RingFrame& frame, std::int64_t present_us) {
  // Le retard ne se lit que pendant la lecture : en pause ou avant le départ, l'horloge est figée.
  const ClockSnapshot now = clock_.snapshot();
  const bool running = !paused_ && now.valid && now.epoch == epoch_ && frame.pts_us != kNoPts;
  stats_.paced(frame.pts_us, running ? now.media_at(present_us) - frame.pts_us : 0, pacing_period_us());
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
  if (stream_ended_ && !paused_ && queue_.empty() && host_now - last_fresh_host_us_ > kEndGraceUs) {
    finished_ = true;
  }
}

void Session::end() {
  for (const RingFrame& frame : queue_.take_all()) retire(frame);
  if (current_) retire(*current_);
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
    lines.push_back(std::format("image {} (σ {}) · rendu {} · sautées {}",
                                format_ms(stats_.frame_interval.median()),
                                format_ms(static_cast<std::int64_t>(stats_.frame_interval.stddev())),
                                format_ms(stats_.render.median()), stats_.dropped));
    const std::int64_t period = pacing_period_us();
    lines.push_back(std::format(
        "fluidité {}{} im/s · manquées {} · retard p95 {}", format_fps(stats_.shown_fps),
        period > 0 ? std::format(" / {}", format_fps(1e6 / static_cast<double>(period))) : std::string(),
        stats_.missing, format_ms(stats_.lag.percentile(95))));
  }
  if ((show.all || show.rife) && config_.extra != nullptr) {
    if (auto rife = config_.extra->rife()) lines.push_back(std::move(*rife));
  }
  if ((show.all || show.memory) && config_.extra != nullptr) {
    if (auto memory = config_.extra->memory()) lines.push_back(std::move(*memory));
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
