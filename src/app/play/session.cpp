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

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

}  // namespace

Session::Session(SessionConfig config, Stages& stages, Screen& screen, const MasterClock& clock)
    : config_(std::move(config)), stages_(stages), screen_(screen), clock_(clock) {
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
  std::visit(Overloaded{[this](const cmd::Play&) {
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
                        [](const auto&) {}},  // le reste ne concerne pas la tranche verticale
             command);
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
  subtitles_.clear();
  subtitle_requests_ = 0;
  prerolling_ = true;
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
  subtitles_.push(std::move(frame), current_ ? current_->pts_us : kNoPts);
}

void Session::on_refusal(const StageRefusal& refusal) {
  refusals_.push_back(
      std::format("{} : {} {}", refusal.stage, name(refusal.refusal.code), refusal.refusal.detail));
  if (refusals_.size() > kMaxRefusals) refusals_.erase(refusals_.begin());
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
                                               .video_matrix = config_.video_matrix}};
  if (stages_.request_subtitles(request)) ++subtitle_requests_;
}

void Session::retire(const RingFrame& frame) {
  stages_.slot_free(screen_.retire(frame));
}

void Session::choose(std::int64_t host_now) {
  if (prerolling_) {
    if (queue_.first_of(epoch_) == nullptr) return;  // l'image d'avant le seek reste à l'écran
    const std::optional<RingFrame> first = queue_.pop_front();
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
  // L'image paraît à la vsync qui suit : on vise le milieu de la période mesurée.
  const std::int64_t lead_us = stats_.present_interval.median() / 2;
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

const subs::SubtitleFrame* Session::subtitles_for(std::int64_t pts_us) {
  const subs::SubtitleFrame* frame = subtitles_.for_pts(pts_us);
  // Une image montrée sans réponse alors que d'autres attendent : le défaut du clignotement.
  if (frame == nullptr && subtitles_.size() > 0) ++stats_.subtitle_gaps;
  return frame;
}

std::expected<std::optional<PresentRecord>, Refusal> Session::tick(std::int64_t host_now) {
  fresh_ = false;
  choose(host_now);
  const bool fresh = fresh_;
  if (osd_.empty() || host_now - osd_at_us_ >= kOsdPeriodUs) {
    osd_ = osd_lines();
    osd_at_us_ = host_now;
  }
  const ShowRequest request{.frame = current_ ? &*current_ : nullptr,
                            .subtitles = current_ ? subtitles_for(current_->pts_us) : nullptr,
                            .osd = osd_};
  auto timing = screen_.show(request);
  if (!timing) return std::unexpected(std::move(timing.error()));

  std::optional<PresentRecord> record;
  if (timing->presented) {
    const std::int64_t previous = stats_.last_new_present_us;
    stats_.presented(timing->render_us, timing->present_us, fresh);
    record = PresentRecord{.host_us = timing->present_us,
                           .pts_us = current_ ? current_->pts_us : kNoPts,
                           .fresh = fresh,
                           .render_us = timing->render_us,
                           .interval_us = fresh && previous != 0 ? timing->present_us - previous : 0,
                           .dropped = stats_.dropped};
  }
  check_end(host_now);
  return record;
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
  std::vector<std::string> lines = config_.header;
  lines.push_back(std::format("image {} (σ {}) · rendu {} · sautées {}",
                              format_ms(stats_.frame_interval.median()),
                              format_ms(static_cast<std::int64_t>(stats_.frame_interval.stddev())),
                              format_ms(stats_.render.median()), stats_.dropped));
  const std::int64_t position = current_ ? current_->pts_us : kNoPts;
  lines.push_back(std::format("{} / {}{}", format_position(position), format_position(config_.duration_us),
                              paused_ ? " · pause" : ""));
  for (const std::string& refusal : refusals_) lines.push_back(refusal);
  return lines;
}

}  // namespace ajp::app::play
