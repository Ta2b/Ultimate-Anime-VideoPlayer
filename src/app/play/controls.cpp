#include "app/play/controls.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

namespace ajp::app::play {
namespace {

constexpr std::int64_t kSecondUs = 1'000'000;

float ramp(std::int64_t elapsed_us) {
  return std::clamp(static_cast<float>(elapsed_us) / static_cast<float>(kFadeUs), 0.0F, 1.0F);
}

}  // namespace

std::string format_clock(std::int64_t us, std::int64_t duration_us) {
  if (us == kNoPts || us < 0) return "--:--";
  const std::int64_t total = us / kSecondUs;
  const std::int64_t seconds = total % 60;
  const std::int64_t minutes = (total / 60) % 60;
  const std::int64_t hours = total / 3600;
  if (hours > 0 || duration_us >= 3600 * kSecondUs)
    return std::format("{}:{:02}:{:02}", hours, minutes, seconds);
  return std::format("{}:{:02}", total / 60, seconds);
}

float subtitle_lift(float text_bottom, float height, float reserved) {
  if (reserved <= 0.0F) return 0.0F;
  return std::max(0.0F, text_bottom - (height - reserved));
}

double fraction_of(std::int64_t position_us, std::int64_t duration_us) {
  if (duration_us <= 0 || position_us == kNoPts) return 0.0;
  return std::clamp(static_cast<double>(position_us) / static_cast<double>(duration_us), 0.0, 1.0);
}

std::int64_t time_at(double fraction, std::int64_t duration_us) {
  if (duration_us <= 0 || std::isnan(fraction)) return 0;
  return static_cast<std::int64_t>(
      std::llround(std::clamp(fraction, 0.0, 1.0) * static_cast<double>(duration_us)));
}

void Controls::activity(std::int64_t now_us) {
  // Une activité après l'effacement fait repartir le fondu d'apparition.
  if (!active_at_ || left_ || now_us - *active_at_ >= kHideAfterUs + kFadeUs) shown_since_ = now_us;
  active_at_ = now_us;
  left_ = false;
}

void Controls::pointer_left(std::int64_t now_us) {
  if (left_) return;
  left_ = true;
  left_at_ = now_us;
}

float Controls::opacity(std::int64_t now_us, bool paused) const {
  // OLED : une pause longue sans activité laisse la barre s'effacer, comme en lecture.
  const bool paused_hold = paused && !(oled_care_ && active_at_ && now_us - *active_at_ >= kPausedHideUs);
  if (paused && !paused_hold && !held_ && !dragging_) {
    if (left_) return 0.0F;
    return 1.0F - ramp(now_us - (*active_at_ + kPausedHideUs));
  }
  if (paused_hold || held_ || dragging_) {
    // Fondu d'apparition seulement s'il vient d'une activité récente.
    return active_at_ && now_us - shown_since_ < kFadeUs && !left_ ? ramp(now_us - shown_since_) : 1.0F;
  }
  if (!active_at_) return 0.0F;
  const std::int64_t hide_at = left_ ? left_at_ : *active_at_ + kHideAfterUs;
  const float out = 1.0F - ramp(now_us - hide_at);
  const float in = ramp(now_us - shown_since_);
  return std::min(in, out);
}

bool Controls::cursor_hidden(std::int64_t now_us, bool paused) const {
  return !left_ && opacity(now_us, paused) == 0.0F;
}

cmd::SetVolume Controls::set_volume(float volume) {
  volume_ = std::isnan(volume) ? volume_ : std::clamp(volume, 0.0F, kMaxVolume);
  muted_ = false;
  return volume_command();
}

cmd::SetVolume Controls::step_volume(float delta) {
  // Les pas tombent sur des multiples de kVolumeStep, même après un réglage à la souris.
  const float steps = std::round((volume_ + delta) / kVolumeStep);
  return set_volume(steps * kVolumeStep);
}

cmd::SetVolume Controls::toggle_mute() {
  muted_ = !muted_;
  return volume_command();
}

std::optional<Controls::VideoClick> Controls::click(int clicks) {
  if (clicks == 1) return VideoClick::TogglePause;
  if (clicks == 2) return VideoClick::FullscreenUndoPause;
  return std::nullopt;
}

void Controls::press(double fraction) {
  dragging_ = true;
  drag_fraction_ = std::clamp(fraction, 0.0, 1.0);
}

std::optional<cmd::Seek> Controls::drag_to(double fraction, std::int64_t duration_us, std::int64_t now_us) {
  dragging_ = true;
  drag_fraction_ = std::clamp(fraction, 0.0, 1.0);
  if (duration_us <= 0) return std::nullopt;
  const std::int64_t target = time_at(drag_fraction_, duration_us);
  if (target == drag_target_us_ || (drag_seek_at_ && now_us - *drag_seek_at_ < kDragSeekUs))
    return std::nullopt;
  drag_seek_at_ = now_us;
  drag_target_us_ = target;
  return cmd::Seek{.target_us = target, .exact = false};
}

cmd::Seek Controls::release(double fraction, std::int64_t duration_us) {
  dragging_ = false;
  drag_seek_at_.reset();
  drag_target_us_ = -1;
  return cmd::Seek{.target_us = time_at(fraction, duration_us), .exact = true};
}

bool Controls::brief_shown(std::int64_t now_us) const {
  if (toast_.empty()) return false;
  const std::int64_t age = now_us - toast_at_us_;
  return age >= 0 && age < kToastUs;
}

const Controls::Notice* Controls::notice_at(std::int64_t now_us) const {
  for (const Notice& notice : notices_) {
    if (now_us >= notice.at_us && now_us < notice.at_us + notice.duration_us) return &notice;
  }
  return nullptr;
}

void Controls::notify(const i18n::Translated& text, std::int64_t now_us) {
  toast_ = text.str();
  toast_at_us_ = now_us;
  // Les avis en attente repartent après ce message bref, dans leur ordre.
  std::int64_t cursor = now_us + kToastUs;
  for (Notice& notice : notices_) {
    if (notice.at_us + notice.duration_us <= now_us) continue;
    notice.at_us = cursor;
    cursor += notice.duration_us;
  }
}

void Controls::warn(const i18n::Translated& text, std::int64_t now_us) {
  std::erase_if(notices_, [&](const Notice& n) { return n.at_us + n.duration_us <= now_us; });
  if (!notices_.empty() && notices_.back().text == text.str()) return;
  if (notices_.size() >= kNoticesKept) notices_.erase(notices_.begin());
  std::int64_t start = brief_shown(now_us) ? toast_at_us_ + kToastUs : now_us;
  if (!notices_.empty()) start = std::max(start, notices_.back().at_us + notices_.back().duration_us);
  const std::int64_t duration = std::clamp<std::int64_t>(
      static_cast<std::int64_t>(text.str().size()) * kNoticeCharUs, kNoticeUs, kNoticeMaxUs);
  notices_.push_back(Notice{.text = text.str(), .at_us = start, .duration_us = duration});
}

void Controls::defer(const i18n::Translated& text) {
  if (std::ranges::find(deferred_, text.str()) == deferred_.end()) deferred_.push_back(text.str());
}

void Controls::show_deferred(std::int64_t now_us) {
  std::vector<std::string> waiting = std::move(deferred_);
  deferred_.clear();
  for (const std::string& text : waiting) warn(i18n::Translated::data(text), now_us);
}

std::optional<std::string_view> Controls::toast(std::int64_t now_us) const {
  if (brief_shown(now_us) && toast_opacity(now_us) > 0.0F) return toast_;
  if (const Notice* notice = notice_at(now_us)) return notice->text;
  return std::nullopt;
}

float Controls::toast_opacity(std::int64_t now_us) const {
  if (brief_shown(now_us)) {
    const std::int64_t age = now_us - toast_at_us_;
    return 1.0F - ramp(age - (kToastUs - kFadeUs));
  }
  if (const Notice* notice = notice_at(now_us)) {
    const std::int64_t age = now_us - notice->at_us;
    return 1.0F - ramp(age - (notice->duration_us - kFadeUs));
  }
  return 0.0F;
}

bool Controls::fading_in(std::int64_t now_us) const {
  return active_at_ && !left_ && now_us >= shown_since_ && now_us - shown_since_ < kFadeUs;
}

bool Controls::animating(std::int64_t now_us, bool paused) const {
  const float bar = opacity(now_us, paused);
  const float toast = toast_opacity(now_us);
  return (bar > 0.0F && bar < 1.0F) || toast > 0.0F || fading_in(now_us);
}

bool Controls::needs_frame(std::int64_t now_us, const ControlsView& view, bool input_pending) const {
  if (framed_at_ && now_us - *framed_at_ < kUiFrameUs) return false;
  const bool visible = shown(now_us, view.paused) || toast_opacity(now_us) > 0.0F || fading_in(now_us);
  if (!visible) return drawn_;  // une dernière fois, pour ôter la barre de l'écran
  if (input_pending || !drawn_ || animating(now_us, view.paused) || view.paused != framed_paused_)
    return true;
  const std::int64_t step = view.position_us == kNoPts ? -1 : view.position_us / kPositionStepUs;
  return step != framed_position_step_;
}

void Controls::framed(std::int64_t now_us, const ControlsView& view, bool drawn) {
  framed_at_ = now_us;
  framed_position_step_ = view.position_us == kNoPts ? -1 : view.position_us / kPositionStepUs;
  framed_paused_ = view.paused;
  drawn_ = drawn;
}

}  // namespace ajp::app::play
