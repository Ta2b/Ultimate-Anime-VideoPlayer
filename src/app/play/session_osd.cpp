// Ce que la session montre hors de l'image : les lignes de l'OSD et l'état de la barre de lecture.
#include <format>

#include "app/play/i18n.h"
#include "app/play/session.h"

namespace ajp::app::play {

// Les deux lignes des images : intervalle, rendu, images sautées ; fluidité, manques, retard et calage.
void Session::frame_lines(std::vector<std::string>& lines) const {
  const std::uint64_t skipped = config_.extra != nullptr ? config_.extra->skipped() : 0;
  lines.push_back(i18n::tr_format(
      "frame {} (σ {}) · render {} · skipped {}{}", format_ms(stats_.frame_interval.median()),
      format_ms(static_cast<std::int64_t>(stats_.frame_interval.stddev())), format_ms(stats_.render.median()),
      stats_.dropped, skipped > 0 ? i18n::tr_format(" (+ {} before compute)", skipped) : std::string()));
  const std::int64_t period = pacing_period_us();
  std::string synced;
  if (speed_ != 1.0) {
    synced = i18n::tr_format(" · locked to the display {:+.3f} %", (speed_ - 1.0) * 100.0);
  }
  lines.push_back(i18n::tr_format(
      "smoothness {}{} fps · missed {}{} · p95 delay {}{}", format_fps(stats_.shown_fps),
      period > 0 && !(config_.extra != nullptr && config_.extra->uncapped())
          ? std::format(" / {}", format_fps(1e6 / static_cast<double>(period)))
          : std::string(),
      stats_.missing,
      stats_.uninterpolated > 0 ? i18n::tr_format(" (+ {} cuts and held drawings)", stats_.uninterpolated)
                                : std::string(),
      format_ms(stats_.lag.percentile(95)), synced));
}

// Ce que la chaîne mesure elle-même : interpolation, mémoire, GPU.
void Session::extra_lines(const StatsSettings& show, std::vector<std::string>& lines) const {
  if (show.all || show.rife) {
    if (auto rife = config_.extra->rife()) lines.push_back(std::move(*rife));
  }
  if (show.all || show.memory) {
    if (auto memory = config_.extra->memory()) lines.push_back(std::move(*memory));
    if (auto gpu = config_.extra->gpu()) lines.push_back(std::move(*gpu));
  }
}

std::vector<std::string> Session::osd_lines() const {
  const StatsSettings& show = show_stats_;
  std::vector<std::string> lines;
  if (show.all || show.chain) lines.insert(lines.end(), config_.header.begin(), config_.header.end());
  if (show.all || show.tracks) lines.insert(lines.end(), config_.tracks.begin(), config_.tracks.end());
  if ((show.all || show.output) && !config_.output.empty()) lines.push_back(config_.output);
  if (show.all || show.frames) frame_lines(lines);
  if (config_.extra != nullptr) extra_lines(show, lines);
  if (show.all || show.position) {
    const std::int64_t position = current_ ? current_->pts_us : kNoPts;
    lines.push_back(std::format("{} / {}{}", format_position(position), format_position(config_.duration_us),
                                paused_ ? i18n::tr(" · paused") : ""));
  }
  // Les refus (nom d'étage, code interne, détail en français) sont pour les statistiques et le journal :
  // incrustés sur l'image, un refus passager y restait jusqu'à la fin de l'épisode.
  if (show.all) lines.insert(lines.end(), refusals_.begin(), refusals_.end());
  return lines;
}

// Débogage des paires : la teinte de l'image montrée si sa paire a été jugée tenue ou coupée.
std::optional<PairTint> Session::pair_tint_now() const {
  if (!show_stats_.pairs || config_.extra == nullptr || !current_ || current_->pts_us == kNoPts)
    return std::nullopt;
  const auto verdict = config_.extra->pair_verdict(current_->epoch, current_->pts_us);
  if (!verdict) return std::nullopt;
  return pair_tint(*verdict);
}

// La présentation de ce tour : l'image courante, ses sous-titres, l'OSD et l'interface.
ShowRequest Session::show_request(const UiFrame& ui, const subs::SubtitleFrame* subtitles) const {
  const bool shown = ui.draw != nullptr;
  return ShowRequest{
      .frame = current_ ? &*current_ : nullptr,
      .subtitles = subtitles,
      .osd = osd_,
      .ui = ui.draw,
      .ui_serial = shown ? ui.serial : 0,
      .ui_bottom = shown ? ui.bottom : 0,
      .ui_top = shown ? ui.top : 0,
      .redraws_expected = config_.steady_cadence || config_.vrr_floor_us > 0 || shown || !osd_.empty(),
      .target_us = fresh_ ? target_host_us_ : 0,
      .tint = pair_tint_now()};
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
