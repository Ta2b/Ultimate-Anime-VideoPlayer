#include "app/play/chain_shots.h"

#include <algorithm>

#include "app/play/chain.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

constexpr std::int64_t kSeriesCoverUs = 60'000'000;  // une minute analysée : la moyenne de la série tient
constexpr std::int64_t kSeekBehindUs = 2'000'000;    // la lecture revient derrière l'analyse : saut
constexpr std::int64_t kSeekAheadUs = 5'000'000;     // ou la dépasse de loin
constexpr std::int64_t kSeekEveryUs = 1'000'000;

}  // namespace

void open_shots(ShotWork& work, const std::filesystem::path& file, const PlaySettings& settings,
                std::int64_t start_us) {
  if (!settings.grade || !settings.grade_values.adaptive || file.empty()) {
    work.probe.reset();
    return;
  }
  if (work.file == file && work.timeline) {
    if (!work.probe || (work.probe->failure() && work.probe->finished())) return;
    work.probe->seek(std::max<std::int64_t>(start_us, 0));
    return;
  }
  work.probe.reset();  // l'analyse de l'autre fichier s'arrête avant que sa chronologie parte
  work.file = file;
  work.timeline = std::make_unique<media::ShotTimeline>();
  work.probe = media::ShotProbe::start(file, *work.timeline, std::max<std::int64_t>(start_us, 0));
}

std::optional<ShotScene> next_shot(Chain& to, std::int64_t now_us) {
  if (to.shots == nullptr || !to.shots->timeline) return std::nullopt;
  const media::ShotTimeline& timeline = *to.shots->timeline;
  const ControlsView view = to.session.view();
  media::ShotProbe* probe = to.shots->probe.get();
  if (probe != nullptr && view.position_us != kNoPts) {
    probe->set_playhead(view.position_us);
    const std::int64_t frontier = probe->frontier_us();
    const bool away =
        view.position_us < frontier - kSeekBehindUs || view.position_us > frontier + kSeekAheadUs;
    if (away && !timeline.at(view.position_us) && now_us - to.shots_seek_us > kSeekEveryUs) {
      to.shots_seek_us = now_us;
      probe->seek(view.position_us);
    }
  }
  // L'image qui suit la dernière préparée : celle que RTX HDR traite au prochain tour, et que le grade
  // de l'écran recevra après la dernière (set_grade_after). Son instant : la dernière plus l'écart entre
  // les deux dernières (la cadence de sortie, interpolation comprise).
  const RingFrame* newest = to.session.newest_frame();
  if (newest == nullptr || newest->pts_us == kNoPts) return std::nullopt;
  if (newest->pts_us != to.shots_newest_pts) {
    if (to.shots_newest_pts != kNoPts && newest->pts_us > to.shots_newest_pts)
      to.shots_step_us = std::clamp<std::int64_t>(newest->pts_us - to.shots_newest_pts, 4'000, 50'000);
    to.shots_newest_pts = newest->pts_us;
  }
  const auto shot = timeline.at(newest->pts_us + to.shots_step_us);
  if (!shot) return std::nullopt;
  const media::ShotStats& s = shot->stats;
  return ShotScene{.stats = {.mean = s.mean,
                             .p05 = s.p05,
                             .p50 = s.p50,
                             .p95 = s.p95,
                             .p995 = s.p995,
                             .dark = s.dark,
                             .bright = s.bright,
                             .blue = s.blue,
                             .chroma = s.chroma},
                   .serial = shot->serial,
                   .fade = shot->fade};
}

std::optional<double> shots_series_luma(const Chain& to) {
  if (to.shots == nullptr || !to.shots->timeline) return std::nullopt;
  const bool whole = to.shots->probe && to.shots->probe->finished() && !to.shots->probe->failure();
  if (!whole && to.shots->timeline->covered_us() < kSeriesCoverUs) return std::nullopt;
  return to.shots->timeline->mean_luma();
}

}  // namespace ajp::app::play
