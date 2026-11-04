#include "app/play/chain_shots.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "app/play/chain.h"
#include "core/log.h"
#include "media/shot_cache.h"

namespace ajp::app::play {
namespace {

constexpr std::int64_t kSeriesCoverUs = 60'000'000;  // une minute analysée : la moyenne de la série tient
constexpr std::int64_t kSeekBehindUs = 2'000'000;    // la lecture revient derrière l'analyse : saut
constexpr std::int64_t kSeekAheadUs = 5'000'000;     // ou la dépasse de loin
constexpr std::int64_t kSeekEveryUs = 1'000'000;

// La vidéo, du cache si elle y est : sa chronologie entière, sans analyse, ou la part déjà faite, que
// l'analyse complète.
ShotJob cached_job(const ShotWork& work, const std::filesystem::path& file) {
  ShotJob job;
  job.cache_dir = work.cache_dir;
  job.file = file;
  job.key = media::shot_cache_key(file);
  job.timeline = std::make_unique<media::ShotTimeline>();
  if (work.cache_dir.empty()) return job;
  if (auto cached = media::load_shot_cache(work.cache_dir, job.key)) {
    for (const media::Shot& shot : cached->shots) job.timeline->add(shot);
    job.complete = cached->complete;
  }
  return job;
}

// Analysée jusqu'au bout sans trou : au cache.
void keep_if_done(const ShotWork& work, ShotJob& job) {
  if (job.complete || !job.probe || !job.probe->finished() || job.probe->failure()) return;
  const media::ShotTimeline& timeline = *job.timeline;
  if (timeline.size() == 0 || timeline.first_gap_us() < timeline.end_us()) return;
  job.complete = true;
  if (work.cache_dir.empty()) return;
  const auto log = log_channel("app");
  if (media::save_shot_cache(work.cache_dir, job.key, timeline.shots(), true))
    log->info("plans de {} gardés : {} plans", job.file.filename().string(), timeline.size());
}

}  // namespace

ShotJob& ShotJob::operator=(ShotJob&& other) noexcept {
  if (this == &other) return *this;
  retire();
  cache_dir = std::move(other.cache_dir);
  file = std::move(other.file);
  key = std::move(other.key);
  timeline = std::move(other.timeline);
  probe = std::move(other.probe);
  complete = other.complete;
  return *this;
}

ShotJob::~ShotJob() {
  retire();
}

void ShotJob::retire() noexcept {
  probe.reset();  // le thread d'analyse s'arrête avant la lecture de la chronologie
  if (complete || !timeline || timeline->size() == 0 || cache_dir.empty() || key.empty()) return;
  try {
    if (media::save_shot_cache(cache_dir, key, timeline->shots(), false)) {
      const auto log = log_channel("app");
      log->info("plans de {} gardés en partie : {} plans", file.filename().string(), timeline->size());
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch) : le cache est un confort, pas une obligation
  }
  timeline.reset();  // gardée une fois
}

void open_shots(ShotWork& work, const std::filesystem::path& file, const PlaySettings& settings,
                std::int64_t start_us, bool source_hdr) {
  // Le grade d'une source HDR est neutre (grade_params.cpp) : sans grade natif HDR demandé, aucun plan à
  // lire.
  if (!settings.grade || !settings.grade_values.adaptive || file.empty() ||
      (source_hdr && !settings.hdr_native)) {
    work.current.probe.reset();
    return;
  }
  const std::int64_t from = std::max<std::int64_t>(start_us, 0);
  if (work.current.file == file && work.current.timeline) {
    if (work.current.complete) return;
    if (!work.current.probe) {  // arrêtée quand le grade adaptatif s'est coupé : elle reprend
      work.current.probe = media::ShotProbe::start(file, *work.current.timeline, from, work.probe);
    } else if (!work.current.probe->failure()) {
      work.current.probe->seek(from);
    }
    return;
  }
  work.current.retire();  // l'analyse de l'autre fichier s'arrête, sa part faite va au cache
  work.current = cached_job(work, file);
  if (work.current.complete) {
    const auto log = log_channel("app");
    log->info("plans de {} lus du cache : {} plans", file.filename().string(), work.current.timeline->size());
    return;
  }
  if (!work.current.probe)
    work.current.probe = media::ShotProbe::start(file, *work.current.timeline, from, work.probe);
}

void tend_shots(ShotWork& work) {
  keep_if_done(work, work.current);
}

namespace {

ShotScene scene_of(const media::ShotStats& s, std::uint64_t serial, bool fade) {
  return ShotScene{.stats = {.mean = s.mean,
                             .p05 = s.p05,
                             .p50 = s.p50,
                             .p95 = s.p95,
                             .p995 = s.p995,
                             .dark = s.dark,
                             .bright = s.bright,
                             .blue = s.blue,
                             .chroma = s.chroma},
                   .light = {.apl_nits = s.apl_nits, .lit = s.lit},
                   .serial = serial,
                   .fade = fade};
}

// Sans plan analysé : la scène que l'analyse à l'ingestion a vue pour cette image (media/live_scenes.h).
// Ses numéros ne croisent jamais ceux des plans.
std::optional<ShotScene> live_scene(const Chain& to, std::int64_t pts_us) {
  const auto live = to.live.scenes().at(pts_us);
  if (!live) return std::nullopt;
  return scene_of(live->stats, (std::uint64_t{1} << 62U) | live->serial, false);
}

std::optional<std::int64_t> next_pts(Chain& to) {
  const RingFrame* newest = to.session.newest_frame();
  if (newest == nullptr || newest->pts_us == kNoPts) return std::nullopt;
  return newest->pts_us + to.shots_step_us;
}

}  // namespace

std::optional<ShotScene> next_live(Chain& to) {
  const auto pts = next_pts(to);
  return pts ? live_scene(to, *pts) : std::nullopt;
}

void feed_plan_shots(Chain& to) {
  if (to.shots == nullptr || !to.shots->current.timeline || to.shots->current.file != to.file) return;
  const media::ShotTimeline& timeline = *to.shots->current.timeline;
  if (timeline.size() == to.plan_shots_seen) return;
  to.plan_shots_seen = timeline.size();
  std::vector<media::Shot> shots = timeline.shots();
  if (to.shot_cuts) to.live.cuts().set_shots(shots);
  to.screen.set_plan_shots(std::move(shots));
}

std::optional<ShotScene> next_shot(Chain& to, std::int64_t now_us) {
  if (to.shots == nullptr || !to.shots->current.timeline || to.shots->current.file != to.file) {
    const auto pts = next_pts(to);
    return pts ? live_scene(to, *pts) : std::nullopt;
  }
  ShotJob& job = to.shots->current;
  tend_shots(*to.shots);
  const media::ShotTimeline& timeline = *job.timeline;
  const ControlsView view = to.session.view();
  // Le grade adaptatif s'est allumé après l'ouverture (écran sans profil, où with_calibrated_grade le coupe,
  // puis la calibration ou le panneau le rallume) : l'analyse arrêtée à l'ouverture reprend ici, sinon seul
  // ce qu'elle avait vu en partant serait connu.
  if (!job.probe && !job.complete && view.position_us != kNoPts)
    job.probe = media::ShotProbe::start(job.file, *job.timeline, std::max<std::int64_t>(view.position_us, 0),
                                        to.shots->probe);
  media::ShotProbe* probe = job.probe.get();
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
  // L'image qui suit la dernière préparée : celle que le GPU traite au prochain tour, et que le grade
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
  if (!shot) return live_scene(to, newest->pts_us + to.shots_step_us);
  return scene_of(shot->stats, shot->serial, shot->fade);
}

std::optional<double> shots_series_luma(const Chain& to) {
  if (to.shots == nullptr || !to.shots->current.timeline) return std::nullopt;
  const ShotJob& job = to.shots->current;
  if (!job.complete && job.timeline->covered_us() < kSeriesCoverUs) return std::nullopt;
  return job.timeline->mean_luma();
}

}  // namespace ajp::app::play
