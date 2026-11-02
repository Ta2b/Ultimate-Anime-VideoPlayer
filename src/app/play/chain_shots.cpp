#include "app/play/chain_shots.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "app/play/chain.h"
#include "app/play/episodes.h"
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
                std::int64_t start_us) {
  if (!settings.grade || !settings.grade_values.adaptive || file.empty()) {
    work.current.probe.reset();
    work.next.probe.reset();
    return;
  }
  const std::int64_t from = std::max<std::int64_t>(start_us, 0);
  if (work.current.file == file && work.current.timeline) {
    if (work.current.complete) return;
    if (!work.current.probe) {  // arrêtée quand le grade adaptatif s'est coupé : elle reprend
      work.current.probe = media::ShotProbe::start(file, *work.current.timeline, from);
    } else if (!work.current.probe->failure()) {
      work.current.probe->seek(from);
    }
    return;
  }
  work.current.retire();  // l'analyse de l'autre fichier s'arrête, sa part faite va au cache
  work.next_tried = false;
  if (work.next.file == file && work.next.timeline) {
    work.current = std::exchange(work.next, ShotJob{});  // l'épisode analysé d'avance
    if (work.current.probe && !work.current.complete) work.current.probe->seek(from);
  } else {
    work.next = ShotJob{};
    work.current = cached_job(work, file);
  }
  if (work.current.complete) {
    const auto log = log_channel("app");
    log->info("plans de {} lus du cache : {} plans", file.filename().string(), work.current.timeline->size());
    return;
  }
  if (!work.current.probe) work.current.probe = media::ShotProbe::start(file, *work.current.timeline, from);
}

void tend_shots(ShotWork& work) {
  keep_if_done(work, work.current);
  keep_if_done(work, work.next);
  if (work.next.probe && work.next.complete) work.next.probe.reset();
  // La vidéo lue est connue : l'épisode suivant s'analyse en entier, sur deux threads, pour s'ouvrir prêt.
  if (!work.current.complete || work.next_tried || work.current.file.empty()) return;
  work.next_tried = true;
  const auto next = sibling_episode(work.current.file, 1);
  if (!next) return;
  work.next = cached_job(work, *next);
  if (work.next.complete || work.next.key.empty()) return;
  work.next.probe = media::ShotProbe::start(
      *next, *work.next.timeline, 0,
      media::ShotProbeOptions{.threads = 2, .lead_us = std::numeric_limits<std::int64_t>::max() / 2});
  const auto log = log_channel("app");
  log->info("pré-analyse de l'épisode suivant : {}", next->filename().string());
}

std::optional<ShotScene> next_shot(Chain& to, std::int64_t now_us) {
  if (to.shots == nullptr || !to.shots->current.timeline) return std::nullopt;
  tend_shots(*to.shots);
  const media::ShotTimeline& timeline = *to.shots->current.timeline;
  const ControlsView view = to.session.view();
  media::ShotProbe* probe = to.shots->current.probe.get();
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
  if (to.shots == nullptr || !to.shots->current.timeline) return std::nullopt;
  const ShotJob& job = to.shots->current;
  if (!job.complete && job.timeline->covered_us() < kSeriesCoverUs) return std::nullopt;
  return job.timeline->mean_luma();
}

}  // namespace ajp::app::play
