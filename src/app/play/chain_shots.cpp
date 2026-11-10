#include "app/play/chain_shots.h"

#include <algorithm>
#include <expected>
#include <tuple>
#include <utility>
#include <vector>

#include "app/play/chain.h"
#include "app/play/latest_writer.h"
#include "app/play/options.h"
#include "core/log.h"
#include "core/utf8_path.h"
#include "media/shot_cache.h"

namespace ajp::app::play {
namespace {

constexpr std::int64_t kSeekBehindUs = 2'000'000;  // la lecture revient derrière l'analyse : saut
constexpr std::int64_t kSeekAheadUs = 5'000'000;   // ou la dépasse de loin
constexpr std::int64_t kSeekEveryUs = 1'000'000;

// Le grade lit les plans (sauf celui, neutre, d'une source HDR sans grade natif) ; l'interpolation aussi,
// sous
// --shot-cuts.
bool wants_shots(bool shot_cuts, const PlaySettings& settings, bool source_hdr) {
  const bool grade = settings.grade && (!source_hdr || settings.hdr_native);
  return grade || (shot_cuts && settings.rife_factor != 0);
}

// L'analyse de la vidéo depuis from_us. Son thread lit d'abord le cache : la chronologie entière, sans
// analyse, ou la part déjà faite, que l'analyse complète.
void start_probe(const ShotWork& work, ShotJob& job, std::int64_t from_us) {
  media::ShotProbeOptions options = work.probe;
  if (!work.cache_dir.empty() && !job.cache->read.load(std::memory_order_acquire)) {
    options.preload = [state = job.cache, dir = work.cache_dir, file = job.file,
                       variant = job.variant](media::ShotTimeline& timeline) {
      state->key = media::shot_cache_key(file);
      bool complete = false;
      if (auto cached = media::load_shot_cache(dir, state->key, variant)) {
        for (const media::Shot& shot : cached->shots) timeline.add(shot);
        for (const media::FlashSpan& span : cached->flashes) timeline.add_flash(span);
        complete = cached->complete;
        if (complete)
          log_channel("app")->info("plans de {} lus du cache : {} plans", core::to_utf8(file.filename()),
                                   cached->shots.size());
      }
      state->complete.store(complete, std::memory_order_release);
      state->read.store(true, std::memory_order_release);
      return complete;
    };
  }
  job.probe = media::ShotProbe::start(job.file, *job.timeline, from_us, std::move(options));
}

ShotJob new_job(const ShotWork& work, const std::filesystem::path& file) {
  ShotJob job;
  job.cache_dir = work.cache_dir;
  job.file = file;
  job.cache = std::make_shared<ShotJob::Cache>();
  job.variant = work.probe.cuvid_resize.empty() ? 0 : 1;
  job.timeline = std::make_unique<media::ShotTimeline>();
  return job;
}

// Analysée jusqu'au bout sans trou : au cache.
void keep_if_done(const ShotWork& work, ShotJob& job) {
  if (job.complete || !job.probe || !job.probe->finished() || job.probe->failure()) return;
  if (job.cache->complete.load(std::memory_order_acquire)) {  // lue entière du cache : rien à écrire
    job.complete = true;
    return;
  }
  const media::ShotTimeline& timeline = *job.timeline;
  if (timeline.size() == 0 || timeline.first_gap_us() < timeline.end_us()) return;
  job.complete = true;
  if (work.cache_dir.empty() || !job.cache->read.load(std::memory_order_acquire)) return;
  const auto save = [dir = work.cache_dir, key = job.cache->key, name = core::to_utf8(job.file.filename()),
                     shots = timeline.shots(), flashes = timeline.flashes(),
                     variant = job.variant]() -> std::expected<void, Refusal> {
    const auto log = log_channel("app");
    if (media::save_shot_cache(dir, key, shots, true, media::kShotCacheKeep, flashes, variant))
      log->info("plans de {} gardés : {} plans", name, shots.size());
    return {};
  };
  if (work.writer != nullptr) {
    work.writer->post("shots:" + job.cache->key, save);
  } else {
    std::ignore = save();
  }
}

}  // namespace

ShotJob& ShotJob::operator=(ShotJob&& other) noexcept {
  if (this == &other) return *this;
  retire();
  cache_dir = std::move(other.cache_dir);
  file = std::move(other.file);
  cache = std::move(other.cache);
  timeline = std::move(other.timeline);
  probe = std::move(other.probe);
  variant = other.variant;
  complete = other.complete;
  return *this;
}

ShotJob::~ShotJob() {
  retire();
}

void ShotJob::retire() noexcept {
  probe.reset();  // le thread d'analyse s'arrête avant la lecture de la chronologie
  if (!cache || !cache->read.load(std::memory_order_acquire)) return;
  if (complete || cache->complete.load(std::memory_order_acquire)) return;
  if (!timeline || timeline->size() == 0 || cache_dir.empty() || cache->key.empty()) return;
  try {
    if (media::save_shot_cache(cache_dir, cache->key, timeline->shots(), false, media::kShotCacheKeep,
                               timeline->flashes(), variant)) {
      const auto log = log_channel("app");
      log->info("plans de {} gardés en partie : {} plans", core::to_utf8(file.filename()), timeline->size());
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch) : le cache est un confort, pas une obligation
  }
  timeline.reset();  // gardée une fois
}

void open_shots(ShotWork& work, const std::filesystem::path& file, const PlaySettings& settings,
                std::int64_t start_us, bool source_hdr) {
  if (file.empty() || !wants_shots(work.shot_cuts, settings, source_hdr)) {
    work.current.probe.reset();
    return;
  }
  const std::int64_t from = std::max<std::int64_t>(start_us, 0);
  if (work.current.file == file && work.current.timeline) {
    if (work.current.complete) return;
    if (!work.current.probe) {  // arrêtée quand plus rien ne lisait les plans : elle reprend
      start_probe(work, work.current, from);
    } else if (!work.current.probe->failure()) {
      work.current.probe->seek(from);
    }
    return;
  }
  work.current.retire();  // l'analyse de l'autre fichier s'arrête, sa part faite va au cache
  work.current = new_job(work, file);
  start_probe(work, work.current, from);
}

ShotWork shot_work(std::filesystem::path cache_dir, const PlayOptions& options) {
  ShotWork work{.cache_dir = std::move(cache_dir)};
  if (options.probe_nvdec) work.probe.cuvid_resize = media::kProbeCuvidSize;  // 8.8
  work.shot_cuts = options.shot_cuts;
  // La pré-analyse marque les flashs à l'échelle du grade sous son pic par défaut ; l'analyse à l'ingestion,
  // qui suit le pic réel, reste le secours.
  work.probe.flash_gain = flash_gain_of(render::grade_limits::kDefaultPeakNits);
  return work;
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
  return scene_of(live->stats, kLiveSerialBit | live->serial, false);
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

bool next_flashing(Chain& to) {
  const auto pts = next_pts(to);
  if (!pts) return false;
  // La pré-analyse d'abord : elle a vu la séquence entière et la marque dès sa première transition (A42-17).
  // L'analyse à l'ingestion reste le secours, avec le retard de ce qu'elle a déjà montré.
  if (to.shots != nullptr && to.shots->current.timeline && to.shots->current.file == to.file &&
      to.shots->current.timeline->flashing(*pts))
    return true;
  const auto live = to.live.scenes().at(*pts);
  return live && live->flashing;
}

void feed_plan_shots(Chain& to) {
  if (to.shots == nullptr || !to.shots->current.timeline || to.shots->current.file != to.file) return;
  const media::ShotTimeline& timeline = *to.shots->current.timeline;
  // La version, pas le nombre (A42-11) : un plan entier qui en remplace un partiel ne change pas le compte.
  if (timeline.version() == to.plan_shots_seen) return;
  to.plan_shots_seen = timeline.version();
  std::vector<media::Shot> shots = timeline.shots();
  to.anime_series_apl = series_apl_of(shots);  // aussi pour les noirs profonds (P3.52)
  if (to.shot_cuts) to.live.cuts().set_shots(shots);
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
  // Le grade s'est rallumé après l'ouverture (le panneau) : l'analyse arrêtée à l'ouverture reprend ici,
  // sinon seul ce qu'elle avait vu en partant serait connu. Coupé au panneau sans que l'interpolation lise
  // les plans, elle ne repart pas (A42-09).
  if (!job.probe && !job.complete && view.position_us != kNoPts &&
      wants_shots(to.shot_cuts, to.running, to.source_hdr))
    start_probe(*to.shots, job, std::max<std::int64_t>(view.position_us, 0));
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
  ShotScene scene = scene_of(shot->stats, shot->serial, shot->fade);
  scene.planned = true;
  return scene;
}

}  // namespace ajp::app::play
