#include "app/play/chain_shots.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <expected>
#include <tuple>
#include <utility>
#include <vector>

#include "app/play/chain.h"
#include "app/play/host_facts.h"
#include "app/play/latest_writer.h"
#include "app/play/options.h"
#include "app/play/pumped_task.h"
#include "core/log.h"
#include "core/utf8_path.h"
#include "media/shot_cache.h"

namespace ajp::app::play {
namespace {

constexpr std::int64_t kSeekBehindUs = 2'000'000;  // la lecture revient derrière l'analyse : saut
constexpr std::int64_t kSeekAheadUs = 5'000'000;   // ou la dépasse de loin
constexpr std::int64_t kSeekEveryUs = 1'000'000;

// Un échec d'écriture du cache des plans (dossier en lecture seule, disque plein) : dit une fois par séance,
// sinon chaque vidéo se réanalyse à chaque ouverture sans une ligne au journal (A42-15).
void note_cache_failure(const std::filesystem::path& dir) {
  static std::atomic<bool> said{false};
  if (said.exchange(true)) return;
  log_channel("app")->warn("plans : le cache {} ne s'écrit pas, chaque vidéo sera réanalysée",
                           core::to_utf8(dir.filename()));
}

// Le grade lit les plans (sauf celui, neutre, d'une source HDR sans grade natif) ; l'interpolation aussi,
// sous --shot-cuts. B42-02 : en sortie HDR, le grade efface l'exposition par plan et la marge selon la scène
// (render::steady_hdr) ; il n'y lit plus que les flashs dangereux et la peau des visages (face_skin). Sans
// l'un ni l'autre, la pré-analyse (un second décodage logiciel de tout le fichier) ne part pas : la série des
// noirs profonds reste alors à sa clé moyenne.
bool wants_shots(bool shot_cuts, const PlaySettings& settings, bool source_hdr, bool hdr_output) {
  const bool grade = settings.grade && (!source_hdr || settings.hdr_native) &&
                     (!hdr_output || settings.flash_guard || settings.face_skin);
  return grade || (shot_cuts && settings.rife_factor != 0);
}

// La variante du cache (media/shot_cache.h) : le chemin de décodage, et le gain du limiteur de flashs avec
// lequel les séquences de flashs ont été marquées (B42-07) ; un autre écran les juge à nouveau.
// La recherche des visages (face_skin) en fait une autre : un cache sans visages se refait quand elle
// s'allume.
std::uint32_t cache_variant(bool nvdec, double flash_gain, bool faces) {
  const auto gain = static_cast<std::uint32_t>(std::lround(std::clamp(flash_gain, 0.0, 100.0) * 100.0));
  return (nvdec ? 1U : 0U) + (gain << 1U) + (faces ? 1U << 30U : 0U);
}

// Le thread de l'analyse garde au cache la part faite quand elle s'arrête (B42-05), sauf si la vidéo y est
// déjà entière.
void keep_on_stop(const std::shared_ptr<ShotJob::Cache>& state, const std::filesystem::path& dir,
                  const std::string& name, std::uint32_t variant, const media::ShotTimeline& timeline,
                  bool complete) {
  if (dir.empty() || !state->read.load(std::memory_order_acquire)) return;
  if (state->complete.load(std::memory_order_acquire) || state->kept.load(std::memory_order_acquire)) return;
  if (timeline.size() == 0 || state->key.empty()) return;
  try {
    if (media::save_shot_cache(dir, state->key, timeline.shots(), complete, media::kShotCacheKeep,
                               timeline.flashes(), variant)) {
      log_channel("app")->info("plans de {} gardés{} : {} plans", name, complete ? "" : " en partie",
                               timeline.size());
    } else {
      note_cache_failure(dir);
    }
  } catch (...) {  // NOLINT(bugprone-empty-catch) : le cache est un confort, pas une obligation
  }
}

// L'analyse de la vidéo depuis from_us. Son thread lit d'abord le cache : la chronologie entière, sans
// analyse, ou la part déjà faite, que l'analyse complète.
void start_probe(const ShotWork& work, ShotJob& job, std::int64_t from_us) {
  media::ShotProbeOptions options = work.probe;
  options.flash_gain = job.flash_gain;
  options.faces = job.faces;
  // NVDEC réduit : à 256 × 144, un visage de 24 lignes ferait le sixième de l'image ; la recherche des
  // visages décode plus grand.
  if (job.faces && !options.cuvid_resize.empty()) options.cuvid_resize = media::kProbeCuvidFacesSize;
  // B30-01 : une analyse bloquée sur un partage muet ne fige pas la fenêtre à la fermeture.
  options.abandon = [](std::jthread thread, std::function<bool()> finished) {
    abandon_thread(std::move(thread), std::move(finished));
  };
  options.on_stop = [state = job.cache, dir = work.cache_dir, name = core::to_utf8(job.file.filename()),
                     variant = job.variant](const media::ShotTimeline& timeline, bool complete) {
    keep_on_stop(state, dir, name, variant, timeline, complete);
  };
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
  job.probe = media::ShotProbe::start(job.file, job.timeline, from_us, std::move(options));
}

// B42-07 : les flashs se jugent à la lumière que l'écran montrera (son pic), et sans gain pour une source
// déjà en PQ ou HLG, que le grade ne relève pas.
ShotJob new_job(const ShotWork& work, const std::filesystem::path& file, const ShotOutput& output,
                bool faces) {
  ShotJob job;
  job.cache_dir = work.cache_dir;
  job.file = file;
  job.cache = std::make_shared<ShotJob::Cache>();
  job.flash_gain = output.source_hdr ? 1.0 : flash_gain_of(output.peak_nits);
  job.faces = faces;
  job.variant = cache_variant(!work.probe.cuvid_resize.empty(), job.flash_gain, faces);
  job.timeline = std::make_shared<media::ShotTimeline>();
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
    else
      note_cache_failure(dir);
    return {};
  };
  job.cache->kept.store(true, std::memory_order_release);  // l'arrêt de l'analyse n'écrit plus sa part
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
  flash_gain = other.flash_gain;
  faces = other.faces;
  complete = other.complete;
  return *this;
}

ShotJob::~ShotJob() {
  retire();
}

void ShotJob::retire() noexcept {
  // Le thread de l'analyse s'arrête (au plus deux secondes, puis laissé à son sort : B30-01) et garde au
  // cache sa part faite, de son côté (B42-05).
  probe.reset();
}

void open_shots(ShotWork& work, const std::filesystem::path& file, const PlaySettings& settings,
                std::int64_t start_us, const ShotOutput& output) {
  if (file.empty() || !wants_shots(work.shot_cuts, settings, output.source_hdr, output.hdr_output)) {
    work.current.probe.reset();
    return;
  }
  const std::int64_t from = std::max<std::int64_t>(start_us, 0);
  // La recherche des visages allumée ou éteinte : l'analyse recommence, sous une autre variante du cache.
  if (work.current.file == file && work.current.timeline && work.current.faces == settings.face_skin) {
    if (work.current.complete) return;
    if (!work.current.probe) {  // arrêtée quand plus rien ne lisait les plans : elle reprend
      start_probe(work, work.current, from);
    } else if (!work.current.probe->failure()) {
      work.current.probe->seek(from);
    }
    return;
  }
  work.current.retire();  // l'analyse de l'autre fichier s'arrête, sa part faite va au cache
  work.current = new_job(work, file, output, settings.face_skin);
  start_probe(work, work.current, from);
}

ShotWork shot_work(std::filesystem::path cache_dir, const PlayOptions& options) {
  ShotWork work{.cache_dir = std::move(cache_dir)};
  if (options.probe_nvdec) work.probe.cuvid_resize = media::kProbeCuvidSize;  // 8.8
  work.shot_cuts = options.shot_cuts;
  // Le gain du limiteur de flashs de la pré-analyse suit le pic de l'écran de chaque vidéo (new_job, B42-07).
  return work;
}

void tend_shots(ShotWork& work) {
  keep_if_done(work, work.current);
}

namespace {

render::SkinTones skin_tones_of(const media::PlanSkins& skins) {
  render::SkinTones out;
  for (const media::SkinTone& t : skins.view()) {
    out.hue.at(static_cast<std::size_t>(out.count)) = t.hue;
    out.sat.at(static_cast<std::size_t>(out.count)) = t.sat;
    ++out.count;
  }
  return out;
}

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
  const bool wanted = wants_shots(to.shot_cuts, to.running, to.source_hdr, to.screen.hdr_output());
  if (!job.probe && !job.complete && view.position_us != kNoPts && wanted)
    start_probe(*to.shots, job, std::max<std::int64_t>(view.position_us, 0));
  // Plus rien ne lit les plans (sortie passée en HDR, limiteur décoché) : l'analyse s'arrête (B42-02).
  if (job.probe && !wanted) job.probe.reset();
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
  scene.start_us = shot->start_us;
  scene.end_us = shot->end_us;
  scene.cut_before = shot->cut_before;
  scene.skins = skin_tones_of(shot->skins);
  return scene;
}

void open_shots_early(ShotWork& work, const std::filesystem::path& file, PlaySettings settings,
                      std::int64_t start_us, const media::TrackInfo& video, std::optional<double> peak_seen,
                      bool sdr) {
  const std::optional<ColorInfo> color = media::track_color(video);
  const bool source_hdr = color && (color->transfer == Transfer::Pq || color->transfer == Transfer::Hlg);
  const int peak = settings.true_hdr_peak > 0 ? settings.true_hdr_peak : auto_peak_nits(peak_seen);
  const bool hdr_output = !sdr && system_hdr(peak_seen);
  settings.hdr_native = hdr_output;
  open_shots(
      work, file, settings, start_us,
      ShotOutput{.source_hdr = source_hdr, .hdr_output = hdr_output, .peak_nits = static_cast<double>(peak)});
}

}  // namespace ajp::app::play
