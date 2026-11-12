#include "media/shot_probe.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <utility>
#include <vector>

#include "core/thread.h"
#include "core/thread_role.h"
#include "media/demuxer.h"
#include "media/face_skin.h"
#include "media/flash_guard.h"
#include "media/log.h"
#include "media/shot_probe_steps.h"
#include "media/video_decoder.h"

namespace ajp::media {

namespace {

// La découpe et le limiteur de flashs d'une analyse, qui écrivent dans la chronologie. Les flashs dangereux
// vus d'avance (A42-17) : le plafond du grade les prend dès leur première transition, au lieu d'attendre que
// la lecture en ait montré une seconde.
class Analysis {
 public:
  Analysis(ShotTimeline& timeline, double flash_gain, bool starts_at_cut)
      : timeline_(timeline), cutter_(starts_at_cut) {
    flashes_.set_gain(flash_gain);
  }
  void push(const ShotFrame& frame) {
    if (flashes_.push(frame.pts_us, zone_nits_of(frame))) {
      if (const auto from = flashes_.danger_from())
        timeline_.add_flash({.from_us = *from, .until_us = flashes_.limited_until()});
    }
    if (auto shot = cutter_.push(frame)) timeline_.add(*shot);
  }
  // Le plan en cours est clos à end_us.
  void finish(std::int64_t end_us) {
    if (auto shot = cutter_.finish(end_us)) timeline_.add(*shot);
  }
  // Un nouveau découpage : son premier plan n'a de vraie coupe devant lui qu'au début de la vidéo.
  void restart(bool starts_at_cut) {
    cutter_ = ShotCutter(starts_at_cut);
    flashes_.reset();
  }

 private:
  ShotTimeline& timeline_;
  ShotCutter cutter_;
  FlashGuard flashes_;
};

}  // namespace

namespace {

// Le fichier ouvert pour l'analyse : sa piste vidéo seule, et son décodeur logiciel.
struct Opened {
  Demuxer demuxer;
  int track = 0;
  VideoDecoder decoder;
};

std::expected<Opened, Refusal> open_video(const std::filesystem::path& file, const std::stop_token& stop,
                                          const ShotProbeOptions& options) {
  // B30-01 : l'arrêt atteint aussi l'ouverture du fichier, sur un partage lent.
  auto demuxer = Demuxer::open(file, {.attachments = false}, stop);
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const auto video = std::ranges::find_if(demuxer->tracks(), [](const TrackInfo& track) {
    return track.kind == TrackKind::Video && !track.attached_picture;
  });
  if (video == demuxer->tracks().end())
    return std::unexpected(Refusal{RefusalCode::TrackUnavailable, "aucune piste vidéo"});
  const int track = video->index;
  const std::array<int, 1> keep{track};
  if (auto ok = demuxer->keep_only(keep); !ok) return std::unexpected(std::move(ok.error()));
  const int threads = options.threads > 0 ? options.threads : background_decoder_threads();
  auto decoder = VideoDecoder::open(
      *video, DecoderOptions{
                  .attempts = {HwApi::Software}, .threads = threads, .cuvid_resize = options.cuvid_resize});
  if (!decoder) return std::unexpected(std::move(decoder.error()));
  return Opened{.demuxer = std::move(*demuxer), .track = track, .decoder = std::move(*decoder)};
}

}  // namespace

// L'état que l'analyse partage avec son thread : il survit à la ShotProbe quand le thread est abandonné
// (B30-01).
struct ShotProbe::State {
  State(std::filesystem::path f, std::shared_ptr<ShotTimeline> t, ShotProbeOptions o)
      : file(std::move(f)), timeline(std::move(t)), options(std::move(o)) {}

  std::filesystem::path file;
  std::shared_ptr<ShotTimeline> timeline;
  ShotProbeOptions options;
  std::atomic<std::int64_t> playhead{0};
  std::atomic<std::int64_t> seek_to{-1};
  std::atomic<std::int64_t> frontier{0};
  std::atomic<bool> finished{false};
  // P2.37 : l'attente de la pré-analyse (fin atteinte, assez d'avance) se réveille au saut ou quand la
  // lecture avance, au lieu de dormir 100 ms par tour.
  std::mutex wake_mutex;
  std::condition_variable_any wake;
  mutable std::mutex failure_mutex;
  std::optional<Refusal> failure;
  // Sous stop_mutex : l'analyse est finie (on_stop peut encore écrire le cache), puis le thread a rendu la
  // main. La destruction attend la première au plus stop_limit (C42-06).
  std::mutex stop_mutex;
  std::condition_variable stop_cv;
  bool analysis_done = false;
  bool stopped = false;

  std::int64_t frontier_us() const { return frontier.load(std::memory_order_relaxed); }
  bool failure_any() const {
    const std::scoped_lock lock(failure_mutex);
    return failure.has_value();
  }
  void fail(Refusal refusal);
  bool preloaded();
  void run(const std::stop_token& stop, std::int64_t start_us);
  bool wait_turn(const std::stop_token& stop, bool ended, ProbePace& pace);
  void mark_analysis_done();
  void mark_stopped();
  bool wait_analysis_done(std::chrono::milliseconds limit);
  bool is_stopped();
};

void ShotProbe::State::fail(Refusal refusal) {
  make_logger()->warn("pré-analyse des plans arrêtée : {}", refusal.detail);
  {
    const std::scoped_lock lock(failure_mutex);
    failure = std::move(refusal);
  }
  finished.store(true, std::memory_order_release);
}

// La part déjà connue de la vidéo (le cache), lue ici plutôt que sur le thread de la fenêtre. Vrai : connue
// entière, rien à analyser.
bool ShotProbe::State::preloaded() {
  if (!options.preload || !options.preload(*timeline)) return false;
  finished.store(true, std::memory_order_release);
  return true;
}

void ShotProbe::State::mark_analysis_done() {
  {
    const std::scoped_lock lock(stop_mutex);
    analysis_done = true;
  }
  stop_cv.notify_all();
}

void ShotProbe::State::mark_stopped() {
  {
    const std::scoped_lock lock(stop_mutex);
    analysis_done = true;
    stopped = true;
  }
  stop_cv.notify_all();
}

bool ShotProbe::State::wait_analysis_done(std::chrono::milliseconds limit) {
  std::unique_lock lock(stop_mutex);
  return stop_cv.wait_for(lock, limit, [this] { return analysis_done; });
}

bool ShotProbe::State::is_stopped() {
  const std::scoped_lock lock(stop_mutex);
  return stopped;
}

ShotProbe::ShotProbe(std::shared_ptr<State> state) : state_(std::move(state)) {}

std::unique_ptr<ShotProbe> ShotProbe::start(std::filesystem::path file,
                                            std::shared_ptr<ShotTimeline> timeline, std::int64_t start_us,
                                            ShotProbeOptions options) {
  auto state = std::make_shared<State>(std::move(file), std::move(timeline), std::move(options));
  state->playhead = start_us;
  state->frontier = start_us;
  std::unique_ptr<ShotProbe> probe(new ShotProbe(state));
  // Un refus (une exception du corps comprise) arrête l'analyse, jamais le lecteur : la lecture garde alors
  // la mesure en direct. Le thread ne tient que l'état partagé : abandonné, il ne touche plus à la ShotProbe.
  auto thread = start_thread("plans", ThreadReport([state](const Refusal& refusal) { state->fail(refusal); }),
                             [state, start_us](const std::stop_token& stop, const ThreadReport& /*report*/) {
                               // Rendu la main, même sur une exception : la destruction ne l'attend plus.
                               struct Stopped {
                                 explicit Stopped(State& s) : state(s) {}
                                 State& state;
                                 Stopped(const Stopped&) = delete;
                                 Stopped& operator=(const Stopped&) = delete;
                                 Stopped(Stopped&&) = delete;
                                 Stopped& operator=(Stopped&&) = delete;
                                 ~Stopped() { state.mark_stopped(); }
                               } const stopped(*state);
                               // Travail de fond : il ne doit jamais retarder le rendu, le calcul ni l'audio.
                               // Sous Linux, les threads de FFmpeg créés ensuite héritent de cette priorité.
                               set_thread_role(ThreadRole::Background);
                               if (state->preloaded()) return;
                               state->run(stop, start_us);
                               // La part faite, au cache (B42-05) : ici plutôt que sur le thread de la
                               // fenêtre, qui n'attend plus que la fin de l'analyse (C42-06).
                               state->mark_analysis_done();
                               if (state->options.on_stop) {
                                 const ShotTimeline& done = *state->timeline;
                                 const bool whole = state->finished.load(std::memory_order_acquire) &&
                                                    !state->failure_any() && done.size() > 0 &&
                                                    done.first_gap_us() >= done.end_us();
                                 state->options.on_stop(done, whole);
                               }
                             });
  if (thread) {
    probe->thread_ = std::move(*thread);
  } else {
    state->fail(std::move(thread.error()));
    state->mark_stopped();
  }
  return probe;
}

ShotProbe::~ShotProbe() {
  if (!thread_.joinable()) return;
  thread_.request_stop();
  state_->wake.notify_all();
  // B30-01 : une lecture suspendue dans le noyau (partage réseau muet) ne rend pas la main au drapeau
  // d'interruption ; passé stop_limit, le thread est confié à abandon au lieu de figer la fenêtre.
  if (!state_->options.abandon) return;  // join
  const bool done = state_->wait_analysis_done(state_->options.stop_limit);
  if (state_->is_stopped()) return;  // join : fini
  // C42-06 : l'analyse est finie, son thread écrit le cache (on_stop) ; la fenêtre ne l'attend pas.
  if (!done)
    make_logger()->warn("pré-analyse des plans : pas d'arrêt en {} ms, laissée à son sort",
                        state_->options.stop_limit.count());
  state_->options.abandon(std::move(thread_), [state = state_] { return state->is_stopped(); });
}

void ShotProbe::set_playhead(std::int64_t us) {
  state_->playhead.store(us, std::memory_order_relaxed);
  state_->wake.notify_one();  // sans verrou : un réveil manqué attend au plus le délai de l'attente
}

void ShotProbe::seek(std::int64_t at_us) {
  {
    const std::scoped_lock lock(state_->wake_mutex);
    state_->playhead.store(at_us, std::memory_order_relaxed);
    state_->seek_to.store(at_us, std::memory_order_relaxed);
  }
  state_->wake.notify_one();
}

bool ShotProbe::finished() const {
  return state_->finished.load(std::memory_order_acquire);
}

std::optional<Refusal> ShotProbe::failure() const {
  const std::scoped_lock lock(state_->failure_mutex);
  return state_->failure;
}

std::int64_t ShotProbe::frontier_us() const {
  return state_->frontier_us();
}

namespace {

// Le fichier ouvert pour l'analyse, et l'interruption de ses lectures à l'arrêt (une lecture bloquée du
// démuxeur, sur un partage lent). Détruite avant lui, l'interruption ne vise jamais un démuxeur fermé.
struct OpenedFile {
  explicit OpenedFile(Opened o) : opened(std::move(o)) {}
  Opened opened;
  std::optional<std::stop_callback<std::function<void()>>> interrupt;
};

std::expected<std::unique_ptr<OpenedFile>, Refusal> open_file(const std::filesystem::path& file,
                                                              const std::stop_token& stop,
                                                              const ShotProbeOptions& options) {
  auto opened = open_video(file, stop, options);
  if (!opened) return std::unexpected(std::move(opened.error()));
  auto out = std::make_unique<OpenedFile>(std::move(*opened));
  Demuxer* demuxer = &out->opened.demuxer;
  out->interrupt.emplace(stop, std::function<void()>([demuxer] { demuxer->interrupt(); }));
  return out;
}

std::int64_t steady_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

// L'attente d'un tour. Vrai : le tour a attendu, la boucle reprend au début (un saut a pu arriver).
bool ShotProbe::State::wait_turn(const std::stop_token& stop, bool ended, ProbePace& pace) {
  const auto seek_asked = [this] { return seek_to.load(std::memory_order_relaxed) >= 0; };
  const auto lead = [this] { return frontier_us() - playhead.load(std::memory_order_relaxed); };
  // Fin atteinte : attendre un saut, sans tour de garde (seek pose seek_to sous wake_mutex).
  if (ended) {
    std::unique_lock lock(wake_mutex);
    wake.wait(lock, stop, seek_asked);
    return true;
  }
  // Assez d'avance : attendre un saut, ou que la lecture approche.
  if (lead() > options.lead_us) {
    std::unique_lock lock(wake_mutex);
    wake.wait_for(lock, stop, std::chrono::milliseconds(500),
                  [&] { return seek_asked() || lead() <= options.lead_us; });
    return true;
  }
  // L'avance prise : au rythme de options.pace (C42-01).
  const std::int64_t delay = pace.delay_us(frontier_us(), lead(), steady_us());
  if (delay <= 0) return false;
  std::unique_lock lock(wake_mutex);
  wake.wait_for(lock, stop, std::chrono::microseconds(std::min<std::int64_t>(delay, 500'000)), seek_asked);
  return true;
}

// Le journal donne les threads du décodeur ouvert, pas ceux demandés : un seul sous NVDEC réduit (A42-06).
// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
void ShotProbe::State::run(const std::stop_token& stop, std::int64_t start_us) {
  auto log = make_logger();
  auto first = open_file(file, stop, options);
  if (!first) return fail(std::move(first.error()));
  // C42-07 : fermé à la fin de l'analyse, rouvert au prochain saut qui en demande une.
  std::unique_ptr<OpenedFile> open = std::move(*first);
  const int track = open->opened.track;

  // Repart de at_us : le plan en cours est clos là, un nouveau découpage commence (son premier plan n'a pas
  // de vraie coupe devant lui).
  Analysis analysis(*timeline, options.flash_gain, start_us <= 0);
  std::int64_t last_pts = start_us;
  std::int64_t last_duration = 0;
  // B42-04 : l'instant de la première image d'un découpage parti du début va à la chronologie.
  bool from_origin = start_us <= 0;
  // La fin du fichier (B42-09) : la durée du conteneur, puis la fin atteinte.
  std::int64_t file_end_us = open->opened.demuxer.duration_us();
  // Le bout d'une zone connue sautée (le saut retombe sur l'image clé d'avant : ce garde l'empêche de se
  // répéter) ; C42-02 : remis à zéro à tout autre départ, sinon un second passage la décodait en entier.
  std::int64_t skipped_to = -1;
  const auto restart = [&](std::int64_t at_us) -> bool {
    analysis.finish(last_pts + last_duration);
    Demuxer& demuxer = open->opened.demuxer;
    // La fin d'un plan peut dépasser d'une image la durée du conteneur, que le démuxeur refuse.
    const std::int64_t duration = demuxer.duration_us();
    const std::int64_t target = duration > 0 ? std::min(at_us, duration) : at_us;
    if (!demuxer.seek(std::max<std::int64_t>(target, 0), track, SeekTarget::PreviousKeyframe)) return false;
    open->opened.decoder.flush();
    analysis.restart(at_us <= 0);
    from_origin = at_us <= 0;
    skipped_to = -1;
    finished.store(false, std::memory_order_release);
    frontier.store(at_us, std::memory_order_relaxed);
    last_pts = at_us;
    return true;
  };
  if (start_us > 0 && !restart(start_us))
    return fail(Refusal{RefusalCode::SeekFailed, "saut de la pré-analyse"});
  log->info("pré-analyse des plans : {} thread(s) de décodage, depuis {:.1f} s",
            open->opened.decoder.threads(), static_cast<double>(start_us) / 1e6);

  const auto started = std::chrono::steady_clock::now();
  std::int64_t frames = 0;
  FaceSearch faces(options.faces);
  ProbePace pace(options.pace, options.pace_after_us);
  std::int64_t filled_from = -1;
  bool ended = false;
  while (!stop.stop_requested()) {
    if (const std::int64_t to = seek_to.exchange(-1, std::memory_order_relaxed); to >= 0) {
      if (const auto from =
              restart_for_seek(*timeline, to, frontier_us(), file_end_us, options.skip_known_us)) {
        if (!open) {
          auto reopened = open_file(file, stop, options);
          if (!reopened) return fail(std::move(reopened.error()));
          open = std::move(*reopened);
        }
        if (!restart(*from)) return fail(Refusal{RefusalCode::SeekFailed, "saut de la pré-analyse"});
        ended = false;
      }
    }
    if (wait_turn(stop, ended, pace)) continue;
    // Une zone déjà analysée devant (après un retour en arrière) : sautée.
    if (const std::int64_t known = timeline->analyzed_until(last_pts + 1);
        frames > 0 && known - last_pts > options.skip_known_us && known > skipped_to) {
      if (!restart(known)) return fail(Refusal{RefusalCode::SeekFailed, "saut de la pré-analyse"});
      skipped_to = known;
      continue;
    }
    Demuxer& demuxer = open->opened.demuxer;
    VideoDecoder& decoder = open->opened.decoder;
    auto received = decoder.receive();
    if (!received) return fail(std::move(received.error()));
    if (*received) {
      const DecodedFrame& frame = **received;
      if (frame.pts_us == kNoPts) continue;
      auto measured = measure_frame(frame);
      if (!measured) return fail(std::move(measured.error()));
      if (from_origin) {
        timeline->set_origin_us(frame.pts_us);
        from_origin = false;
      }
      last_pts = frame.pts_us;
      last_duration = frame.duration_us;
      frontier.store(frame.pts_us, std::memory_order_relaxed);
      faces.look(frame, *measured);
      ++frames;
      analysis.push(*measured);
      continue;
    }
    if (decoder.drained()) {
      analysis.finish(last_pts + last_duration);
      file_end_us = std::max(file_end_us, last_pts + last_duration);
      // Un départ en cours de vidéo (une reprise, un saut) : le début reste à analyser, pour que la vidéo
      // soit connue entière (le cache, media/shot_cache.h). Chaque trou n'est repris qu'une fois.
      if (const std::int64_t gap = timeline->first_gap_us(); gap < timeline->end_us() && gap > filled_from) {
        filled_from = gap;
        if (!restart(gap)) return fail(Refusal{RefusalCode::SeekFailed, "saut de la pré-analyse"});
        continue;
      }
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      log->info("pré-analyse des plans finie : {} images en {:.1f} s, {} plans{}", frames, seconds,
                timeline->size(), faces.summary(*timeline));
      finished.store(true, std::memory_order_release);
      ended = true;
      open.reset();
      continue;
    }
    auto packet = demuxer.read();
    if (!packet) return fail(std::move(packet.error()));
    if (!*packet) {
      if (auto ok = decoder.send_end(); !ok) return fail(std::move(ok.error()));
    } else if ((*packet)->track == track) {
      if (auto ok = decoder.send(**packet); !ok) return fail(std::move(ok.error()));
    }
  }
}

}  // namespace ajp::media
