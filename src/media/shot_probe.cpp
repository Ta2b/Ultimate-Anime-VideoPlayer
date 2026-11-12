#include "media/shot_probe.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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
#include "kernels/ref/frame_stats.h"
#include "media/demuxer.h"
#include "media/face_skin.h"
#include "media/flash_guard.h"
#include "media/log.h"
#include "media/shot_probe_steps.h"
#include "media/video_decoder.h"
#include "media/yuv_sample.h"

namespace ajp::media {
namespace {

// Les formats que la pré-analyse lit : ceux de yuv::layout_of (4:2:0), plus P016 (NVDEC en 12 bits) et
// Yuv444p16 (B42-10). Le décodeur logiciel de l'analyse ramène déjà le 4:2:2, le 4:4:4 et le 12 bits au 4:2:0
// (frame_data.cpp, to_chain_yuv). chroma_step : 2 en 4:2:0, 1 en 4:4:4.
struct ProbeLayout {
  yuv::Layout layout;
  int chroma_step = 2;
};

std::optional<ProbeLayout> probe_layout(PixelFormat format) {
  if (const std::optional<yuv::Layout> layout = yuv::layout_of(format)) return ProbeLayout{.layout = *layout};
  if (format == PixelFormat::P016)
    return ProbeLayout{.layout = {.bits = 16, .semi_planar = true, .high = true, .shift = 0}};
  if (format == PixelFormat::Yuv444p16)
    return ProbeLayout{.layout = {.bits = 16, .high = true}, .chroma_step = 1};
  return std::nullopt;
}

}  // namespace

std::expected<ShotFrame, Refusal> measure_frame(const DecodedFrame& frame) {
  namespace fs = kernels::frame_stats;
  const std::optional<ProbeLayout> probe = probe_layout(frame.format);
  if (!probe) return refuse(RefusalCode::FormatUnsupported, "format de pixel non pris par la pré-analyse");
  const yuv::Layout* layout = &probe->layout;
  const int step = probe->chroma_step;
  if (frame.data == nullptr || frame.data->memory() != FrameMemory::Host)
    return refuse(RefusalCode::FormatUnsupported, "image décodée hors de la mémoire hôte");
  if (frame.width < 2 || frame.height < 2) return refuse(RefusalCode::FormatUnsupported, "image trop petite");
  const bool bt601 = frame.color.matrix == ColorMatrix::Bt601;
  const bool bt2020 = frame.color.matrix == ColorMatrix::Bt2020Ncl;
  const double kr = bt601 ? 0.299 : bt2020 ? 0.2627 : 0.2126;
  const double kb = bt601 ? 0.114 : bt2020 ? 0.0593 : 0.0722;
  const double kg = 1.0 - kr - kb;
  const bool full = frame.color.range == ColorRange::Full;
  const double scale = std::ldexp(1.0, layout->bits - 8);
  const double y_off = full ? 0.0 : 16.0 * scale;
  const double y_range = (full ? 255.0 : 219.0) * scale;
  const double c_range = (full ? 255.0 : 224.0) * scale;
  const double c_mid = 128.0 * scale;
  const auto luma_plane = frame.data->host_plane(0);
  const std::size_t luma_stride = frame.data->host_stride(0);
  const auto u_plane = frame.data->host_plane(1);
  const std::size_t u_stride = frame.data->host_stride(1);
  std::span<const std::byte> v_plane;
  std::size_t v_stride = 0;
  if (!layout->semi_planar) {
    v_plane = frame.data->host_plane(2);
    v_stride = frame.data->host_stride(2);
  }
  // Les points de la grille de l'analyse (kernels/spec/frame_stats.h), convertis en R'G'B' : la
  // référence lit alors chaque point de cette image à sa place. Un point par case, pas une moyenne de bloc
  // (B42-11) : c'est le contrat de frame_stats.h, que suit aussi le noyau GPU de l'analyse à l'ingestion ;
  // les deux mesures restent comparables plan par plan.
  kernels::RgbImage rgb(fs::kGridW, fs::kGridH);
  for (int gy = 0; gy < fs::kGridH; ++gy) {
    const int sy = std::min((2 * gy + 1) * frame.height / (2 * fs::kGridH), frame.height - 1);
    for (int gx = 0; gx < fs::kGridW; ++gx) {
      const int sx = std::min((2 * gx + 1) * frame.width / (2 * fs::kGridW), frame.width - 1);
      const double yy = (yuv::sample(luma_plane, luma_stride, sx, sy, *layout) - y_off) / y_range;
      double cb = 0.0, cr = 0.0;
      if (layout->semi_planar) {
        cb = yuv::sample(u_plane, u_stride, sx / step, sy / step, *layout, 0, 2);
        cr = yuv::sample(u_plane, u_stride, sx / step, sy / step, *layout, 1, 2);
      } else {
        cb = yuv::sample(u_plane, u_stride, sx / step, sy / step, *layout);
        cr = yuv::sample(v_plane, v_stride, sx / step, sy / step, *layout);
      }
      cb = (cb - c_mid) / c_range;
      cr = (cr - c_mid) / c_range;
      // V se calcule avant de borner R et B (A42-07) : un canal hors plage le faussait.
      const double r = yy + 2.0 * (1.0 - kr) * cr;
      const double b = yy + 2.0 * (1.0 - kb) * cb;
      const double g = (yy - kr * r - kb * b) / kg;
      rgb.planes[0].at(gx, gy) = static_cast<float>(std::clamp(r, 0.0, 1.0));
      rgb.planes[1].at(gx, gy) = static_cast<float>(std::clamp(g, 0.0, 1.0));
      rgb.planes[2].at(gx, gy) = static_cast<float>(std::clamp(b, 0.0, 1.0));
    }
  }
  const kernels::FrameTransfer transfer = frame.color.transfer == Transfer::Pq ? kernels::FrameTransfer::Pq
                                          : frame.color.transfer == Transfer::Hlg
                                              ? kernels::FrameTransfer::Hlg
                                              : kernels::FrameTransfer::Sdr;
  auto counts = kernels::ref::frame_counts(rgb, transfer);
  if (!counts) return refuse(RefusalCode::FormatUnsupported, counts.error().detail);
  return shot_frame_of(frame.pts_us, kernels::ref::frame_stats_of(*counts), transfer);
}

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

// Le journal donne les threads du décodeur ouvert, pas ceux demandés : un seul sous NVDEC réduit (A42-06).
// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
void ShotProbe::State::run(const std::stop_token& stop, std::int64_t start_us) {
  auto log = make_logger();
  auto first = open_file(file, stop, options);
  if (!first) return fail(std::move(first.error()));
  // C42-07 : fermé quand l'analyse a fini (le fichier, le démuxeur et les tampons du décodeur), rouvert au
  // prochain saut qui demande une analyse.
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
  // Le bout d'une zone connue où l'analyse a sauté : le saut retombe sur l'image clé d'avant, ce garde
  // l'empêche de se répéter. C42-02 : remis à zéro par tout autre départ, sinon un second passage jusqu'à la
  // même zone (le comblement d'un trou plus bas) la décodait en entier.
  std::int64_t skipped_to = -1;
  const auto restart = [&](std::int64_t at_us) -> bool {
    analysis.finish(last_pts + last_duration);
    Demuxer& demuxer = open->opened.demuxer;
    // La fin d'un plan (dernière image plus sa durée) peut dépasser d'une image la durée du conteneur, que le
    // démuxeur refuse : le saut se borne à cette durée.
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
    const auto ahead = [this] {
      return frontier_us() - playhead.load(std::memory_order_relaxed) > options.lead_us;
    };
    // Fin atteinte : attendre un saut, sans tour de garde (seek pose seek_to sous wake_mutex).
    if (ended) {
      std::unique_lock lock(wake_mutex);
      wake.wait(lock, stop, [&] { return seek_to.load(std::memory_order_relaxed) >= 0; });
      continue;
    }
    // Assez d'avance : attendre un saut, ou que la lecture approche.
    if (ahead()) {
      std::unique_lock lock(wake_mutex);
      wake.wait_for(lock, stop, std::chrono::milliseconds(500),
                    [&] { return seek_to.load(std::memory_order_relaxed) >= 0 || !ahead(); });
      continue;
    }
    // L'avance prise : au rythme de options.pace (C42-01).
    if (const std::int64_t delay = pace.delay_us(
            frontier_us(), frontier_us() - playhead.load(std::memory_order_relaxed), steady_us());
        delay > 0) {
      std::unique_lock lock(wake_mutex);
      wake.wait_for(lock, stop, std::chrono::microseconds(std::min<std::int64_t>(delay, 500'000)),
                    [&] { return seek_to.load(std::memory_order_relaxed) >= 0; });
      continue;
    }
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
