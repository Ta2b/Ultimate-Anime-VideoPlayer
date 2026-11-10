#include "media/shot_probe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stop_token>
#include <utility>
#include <vector>

#include "core/thread.h"
#include "core/thread_role.h"
#include "kernels/ref/frame_stats.h"
#include "media/demuxer.h"
#include "media/log.h"
#include "media/video_decoder.h"
#include "media/yuv_sample.h"

namespace ajp::media {
std::expected<ShotFrame, Refusal> measure_frame(const DecodedFrame& frame) {
  namespace fs = kernels::frame_stats;
  const std::optional<yuv::Layout> layout = yuv::layout_of(frame.format);
  if (!layout) return refuse(RefusalCode::FormatUnsupported, "format de pixel non pris par la pré-analyse");
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
  // référence lit alors chaque point de cette image à sa place.
  kernels::RgbImage rgb(fs::kGridW, fs::kGridH);
  for (int gy = 0; gy < fs::kGridH; ++gy) {
    const int sy = std::min((2 * gy + 1) * frame.height / (2 * fs::kGridH), frame.height - 1);
    for (int gx = 0; gx < fs::kGridW; ++gx) {
      const int sx = std::min((2 * gx + 1) * frame.width / (2 * fs::kGridW), frame.width - 1);
      const double yy = (yuv::sample(luma_plane, luma_stride, sx, sy, *layout) - y_off) / y_range;
      double cb = 0.0, cr = 0.0;
      if (layout->semi_planar) {
        cb = yuv::sample(u_plane, u_stride, sx / 2, sy / 2, *layout, 0, 2);
        cr = yuv::sample(u_plane, u_stride, sx / 2, sy / 2, *layout, 1, 2);
      } else {
        cb = yuv::sample(u_plane, u_stride, sx / 2, sy / 2, *layout);
        cr = yuv::sample(v_plane, v_stride, sx / 2, sy / 2, *layout);
      }
      cb = (cb - c_mid) / c_range;
      cr = (cr - c_mid) / c_range;
      const double r = std::clamp(yy + 2.0 * (1.0 - kr) * cr, 0.0, 1.0);
      const double b = std::clamp(yy + 2.0 * (1.0 - kb) * cb, 0.0, 1.0);
      const double g = std::clamp((yy - kr * r - kb * b) / kg, 0.0, 1.0);
      rgb.planes[0].at(gx, gy) = static_cast<float>(r);
      rgb.planes[1].at(gx, gy) = static_cast<float>(g);
      rgb.planes[2].at(gx, gy) = static_cast<float>(b);
    }
  }
  const kernels::FrameTransfer transfer = frame.color.transfer == Transfer::Pq ? kernels::FrameTransfer::Pq
                                          : frame.color.transfer == Transfer::Hlg
                                              ? kernels::FrameTransfer::Hlg
                                              : kernels::FrameTransfer::Sdr;
  auto counts = kernels::ref::frame_counts(rgb, transfer);
  if (!counts) return refuse(RefusalCode::FormatUnsupported, counts.error().detail);
  return shot_frame_of(frame.pts_us, kernels::ref::frame_stats_of(*counts));
}

ShotProbe::ShotProbe(std::filesystem::path file, ShotTimeline& timeline, ShotProbeOptions options)
    : file_(std::move(file)), timeline_(timeline), options_(std::move(options)) {}

std::unique_ptr<ShotProbe> ShotProbe::start(std::filesystem::path file, ShotTimeline& timeline,
                                            std::int64_t start_us, ShotProbeOptions options) {
  std::unique_ptr<ShotProbe> probe(new ShotProbe(std::move(file), timeline, std::move(options)));
  probe->playhead_ = start_us;
  probe->frontier_ = start_us;
  ShotProbe* self = probe.get();
  // Un refus (une exception du corps comprise) arrête l'analyse, jamais le lecteur : la lecture garde alors
  // la mesure en direct.
  auto thread = start_thread("plans", ThreadReport([self](const Refusal& refusal) { self->fail(refusal); }),
                             [self, start_us](const std::stop_token& stop, const ThreadReport& /*report*/) {
                               // Travail de fond : il ne doit jamais retarder le rendu, le calcul ni l'audio.
                               // Sous Linux, les threads de FFmpeg créés ensuite héritent de cette priorité.
                               set_thread_role(ThreadRole::Background);
                               self->run(stop, start_us);
                             });
  if (thread)
    probe->thread_ = std::move(*thread);
  else
    probe->fail(std::move(thread.error()));
  return probe;
}

ShotProbe::~ShotProbe() {
  thread_.request_stop();
}

void ShotProbe::set_playhead(std::int64_t us) {
  playhead_.store(us, std::memory_order_relaxed);
  wake_.notify_one();  // sans verrou : un réveil manqué attend au plus le délai de l'attente
}

void ShotProbe::seek(std::int64_t at_us) {
  {
    const std::scoped_lock lock(wake_mutex_);
    playhead_.store(at_us, std::memory_order_relaxed);
    seek_to_.store(at_us, std::memory_order_relaxed);
  }
  wake_.notify_one();
}

bool ShotProbe::finished() const {
  return finished_.load(std::memory_order_acquire);
}

std::optional<Refusal> ShotProbe::failure() const {
  const std::scoped_lock lock(failure_mutex_);
  return failure_;
}

std::int64_t ShotProbe::frontier_us() const {
  return frontier_.load(std::memory_order_relaxed);
}

void ShotProbe::fail(Refusal refusal) {
  make_logger()->info("pré-analyse des plans arrêtée : {}", refusal.detail);
  {
    const std::scoped_lock lock(failure_mutex_);
    failure_ = std::move(refusal);
  }
  finished_.store(true, std::memory_order_release);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ShotProbe::run(const std::stop_token& stop, std::int64_t start_us) {
  auto log = make_logger();
  auto demuxer = Demuxer::open(file_);
  if (!demuxer) return fail(std::move(demuxer.error()));
  // L'arrêt (fermeture du fichier) interrompt aussi une lecture bloquée du démuxeur, sur un partage lent.
  const std::stop_callback interrupt(stop, [&demuxer] { demuxer->interrupt(); });
  const TrackInfo* video = nullptr;
  for (const TrackInfo& track : demuxer->tracks())
    if (track.kind == TrackKind::Video && !track.attached_picture) {
      video = &track;
      break;
    }
  if (video == nullptr) return fail(Refusal{RefusalCode::TrackUnavailable, "aucune piste vidéo"});
  const int track = video->index;
  const std::array<int, 1> keep{track};
  if (auto ok = demuxer->keep_only(keep); !ok) return fail(std::move(ok.error()));
  const int threads = options_.threads > 0 ? options_.threads : background_decoder_threads();
  auto decoder = VideoDecoder::open(
      *video, DecoderOptions{
                  .attempts = {HwApi::Software}, .threads = threads, .cuvid_resize = options_.cuvid_resize});
  if (!decoder) return fail(std::move(decoder.error()));

  // Repart de at_us : le plan en cours est clos là, un nouveau découpage commence (son premier plan n'a pas
  // de vraie coupe devant lui).
  ShotCutter cutter(start_us <= 0);
  std::int64_t last_pts = start_us;
  std::int64_t last_duration = 0;
  const auto restart = [&](std::int64_t at_us) -> bool {
    if (auto shot = cutter.finish(last_pts + last_duration)) timeline_.add(*shot);
    // La fin d'un plan (dernière image plus sa durée) peut dépasser d'une image la durée du conteneur, que le
    // démuxeur refuse : le saut se borne à cette durée.
    const std::int64_t duration = demuxer->duration_us();
    const std::int64_t target = duration > 0 ? std::min(at_us, duration) : at_us;
    if (!demuxer->seek(std::max<std::int64_t>(target, 0), track, SeekTarget::PreviousKeyframe)) return false;
    decoder->flush();
    cutter = ShotCutter(at_us <= 0);
    finished_.store(false, std::memory_order_release);
    frontier_.store(at_us, std::memory_order_relaxed);
    last_pts = at_us;
    return true;
  };
  if (start_us > 0 && !restart(start_us))
    return fail(Refusal{RefusalCode::SeekFailed, "saut de la pré-analyse"});
  log->info("pré-analyse des plans : {} threads, depuis {:.1f} s", threads,
            static_cast<double>(start_us) / 1e6);

  const auto started = std::chrono::steady_clock::now();
  std::int64_t frames = 0;
  std::int64_t filled_from = -1;
  std::int64_t skipped_to = -1;
  bool ended = false;
  while (!stop.stop_requested()) {
    if (const std::int64_t to = seek_to_.exchange(-1, std::memory_order_relaxed); to >= 0) {
      // Déjà analysé : rien à refaire là, l'analyse continue où elle en est.
      if (timeline_.analyzed_until(to) - to < options_.skip_known_us || to > frontier_us()) {
        if (!restart(to)) return fail(Refusal{RefusalCode::SeekFailed, "saut de la pré-analyse"});
        ended = false;
      }
    }
    // Fin atteinte, ou assez d'avance : attendre un saut, ou que la lecture approche.
    const auto ahead = [this] {
      return frontier_us() - playhead_.load(std::memory_order_relaxed) > options_.lead_us;
    };
    if (ended || ahead()) {
      std::unique_lock lock(wake_mutex_);
      wake_.wait_for(lock, stop, std::chrono::milliseconds(500),
                     [&] { return seek_to_.load(std::memory_order_relaxed) >= 0 || (!ended && !ahead()); });
      continue;
    }
    // Une zone déjà analysée devant (après un retour en arrière) : sautée.
    if (const std::int64_t known = timeline_.analyzed_until(last_pts + 1);
        frames > 0 && known - last_pts > options_.skip_known_us && known > skipped_to) {
      // Le saut retombe sur l'image clé d'avant : sans ce garde, il se répéterait.
      skipped_to = known;
      if (!restart(known)) return fail(Refusal{RefusalCode::SeekFailed, "saut de la pré-analyse"});
      continue;
    }
    auto received = decoder->receive();
    if (!received) return fail(std::move(received.error()));
    if (*received) {
      const DecodedFrame& frame = **received;
      if (frame.pts_us == kNoPts) continue;
      auto measured = measure_frame(frame);
      if (!measured) return fail(std::move(measured.error()));
      last_pts = frame.pts_us;
      last_duration = frame.duration_us;
      frontier_.store(frame.pts_us, std::memory_order_relaxed);
      ++frames;
      if (auto shot = cutter.push(*measured)) timeline_.add(*shot);
      continue;
    }
    if (decoder->drained()) {
      if (auto shot = cutter.finish(last_pts + last_duration)) timeline_.add(*shot);
      // Un départ en cours de vidéo (une reprise, un saut) : le début reste à analyser, pour que la vidéo
      // soit connue entière (le cache, media/shot_cache.h). Chaque trou n'est repris qu'une fois.
      if (const std::int64_t gap = timeline_.first_gap_us(); gap < timeline_.end_us() && gap > filled_from) {
        filled_from = gap;
        if (!restart(gap)) return fail(Refusal{RefusalCode::SeekFailed, "saut de la pré-analyse"});
        continue;
      }
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      log->info("pré-analyse des plans finie : {} images en {:.1f} s, {} plans", frames, seconds,
                timeline_.size());
      finished_.store(true, std::memory_order_release);
      ended = true;
      continue;
    }
    auto packet = demuxer->read();
    if (!packet) return fail(std::move(packet.error()));
    if (!*packet) {
      if (auto ok = decoder->send_end(); !ok) return fail(std::move(ok.error()));
    } else if ((*packet)->track == track) {
      if (auto ok = decoder->send(**packet); !ok) return fail(std::move(ok.error()));
    }
  }
}

}  // namespace ajp::media
