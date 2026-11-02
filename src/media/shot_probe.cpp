#include "media/shot_probe.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

#include "media/demuxer.h"
#include "media/log.h"
#include "media/video_decoder.h"
#include "media/yuv_sample.h"

namespace ajp::media {
namespace {

constexpr int kGridW = 128;
constexpr int kGridH = 72;
constexpr float kBarLuma = 0.03F;  // une ligne ou une colonne plus sombre partout : une bande noire

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

struct Point {
  float luma = 0.0F;
  float blue = 0.0F;
  float chroma = 0.0F;
};

}  // namespace

std::expected<ShotFrame, Refusal> measure_frame(const DecodedFrame& frame) {
  if (yuv::is_hdr(frame.color))
    return refuse(RefusalCode::FormatUnsupported, "vidéo HDR : pas de pré-analyse");
  const std::optional<yuv::Layout> layout = yuv::layout_of(frame.format);
  if (!layout) return refuse(RefusalCode::FormatUnsupported, "format de pixel non pris par la pré-analyse");
  if (frame.data == nullptr || frame.data->memory() != FrameMemory::Host)
    return refuse(RefusalCode::FormatUnsupported, "image décodée hors de la mémoire hôte");
  if (frame.width < 2 || frame.height < 2) return refuse(RefusalCode::FormatUnsupported, "image trop petite");
  const bool bt601 = frame.color.matrix == ColorMatrix::Bt601;
  const double kr = bt601 ? 0.299 : 0.2126;
  const double kb = bt601 ? 0.114 : 0.0722;
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

  std::vector<Point> points(static_cast<std::size_t>(kGridW) * kGridH);
  std::array<float, kGridW> column_max{};
  std::array<float, kGridH> row_max{};
  for (int gy = 0; gy < kGridH; ++gy) {
    const int sy = std::min((2 * gy + 1) * frame.height / (2 * kGridH), frame.height - 1);
    for (int gx = 0; gx < kGridW; ++gx) {
      const int sx = std::min((2 * gx + 1) * frame.width / (2 * kGridW), frame.width - 1);
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
      // Même pondération que la mesure de RTX HDR (BT.709 des canaux encodés).
      const auto luma = static_cast<float>(0.2126 * r + 0.7152 * g + 0.0722 * b);
      const double hi = std::max({r, g, b}), lo = std::min({r, g, b});
      Point& p = points.at(static_cast<std::size_t>(gy) * kGridW + static_cast<std::size_t>(gx));
      p = Point{.luma = luma,
                .blue = static_cast<float>(std::max(0.0, b - std::max(r, g))),
                .chroma = static_cast<float>(hi - lo)};
      column_max.at(static_cast<std::size_t>(gx)) =
          std::max(column_max.at(static_cast<std::size_t>(gx)), luma);
      row_max.at(static_cast<std::size_t>(gy)) = std::max(row_max.at(static_cast<std::size_t>(gy)), luma);
    }
  }
  // Bandes noires : les lignes et colonnes sombres partout, depuis chaque bord. Une image toute noire garde
  // tout (un fondu au noir se mesure).
  int x0 = 0, x1 = kGridW, y0 = 0, y1 = kGridH;
  while (x0 < kGridW && column_max.at(static_cast<std::size_t>(x0)) < kBarLuma) ++x0;
  while (x1 > x0 && column_max.at(static_cast<std::size_t>(x1 - 1)) < kBarLuma) --x1;
  while (y0 < kGridH && row_max.at(static_cast<std::size_t>(y0)) < kBarLuma) ++y0;
  while (y1 > y0 && row_max.at(static_cast<std::size_t>(y1 - 1)) < kBarLuma) --y1;
  if (x1 - x0 < kGridW / 4 || y1 - y0 < kGridH / 4) {
    x0 = 0, x1 = kGridW, y0 = 0, y1 = kGridH;
  }

  ShotFrame out;
  out.pts_us = frame.pts_us;
  std::array<float, kShotGrid * kShotGrid> grid_count{};
  double sum = 0.0, blue = 0.0, chroma = 0.0;
  for (int gy = y0; gy < y1; ++gy) {
    for (int gx = x0; gx < x1; ++gx) {
      const Point& p = points.at(static_cast<std::size_t>(gy) * kGridW + static_cast<std::size_t>(gx));
      const int bin = std::clamp(static_cast<int>(p.luma * static_cast<float>(kShotBins)), 0, kShotBins - 1);
      out.histogram.at(static_cast<std::size_t>(bin)) += 1.0F;
      const int cx = (gx - x0) * kShotGrid / (x1 - x0), cy = (gy - y0) * kShotGrid / (y1 - y0);
      const auto cell = static_cast<std::size_t>(cy * kShotGrid + cx);
      out.grid.at(cell) += p.luma;
      grid_count.at(cell) += 1.0F;
      sum += p.luma;
      blue += p.blue;
      chroma += p.chroma;
    }
  }
  const auto n = static_cast<float>((x1 - x0) * (y1 - y0));
  for (float& part : out.histogram) part /= n;
  for (std::size_t i = 0; i < out.grid.size(); ++i) out.grid.at(i) /= std::max(grid_count.at(i), 1.0F);
  out.mean = static_cast<float>(sum) / n;
  out.blue = static_cast<float>(blue) / n;
  out.chroma = static_cast<float>(chroma) / n;
  return out;
}

ShotProbe::ShotProbe(std::filesystem::path file, ShotTimeline& timeline, ShotProbeOptions options)
    : file_(std::move(file)), timeline_(timeline), options_(options) {}

std::unique_ptr<ShotProbe> ShotProbe::start(std::filesystem::path file, ShotTimeline& timeline,
                                            std::int64_t start_us, ShotProbeOptions options) {
  std::unique_ptr<ShotProbe> probe(new ShotProbe(std::move(file), timeline, options));
  probe->playhead_ = start_us;
  probe->frontier_ = start_us;
  ShotProbe* self = probe.get();
  probe->thread_ = std::jthread([self, start_us](const std::stop_token& stop) { self->run(stop, start_us); });
  return probe;
}

ShotProbe::~ShotProbe() {
  thread_.request_stop();
}

void ShotProbe::set_playhead(std::int64_t us) {
  playhead_.store(us, std::memory_order_relaxed);
}

void ShotProbe::seek(std::int64_t at_us) {
  playhead_.store(at_us, std::memory_order_relaxed);
  seek_to_.store(at_us, std::memory_order_relaxed);
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

void ShotProbe::run(const std::stop_token& stop, std::int64_t start_us) {
  auto log = make_logger();
  auto demuxer = Demuxer::open(file_);
  if (!demuxer) return fail(std::move(demuxer.error()));
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
  const int threads = options_.threads > 0
                          ? options_.threads
                          : std::max(2, static_cast<int>(std::thread::hardware_concurrency()) / 2);
  auto decoder =
      VideoDecoder::open(*video, DecoderOptions{.attempts = {HwApi::Software}, .threads = threads});
  if (!decoder) return fail(std::move(decoder.error()));

  // Repart de at_us : le plan en cours est clos là, un nouveau découpage commence (son premier plan n'a pas
  // de vraie coupe devant lui).
  ShotCutter cutter(start_us <= 0);
  std::int64_t last_pts = start_us;
  std::int64_t last_duration = 0;
  const auto restart = [&](std::int64_t at_us) -> bool {
    if (auto shot = cutter.finish(last_pts + last_duration)) timeline_.add(*shot);
    if (!demuxer->seek(std::max<std::int64_t>(at_us, 0), track, SeekTarget::PreviousKeyframe)) return false;
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
  bool ended = false;
  while (!stop.stop_requested()) {
    if (const std::int64_t to = seek_to_.exchange(-1, std::memory_order_relaxed); to >= 0) {
      // Déjà analysé : rien à refaire là, l'analyse continue où elle en est.
      if (timeline_.analyzed_until(to) - to < options_.skip_known_us || to > frontier_us()) {
        if (!restart(to)) return fail(Refusal{RefusalCode::SeekFailed, "saut de la pré-analyse"});
        ended = false;
      }
    }
    if (ended) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    // Assez d'avance : attendre que la lecture approche.
    if (frontier_us() - playhead_.load(std::memory_order_relaxed) > options_.lead_us) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    // Une zone déjà analysée devant (après un retour en arrière) : sautée.
    if (const std::int64_t known = timeline_.analyzed_until(last_pts + 1);
        frames > 0 && known - last_pts > options_.skip_known_us) {
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
