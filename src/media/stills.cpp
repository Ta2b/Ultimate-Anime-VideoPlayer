#include "media/stills.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <optional>
#include <vector>

#include "kernels/spec/resample.h"
#include "media/demuxer.h"
#include "media/video_decoder.h"
#include "media/yuv_sample.h"

namespace ajp::media {
namespace {

using yuv::is_hdr;
using yuv::Layout;
using yuv::layout_of;
using yuv::sample;

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::uint8_t to_byte(double v) {
  return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0));
}

// La découpe centrale de spec (origine paire), mean et spread repris. Refus si rien n'en reste.
std::expected<Still, Refusal> crop_still(const Still& in, const StillsSpec& spec) {
  const int width = std::min(in.width, spec.max_width) / spec.align_width * spec.align_width;
  const int height = std::min(in.height, spec.max_height) / spec.align_height * spec.align_height;
  if (width <= 0 || height <= 0)
    return std::unexpected(Refusal{
        RefusalCode::FormatUnsupported,
        std::format("image réduite de {}x{} : trop petite pour la préparation", in.width, in.height)});
  const int x0 = (in.width - width) / 2 / 2 * 2;
  const int y0 = (in.height - height) / 2 / 2 * 2;
  Still out = in;
  out.width = width;
  out.height = height;
  const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t in_plane = static_cast<std::size_t>(in.width) * static_cast<std::size_t>(in.height);
  out.rgb.assign(3 * plane, 0);
  for (std::size_t c = 0; c < 3; ++c)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        out.rgb[c * plane + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)] =
            in.rgb[c * in_plane + static_cast<std::size_t>(y0 + y) * static_cast<std::size_t>(in.width) +
                   static_cast<std::size_t>(x0 + x)];
  return out;
}

double cubic(double x, double b, double c) {
  x = std::abs(x);
  if (x < 1.0)
    return ((12.0 - 9.0 * b - 6.0 * c) * x * x * x + (-18.0 + 12.0 * b + 6.0 * c) * x * x + (6.0 - 2.0 * b)) /
           6.0;
  if (x < 2.0)
    return ((-b - 6.0 * c) * x * x * x + (6.0 * b + 30.0 * c) * x * x + (-12.0 * b - 48.0 * c) * x +
            (8.0 * b + 24.0 * c)) /
           6.0;
  return 0.0;
}

// Les poids d'une dimension : pour chaque sortie, (première source, poids normalisés).
struct Taps {
  std::vector<int> first;
  std::vector<std::vector<double>> weights;
};

Taps taps(int src, int dst, const StillReduce& to) {
  Taps out;
  const double scale = static_cast<double>(src) / static_cast<double>(dst);
  const double widen = std::max(scale, 1.0);
  for (int o = 0; o < dst; ++o) {
    std::vector<double> w;
    int first = 0;
    if (to.filter == StillFilter::Area) {
      const double lo = o * scale, hi = (o + 1) * scale;
      first = static_cast<int>(std::floor(lo));
      for (int s = first; s < static_cast<int>(std::ceil(hi)); ++s)
        w.push_back(std::max(0.0, std::min(hi, s + 1.0) - std::max(lo, static_cast<double>(s))));
    } else {
      const double support = to.filter == StillFilter::Spline36 ? 3.0
                             : to.filter == StillFilter::Cubic  ? 2.0
                                                                : 1.0;
      const double center = (o + 0.5) * scale - 0.5;
      first = static_cast<int>(std::ceil(center - support * widen));
      const int last = static_cast<int>(std::floor(center + support * widen));
      for (int s = first; s <= last; ++s) {
        const double t = (s - center) / widen;
        switch (to.filter) {
          case StillFilter::Spline36:
            w.push_back(kernels::spline36(t));
            break;
          case StillFilter::Cubic:
            w.push_back(cubic(t, to.b, to.c));
            break;
          default:
            w.push_back(std::max(0.0, 1.0 - std::abs(t)));
            break;
        }
      }
    }
    double sum = 0.0;
    for (const double v : w) sum += v;
    if (sum != 0.0)
      for (double& v : w) v /= sum;
    out.first.push_back(first);
    out.weights.push_back(std::move(w));
  }
  return out;
}

}  // namespace

Still reduce_still(const Still& in, const StillReduce& to) {
  if (to.width <= 0 || to.height <= 0 || (to.width == in.width && to.height == in.height)) return in;
  const Taps tx = taps(in.width, to.width, to);
  const Taps ty = taps(in.height, to.height, to);
  const auto clamp_index = [](int i, int n) { return std::clamp(i, 0, n - 1); };
  const std::size_t in_plane = static_cast<std::size_t>(in.width) * static_cast<std::size_t>(in.height);
  const std::size_t out_plane = static_cast<std::size_t>(to.width) * static_cast<std::size_t>(to.height);
  Still out = in;
  out.width = to.width;
  out.height = to.height;
  out.rgb.assign(3 * out_plane, 0);
  std::vector<double> rows(static_cast<std::size_t>(to.width) * static_cast<std::size_t>(in.height));
  for (std::size_t c = 0; c < 3; ++c) {
    const std::uint8_t* src = in.rgb.data() + c * in_plane;
    for (int y = 0; y < in.height; ++y)
      for (int x = 0; x < to.width; ++x) {
        double v = 0.0;
        const auto& w = tx.weights[static_cast<std::size_t>(x)];
        for (std::size_t k = 0; k < w.size(); ++k)
          v += w[k] * src[static_cast<std::size_t>(y) * static_cast<std::size_t>(in.width) +
                          static_cast<std::size_t>(clamp_index(
                              tx.first[static_cast<std::size_t>(x)] + static_cast<int>(k), in.width))];
        rows[static_cast<std::size_t>(y) * static_cast<std::size_t>(to.width) + static_cast<std::size_t>(x)] =
            v;
      }
    for (int y = 0; y < to.height; ++y)
      for (int x = 0; x < to.width; ++x) {
        double v = 0.0;
        const auto& w = ty.weights[static_cast<std::size_t>(y)];
        for (std::size_t k = 0; k < w.size(); ++k)
          v += w[k] * rows[static_cast<std::size_t>(clamp_index(
                               ty.first[static_cast<std::size_t>(y)] + static_cast<int>(k), in.height)) *
                               static_cast<std::size_t>(to.width) +
                           static_cast<std::size_t>(x)];
        out.rgb[c * out_plane + static_cast<std::size_t>(y) * static_cast<std::size_t>(to.width) +
                static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
      }
  }
  return out;
}

std::expected<Still, Refusal> still_from_frame(const DecodedFrame& frame, const StillsSpec& spec) {
  if (spec.reduce.width > 0 && spec.reduce.height > 0 &&
      (frame.width > spec.reduce.width || frame.height > spec.reduce.height)) {
    // L'image entière, réduite, puis la découpe centrale de spec.
    StillsSpec whole = spec;
    whole.reduce = {};
    whole.max_width = frame.width;
    whole.max_height = frame.height;
    whole.align_width = 2;
    whole.align_height = 2;
    auto full = still_from_frame(frame, whole);
    if (!full) return full;
    return crop_still(reduce_still(*full, spec.reduce), spec);
  }
  if (is_hdr(frame.color))
    return refuse(RefusalCode::FormatUnsupported,
                  "vidéo HDR (ou à large gamut) : la préparation se fait sur une vidéo SDR");
  const std::optional<Layout> layout = layout_of(frame.format);
  if (!layout)
    return refuse(RefusalCode::FormatUnsupported, "format de pixel de la vidéo non pris pour la préparation");
  if (frame.data == nullptr || frame.data->memory() != FrameMemory::Host)
    return refuse(RefusalCode::FormatUnsupported, "image décodée hors de la mémoire hôte");
  if (frame.color.matrix != ColorMatrix::Bt601 && frame.color.matrix != ColorMatrix::Bt709)
    return refuse(RefusalCode::FormatUnsupported, "matrice de couleur de la vidéo non prise");

  const int width = std::min(frame.width, spec.max_width) / spec.align_width * spec.align_width;
  const int height = std::min(frame.height, spec.max_height) / spec.align_height * spec.align_height;
  if (width <= 0 || height <= 0)
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("vidéo de {}x{} : trop petite pour la préparation", frame.width, frame.height));
  // Origine paire : la chroma est à demi-résolution.
  const int x0 = (frame.width - width) / 2 / 2 * 2;
  const int y0 = (frame.height - height) / 2 / 2 * 2;

  const bool bt709 = frame.color.matrix == ColorMatrix::Bt709;
  const double kr = bt709 ? 0.2126 : 0.299;
  const double kb = bt709 ? 0.0722 : 0.114;
  const double kg = 1.0 - kr - kb;
  const bool full = frame.color.range == ColorRange::Full;
  const double scale = std::ldexp(1.0, layout->bits - 8);
  const double y_off = full ? 0.0 : 16.0 * scale;
  const double y_range = (full ? 255.0 : 219.0) * scale;
  const double c_range = (full ? 255.0 : 224.0) * scale;
  const double c_mid = 128.0 * scale;

  const auto luma = frame.data->host_plane(0);
  const std::size_t luma_stride = frame.data->host_stride(0);
  const auto chroma_u = frame.data->host_plane(1);
  const std::size_t u_stride = frame.data->host_stride(1);
  std::span<const std::byte> chroma_v;
  std::size_t v_stride = 0;
  if (!layout->semi_planar) {
    chroma_v = frame.data->host_plane(2);
    v_stride = frame.data->host_stride(2);
  }

  Still still;
  still.width = width;
  still.height = height;
  still.pts_us = frame.pts_us;
  const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  still.rgb.resize(3 * plane);
  double sum = 0.0;
  double sum_squares = 0.0;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int sx = x0 + x, sy = y0 + y;
      const double yy = (sample(luma, luma_stride, sx, sy, *layout) - y_off) / y_range;
      double cb = 0.0, cr = 0.0;
      if (layout->semi_planar) {
        cb = sample(chroma_u, u_stride, sx / 2, sy / 2, *layout, 0, 2);
        cr = sample(chroma_u, u_stride, sx / 2, sy / 2, *layout, 1, 2);
      } else {
        cb = sample(chroma_u, u_stride, sx / 2, sy / 2, *layout);
        cr = sample(chroma_v, v_stride, sx / 2, sy / 2, *layout);
      }
      cb = (cb - c_mid) / c_range;
      cr = (cr - c_mid) / c_range;
      const double r = yy + 2.0 * (1.0 - kr) * cr;
      const double b = yy + 2.0 * (1.0 - kb) * cb;
      const double g = (yy - kr * r - kb * b) / kg;
      const std::size_t at =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
      still.rgb[at] = to_byte(r);
      still.rgb[plane + at] = to_byte(g);
      still.rgb[2 * plane + at] = to_byte(b);
      const double luminance = std::clamp(yy, 0.0, 1.0);
      sum += luminance;
      sum_squares += luminance * luminance;
    }
  }
  const double n = static_cast<double>(plane);
  still.mean = sum / n;
  still.spread = std::sqrt(std::max(0.0, sum_squares / n - still.mean * still.mean));
  return still;
}

std::vector<std::size_t> pick_stills(std::span<const Still> stills, const StillsSpec& spec) {
  std::vector<std::size_t> usable;
  for (std::size_t i = 0; i < stills.size(); ++i)
    if (stills[i].mean >= spec.min_mean && stills[i].spread >= spec.min_spread) usable.push_back(i);
  if (usable.size() < static_cast<std::size_t>(spec.minimum)) return {};
  // Les plus contrastées d'abord, puis remises dans l'ordre du temps.
  std::ranges::stable_sort(usable,
                           [&](std::size_t a, std::size_t b) { return stills[a].spread > stills[b].spread; });
  usable.resize(std::min(usable.size(), static_cast<std::size_t>(spec.count)));
  std::ranges::sort(usable,
                    [&](std::size_t a, std::size_t b) { return stills[a].pts_us < stills[b].pts_us; });
  return usable;
}

std::expected<std::vector<Still>, Refusal> extract_stills(const std::filesystem::path& file,
                                                          const StillsSpec& spec, const std::stop_token& stop,
                                                          const std::function<void(float)>& progress) {
  auto demuxer = Demuxer::open(file);
  if (!demuxer) return std::unexpected(std::move(demuxer.error()));
  const TrackInfo* video = nullptr;
  for (const TrackInfo& track : demuxer->tracks())
    if (track.kind == TrackKind::Video && !track.attached_picture) {
      video = &track;
      break;
    }
  if (video == nullptr) return refuse(RefusalCode::TrackUnavailable, "aucune piste vidéo dans ce fichier");
  if (const auto color = track_color(*video); color && is_hdr(*color))
    return refuse(RefusalCode::FormatUnsupported,
                  "vidéo HDR : la préparation du mode Performance se fait sur une vidéo SDR");
  const std::int64_t duration = demuxer->duration_us() > 0 ? demuxer->duration_us() : video->duration_us;
  if (duration <= 0) return refuse(RefusalCode::SeekFailed, "durée de la vidéo inconnue");
  const std::array<int, 1> keep{video->index};
  if (auto ok = demuxer->keep_only(keep); !ok) return std::unexpected(std::move(ok.error()));
  // Le décodage logiciel seul : la préparation ne touche pas au GPU que la lecture occupe.
  auto decoder = VideoDecoder::open(*video, DecoderOptions{.attempts = {HwApi::Software}});
  if (!decoder) return std::unexpected(std::move(decoder.error()));

  std::vector<Still> found;
  bool hdr_seen = false;
  Refusal last_refusal{RefusalCode::DecodeFailed, "aucune image n'a pu être décodée"};
  for (int i = 0; i < spec.candidates; ++i) {
    if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, "préparation interrompue");
    if (progress) progress(static_cast<float>(i) / static_cast<float>(spec.candidates));
    const double position =
        0.08 + 0.84 * (static_cast<double>(i) + 0.5) / static_cast<double>(spec.candidates);
    const auto target = static_cast<std::int64_t>(position * static_cast<double>(duration));
    if (!demuxer->seek(target, video->index, SeekTarget::NearestKeyframe)) continue;
    decoder->flush();
    std::optional<DecodedFrame> frame;
    for (int packets = 0; packets < 400 && !frame; ++packets) {
      auto received = decoder->receive();
      if (!received) break;
      if (*received) {
        frame = std::move(**received);
        break;
      }
      if (decoder->drained()) break;
      auto packet = demuxer->read();
      if (!packet) break;
      if (!*packet) {
        if (!decoder->send_end()) break;
      } else if ((*packet)->track == video->index && !decoder->send(**packet)) {
        break;
      }
    }
    if (!frame) continue;
    auto still = still_from_frame(*frame, spec);
    if (!still) {
      hdr_seen = hdr_seen || still.error().detail.starts_with("vidéo HDR");
      last_refusal = std::move(still.error());
      if (hdr_seen) break;  // une image HDR dit tout de la vidéo
      continue;
    }
    found.push_back(std::move(*still));
  }
  if (progress) progress(1.0F);
  if (hdr_seen) return std::unexpected(std::move(last_refusal));
  if (found.empty()) return std::unexpected(std::move(last_refusal));
  const std::vector<std::size_t> picked = pick_stills(found, spec);
  if (picked.empty())
    return refuse(RefusalCode::DecodeFailed,
                  std::format("{} images décodées, moins de {} assez claires et contrastées : cette vidéo ne "
                              "donne pas de corpus",
                              found.size(), spec.minimum));
  std::vector<Still> out;
  out.reserve(picked.size());
  for (const std::size_t index : picked) out.push_back(std::move(found[index]));
  return out;
}

}  // namespace ajp::media
