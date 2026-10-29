#include "media/stills.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <optional>

#include "media/demuxer.h"
#include "media/video_decoder.h"

namespace ajp::media {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

bool is_hdr(const ColorInfo& color) {
  return color.transfer == Transfer::Pq || color.transfer == Transfer::Hlg ||
         color.matrix == ColorMatrix::Bt2020Ncl || color.primaries == Primaries::Bt2020 ||
         color.hdr.has_value();
}

struct Layout {
  int bits = 8;
  bool semi_planar = false;  // chroma entrelacée (Nv12, P010)
  bool high = false;         // échantillons de 16 bits
  int shift = 0;             // décalage vers le bas des bits utiles (P010 : 6)
};

std::optional<Layout> layout_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::Nv12:
      return Layout{.bits = 8, .semi_planar = true};
    case PixelFormat::Yuv420p:
      return Layout{.bits = 8};
    case PixelFormat::P010:
      return Layout{.bits = 10, .semi_planar = true, .high = true, .shift = 6};
    case PixelFormat::Yuv420p10:
      return Layout{.bits = 10, .high = true};
    default:
      return std::nullopt;
  }
}

int sample(std::span<const std::byte> plane, std::size_t stride, int x, int y, const Layout& layout,
           int channel = 0, int channels = 1) {
  const std::size_t unit = layout.high ? 2U : 1U;
  const std::size_t at =
      static_cast<std::size_t>(y) * stride +
      (static_cast<std::size_t>(x) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(channel)) *
          unit;
  if (!layout.high) return static_cast<int>(plane[at]);
  const unsigned raw = static_cast<unsigned>(plane[at]) | (static_cast<unsigned>(plane[at + 1]) << 8U);
  return static_cast<int>(raw >> static_cast<unsigned>(layout.shift));
}

std::uint8_t to_byte(double v) {
  return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0));
}

}  // namespace

std::expected<Still, Refusal> still_from_frame(const DecodedFrame& frame, const StillsSpec& spec) {
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
