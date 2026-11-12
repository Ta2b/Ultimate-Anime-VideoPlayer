#include "media/telecine.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <tuple>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

#include "media/av_util.h"
#include "media/log.h"
#include "media/video_decoder.h"

namespace ajp::media {
namespace {

// Une ligne sur kRowStep de chaque parité, une colonne sur kColumnStep : de quoi mesurer un champ sans le
// lire en entier.
constexpr int kRowStep = 8;
constexpr int kColumnStep = 4;
// En unités de 8 bits : en deçà d'une différence moyenne de kMoving, rien n'a bougé ; un champ est « répété »
// s'il diffère d'au plus kRepeatShare de l'autre et de moins de kRepeated en valeur.
constexpr double kMoving = 1.0;
constexpr double kRepeated = 0.6;
constexpr double kRepeatShare = 0.3;
// Au moins kMinCompared images comparées dont kMinMoving en mouvement pour conclure ; un 2:3 répète deux fois
// sur cinq (0,4 des images en mouvement), un vrai 30i n'en répète aucune.
constexpr int kMinCompared = 40;
constexpr int kMinMoving = 20;
constexpr double kTelecineShare = 0.2;
constexpr double kNativeShare = 0.08;

// Ce qui est décodé pour mesurer : assez d'images pour conclure, sans retarder l'ouverture.
constexpr int kMaxFrames = 120;
constexpr int kMaxPackets = 600;
constexpr auto kBudget = std::chrono::seconds(4);

// Les images que le décodeur rend, mesurées ; leur nombre.
int drain(AVCodecContext& ctx, AVFrame& frame, TelecineMeter& meter) {
  int measured = 0;
  while (avcodec_receive_frame(&ctx, &frame) >= 0) {
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
    const int blocked = AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_BITSTREAM | AV_PIX_FMT_FLAG_RGB;
    if (desc != nullptr && frame.data[0] != nullptr && (desc->flags & blocked) == 0 &&
        (desc->flags & AV_PIX_FMT_FLAG_PLANAR) != 0 && desc->comp[0].depth >= 8 &&
        desc->comp[0].depth <= 16) {
      const int depth = desc->comp[0].depth;
      meter.push(frame.data[0], frame.linesize[0], frame.width, frame.height, depth > 8 ? 2 : 1, depth);
      ++measured;
    }
    av_frame_unref(&frame);
  }
  return measured;
}

}  // namespace

void TelecineMeter::push(const std::uint8_t* luma, std::ptrdiff_t stride, int width, int height,
                         int sample_bytes, int bits) {
  if (luma == nullptr || width < 16 || height < 2 * kRowStep || (sample_bytes != 1 && sample_bytes != 2))
    return;
  const int shift = std::max(bits - 8, 0);
  std::vector<std::uint16_t> top;
  std::vector<std::uint16_t> bottom;
  for (int y = 0; y + 1 < height; y += kRowStep) {
    for (int parity = 0; parity < 2; ++parity) {
      const std::uint8_t* row = luma + static_cast<std::ptrdiff_t>(y + parity) * stride;
      auto& into = parity == 0 ? top : bottom;
      for (int x = 0; x < width; x += kColumnStep) {
        const auto index = static_cast<std::size_t>(x) * static_cast<std::size_t>(sample_bytes);
        const std::uint16_t value =
            sample_bytes == 1 ? row[index] : static_cast<std::uint16_t>(row[index] | (row[index + 1] << 8));
        into.push_back(static_cast<std::uint16_t>(value >> shift));
      }
    }
  }
  if (top_.size() == top.size() && !top.empty()) {
    const auto mean_difference = [](const std::vector<std::uint16_t>& a,
                                    const std::vector<std::uint16_t>& b) {
      std::int64_t total = 0;
      for (std::size_t i = 0; i < a.size(); ++i)
        total += std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
      return static_cast<double>(total) / static_cast<double>(a.size());
    };
    const double d_top = mean_difference(top, top_);
    const double d_bottom = mean_difference(bottom, bottom_);
    const double high = std::max(d_top, d_bottom);
    const double low = std::min(d_top, d_bottom);
    ++compared_;
    if (high >= kMoving) {
      ++moving_;
      if (low <= kRepeatShare * high && low <= kRepeated) ++repeats_;
    }
  }
  top_ = std::move(top);
  bottom_ = std::move(bottom);
}

TelecineMeter::Verdict TelecineMeter::verdict() const {
  if (compared_ < kMinCompared || moving_ < kMinMoving) return Verdict::Unknown;
  const double share = static_cast<double>(repeats_) / static_cast<double>(moving_);
  if (share >= kTelecineShare) return Verdict::Telecine;
  if (share <= kNativeShare) return Verdict::Native;
  return Verdict::Unknown;
}

std::optional<bool> measure_telecine(Demuxer& demuxer, const TrackInfo& track) {
  if (track.kind != TrackKind::Video || !track.parameters) return std::nullopt;
  const AVCodec* codec = avcodec_find_decoder(track.parameters->av().codec_id);
  if (codec == nullptr) return std::nullopt;
  CodecContextPtr ctx(avcodec_alloc_context3(codec));
  if (!ctx || avcodec_parameters_to_context(ctx.get(), &track.parameters->av()) < 0) return std::nullopt;
  ctx->thread_count = background_decoder_threads();
  if (avcodec_open2(ctx.get(), codec, nullptr) < 0) return std::nullopt;
  FramePtr frame(av_frame_alloc());
  if (!frame) return std::nullopt;

  const std::array<int, 1> keep{track.index};
  if (!demuxer.keep_only(keep)) return std::nullopt;
  // Vers le premier quart : un générique ou un logo fixe au début ne montre pas de mouvement.
  if (const std::int64_t duration = demuxer.duration_us(); duration > 0)
    std::ignore = demuxer.seek(duration / 4, track.index, SeekTarget::NearestKeyframe);

  TelecineMeter meter;
  const auto started = std::chrono::steady_clock::now();
  int frames = 0;
  bool ended = false;
  for (int packets = 0; packets < kMaxPackets && frames < kMaxFrames && !ended; ++packets) {
    if (std::chrono::steady_clock::now() - started > kBudget) break;
    auto packet = demuxer.read();
    if (!packet) break;
    if (!*packet) {
      avcodec_send_packet(ctx.get(), nullptr);
      ended = true;
    } else if ((*packet)->track != track.index || avcodec_send_packet(ctx.get(), (*packet)->av.get()) < 0) {
      continue;
    }
    frames += drain(*ctx, *frame, meter);
  }
  std::ignore = demuxer.seek_file(0, track.index);  // la lecture repart du début
  const auto verdict = meter.verdict();
  make_logger()->info("piste {} : télécinéma mesuré sur {} images ({} en mouvement, {} champs répétés) : {}",
                      track.index, meter.compared(), meter.moving(), meter.repeats(),
                      verdict == TelecineMeter::Verdict::Telecine ? "oui"
                      : verdict == TelecineMeter::Verdict::Native ? "non, vrai 30i"
                                                                  : "non concluant");
  if (verdict == TelecineMeter::Verdict::Unknown) return std::nullopt;
  return verdict == TelecineMeter::Verdict::Telecine;
}

}  // namespace ajp::media
