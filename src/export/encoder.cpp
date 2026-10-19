#include "export/encoder.h"

#include <algorithm>
#include <format>
#include <limits>
#include <string>
#include <utility>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixdesc.h>
}

#include "export/side_data.h"

namespace ajp::exporter {
namespace {

// Dictionnaire d'options libéré à la sortie de portée.
struct Options {
  AVDictionary* dict = nullptr;
  Options() = default;
  Options(const Options&) = delete;
  Options& operator=(const Options&) = delete;
  ~Options() { av_dict_free(&dict); }
  void set(const char* key, const std::string& value) { av_dict_set(&dict, key, value.c_str(), 0); }
};

AVPixelFormat av_format(PixelFormat format) {
  switch (format) {
    case PixelFormat::Yuv420p10:
      return AV_PIX_FMT_YUV420P10;
    case PixelFormat::P010:
      return AV_PIX_FMT_P010;
    case PixelFormat::Rgb16:
      return AV_PIX_FMT_GBRP16;
  }
  return AV_PIX_FMT_NONE;
}

// Étiquettes HDR10 : primaires BT.2020, transfert PQ ; matrice BT.2020 non constante en
// plage limitée pour le YUV, identité en pleine échelle pour le RGB.
void tag_color(AVCodecContext& ctx, PixelFormat format) {
  ctx.color_primaries = AVCOL_PRI_BT2020;
  ctx.color_trc = AVCOL_TRC_SMPTE2084;
  if (format == PixelFormat::Rgb16) {
    ctx.colorspace = AVCOL_SPC_RGB;
    ctx.color_range = AVCOL_RANGE_JPEG;
  } else {
    ctx.colorspace = AVCOL_SPC_BT2020_NCL;
    ctx.color_range = AVCOL_RANGE_MPEG;
    ctx.chroma_sample_location = AVCHROMA_LOC_LEFT;
  }
}

void copy_color(AVFrame& frame, const AVCodecContext& ctx) {
  frame.color_primaries = ctx.color_primaries;
  frame.color_trc = ctx.color_trc;
  frame.colorspace = ctx.colorspace;
  frame.color_range = ctx.color_range;
  frame.chroma_location = ctx.chroma_sample_location;
}

void set_rate(AVCodecContext& ctx, const EncoderSettings& s) {
  if (s.rate != RateControl::Bitrate) return;
  ctx.bit_rate = s.bitrate_kbps * 1000;
  if (s.max_bitrate_kbps != 0) ctx.rc_max_rate = s.max_bitrate_kbps * 1000;
}

// Options propres à chaque famille ; les plages sont déjà vérifiées par validate_settings.
void set_options(AVCodecContext& ctx, const EncoderSpec& spec, const EncoderSettings& s, Options& options) {
  if (s.keyframe_interval) ctx.gop_size = *s.keyframe_interval;
  set_rate(ctx, s);
  const auto quality = std::to_string(s.quality);
  switch (spec.family) {
    case EncoderFamily::Ffv1:
      ctx.level = 3;  // FFV1 version 3 : tranches indépendantes, CRC par tranche
      if (!s.keyframe_interval) ctx.gop_size = 1;
      ctx.thread_type = FF_THREAD_SLICE;
      options.set("slicecrc", "1");
      break;
    case EncoderFamily::Nvenc:
      if (s.preset) options.set("preset", std::format("p{}", *s.preset));
      if (spec.name == "hevc_nvenc") options.set("profile", "main10");
      if (s.rate == RateControl::ConstantQuality) {
        options.set("rc", "vbr");
        options.set("cq", quality);
      } else if (s.rate == RateControl::ConstantQp) {
        options.set("rc", "constqp");
        options.set("qp", quality);
      } else {
        options.set("rc", "vbr");
      }
      break;
    case EncoderFamily::SvtAv1:
      if (s.preset) options.set("preset", std::to_string(*s.preset));
      if (s.rate == RateControl::ConstantQuality) options.set("crf", quality);
      break;
    case EncoderFamily::Vulkan:
      if (spec.name == "hevc_vulkan") options.set("profile", "main10");
      if (s.rate == RateControl::ConstantQp) {
        options.set("rc_mode", "cqp");
        options.set("qp", quality);
      } else {
        options.set("rc_mode", "vbr");
      }
      break;
  }
}

std::expected<void, Refusal> check_format(const AVCodecContext& ctx, const AVCodec& codec,
                                          AVPixelFormat wanted) {
  const void* list = nullptr;
  int count = 0;
  if (const int error =
          avcodec_get_supported_config(&ctx, &codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &list, &count);
      error < 0) {
    return refuse_av(RefusalCode::EncoderUnavailable,
                     std::format("avcodec_get_supported_config {}", codec.name), error);
  }
  if (list == nullptr) return {};  // l'encodeur ne publie pas de liste : avcodec_open2 tranchera
  const auto* formats = static_cast<const AVPixelFormat*>(list);
  if (std::find(formats, formats + count, wanted) != formats + count) return {};
  return refuse(RefusalCode::FormatUnsupported,
                std::format("{} : format {} absent de sa liste", codec.name, av_get_pix_fmt_name(wanted)));
}

}  // namespace

std::expected<VideoEncoder, Refusal> VideoEncoder::open(const EncoderSettings& settings,
                                                        const EncodeGeometry& geometry,
                                                        const Hdr10Mastering& mastering, bool global_header) {
  auto spec = validate_settings(settings, geometry.width, geometry.height);
  if (!spec) return std::unexpected(spec.error());
  if (geometry.frame_rate.num <= 0 || geometry.frame_rate.den <= 0) {
    return refuse(RefusalCode::FieldOutOfRange, "export.frame_rate : cadence nulle ou négative");
  }
  const AVCodec* codec = avcodec_find_encoder_by_name(settings.encoder.c_str());
  if (codec == nullptr) {
    return refuse(RefusalCode::EncoderUnavailable,
                  std::format("avcodec_find_encoder_by_name {} : absent de cette construction de FFmpeg",
                              settings.encoder));
  }

  VideoEncoder encoder;
  encoder.ctx_.reset(avcodec_alloc_context3(codec));
  if (!encoder.ctx_)
    return refuse(RefusalCode::EncoderUnavailable, "avcodec_alloc_context3 : mémoire insuffisante");
  AVCodecContext& ctx = *encoder.ctx_;
  const AVPixelFormat software = av_format(settings.format);
  const bool on_device = (*spec)->family == EncoderFamily::Vulkan;
  ctx.width = geometry.width;
  ctx.height = geometry.height;
  ctx.time_base = kMicroseconds;
  ctx.framerate = geometry.frame_rate;
  ctx.pix_fmt = on_device ? AV_PIX_FMT_VULKAN : software;
  ctx.sw_pix_fmt = software;
  ctx.sample_aspect_ratio = AVRational{1, 1};
  // La durée de chaque image passe au paquet : MP4 en tire la fin de sa liste d'édition, et
  // sans elle la dernière image serait coupée à la lecture.
  ctx.flags |= AV_CODEC_FLAG_FRAME_DURATION;
  if (global_header) ctx.flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  tag_color(ctx, settings.format);
  encoder.frame_duration_us_ = av_rescale_q(1, av_inv_q(geometry.frame_rate), kMicroseconds);

  if (auto ok = check_format(ctx, *codec, ctx.pix_fmt); !ok) return std::unexpected(ok.error());
  if (auto ok = set_encoder_mastering(ctx, mastering); !ok) return std::unexpected(ok.error());
  Options options;
  set_options(ctx, **spec, settings, options);
  if (auto ok = encoder.prepare_frames(**spec); !ok) return std::unexpected(ok.error());

  if (const int error = avcodec_open2(&ctx, codec, &options.dict); error < 0) {
    const char* why = (*spec)->hardware ? " (GPU, pilote ou bibliothèque du constructeur absents ?)" : "";
    return refuse_av(RefusalCode::EncoderUnavailable,
                     std::format("avcodec_open2 {}{}", settings.encoder, why), error);
  }
  if (const AVDictionaryEntry* left = av_dict_iterate(options.dict, nullptr); left != nullptr) {
    return refuse(
        RefusalCode::EncoderUnavailable,
        std::format("avcodec_open2 {} : option {} inconnue de l'encodeur", settings.encoder, left->key));
  }
  return encoder;
}

// Images de travail allouées une fois ; voie Vulkan : appareil FFmpeg et pool d'images.
std::expected<void, Refusal> VideoEncoder::prepare_frames(const EncoderSpec& spec) {
  AVCodecContext& ctx = *ctx_;
  source_.reset(av_frame_alloc());
  converted_.reset(av_frame_alloc());
  sws_.reset(sws_alloc_context());
  if (!source_ || !converted_ || !sws_) {
    return refuse(RefusalCode::EncoderUnavailable,
                  "av_frame_alloc / sws_alloc_context : mémoire insuffisante");
  }
  // Pas de mise à l'échelle : seul le sous-échantillonnage de la chrominance filtre. Les
  // drapeaux rendent la conversion identique d'une machine à l'autre (framemd5).
  sws_->flags = SWS_BICUBIC | SWS_ACCURATE_RND | SWS_FULL_CHR_H_INP | SWS_BITEXACT;
  // Tranches sur tous les cœurs : chaque tranche lit la source entière, le résultat ne dépend
  // pas de leur nombre (framemd5 identique, vérifié en 1080p contre un seul thread).
  sws_->threads = 0;
  if (auto ok = renew_converted(); !ok) return ok;
  if (spec.family != EncoderFamily::Vulkan) return {};

  AVBufferRef* device = nullptr;
  if (const int error = av_hwdevice_ctx_create(&device, AV_HWDEVICE_TYPE_VULKAN, nullptr, nullptr, 0);
      error < 0) {
    return refuse_av(RefusalCode::EncoderUnavailable, "av_hwdevice_ctx_create vulkan", error);
  }
  device_ctx_.reset(device);
  frames_ctx_.reset(av_hwframe_ctx_alloc(device));
  device_.reset(av_frame_alloc());
  if (!frames_ctx_ || !device_)
    return refuse(RefusalCode::EncoderUnavailable, "av_hwframe_ctx_alloc : mémoire");
  auto* frames = reinterpret_cast<AVHWFramesContext*>(frames_ctx_->data);
  frames->format = AV_PIX_FMT_VULKAN;
  frames->sw_format = ctx.sw_pix_fmt;
  frames->width = ctx.width;
  frames->height = ctx.height;
  if (const int error = av_hwframe_ctx_init(frames_ctx_.get()); error < 0) {
    return refuse_av(RefusalCode::EncoderUnavailable, "av_hwframe_ctx_init vulkan", error);
  }
  ctx.hw_frames_ctx = av_buffer_ref(frames_ctx_.get());
  if (ctx.hw_frames_ctx == nullptr) return refuse(RefusalCode::EncoderUnavailable, "av_buffer_ref : mémoire");
  return {};
}

std::expected<void, Refusal> VideoEncoder::renew_converted() {
  const AVCodecContext& ctx = *ctx_;
  av_frame_unref(converted_.get());
  converted_->format = ctx.sw_pix_fmt;
  converted_->width = ctx.width;
  converted_->height = ctx.height;
  copy_color(*converted_, ctx);
  if (const int error = av_frame_get_buffer(converted_.get(), 0); error < 0) {
    return refuse_av(RefusalCode::EncodeFailed, "av_frame_get_buffer", error);
  }
  return {};
}

std::expected<AVFrame*, Refusal> VideoEncoder::convert(const HostFrame& frame) {
  const AVCodecContext& ctx = *ctx_;
  AVFrame& source = *source_;
  source.format = AV_PIX_FMT_RGBA64;
  source.width = frame.width;
  source.height = frame.height;
  source.color_primaries = AVCOL_PRI_BT2020;
  source.color_trc = AVCOL_TRC_SMPTE2084;
  source.colorspace = AVCOL_SPC_RGB;
  source.color_range = AVCOL_RANGE_JPEG;
  // libswscale ne fait que lire la source : la vue sur l'image hôte se passe de copie.
  source.data[0] = reinterpret_cast<std::uint8_t*>(const_cast<std::uint16_t*>(frame.rgba.data()));
  source.linesize[0] = static_cast<int>(frame.stride);

  // L'encodeur peut garder une référence sur l'image précédente : une mémoire neuve alors,
  // sans recopier l'ancienne image.
  if (av_frame_is_writable(converted_.get()) == 0) {
    if (auto ok = renew_converted(); !ok) return std::unexpected(ok.error());
  }
  const int error = sws_scale_frame(sws_.get(), converted_.get(), &source);
  source.data[0] = nullptr;
  if (error < 0) return refuse_av(RefusalCode::EncodeFailed, "sws_scale_frame RGBA64 vers l'encodeur", error);
  if (!device_) return converted_.get();

  av_frame_unref(device_.get());
  if (const int e = av_hwframe_get_buffer(frames_ctx_.get(), device_.get(), 0); e < 0) {
    return refuse_av(RefusalCode::EncodeFailed, "av_hwframe_get_buffer vulkan", e);
  }
  if (const int e = av_hwframe_transfer_data(device_.get(), converted_.get(), 0); e < 0) {
    return refuse_av(RefusalCode::EncodeFailed, "av_hwframe_transfer_data vers vulkan", e);
  }
  copy_color(*device_, ctx);
  return device_.get();
}

std::expected<void, Refusal> VideoEncoder::send(const HostFrame& frame, std::int64_t pts_us) {
  auto converted = convert(frame);
  if (!converted) return std::unexpected(converted.error());
  AVFrame& out = **converted;
  out.pts = pts_us;
  out.duration = frame_duration_us_;
  if (const int error = avcodec_send_frame(ctx_.get(), &out); error < 0) {
    return refuse_av(RefusalCode::EncodeFailed, std::format("avcodec_send_frame {}", ctx_->codec->name),
                     error);
  }
  return {};
}

std::expected<void, Refusal> VideoEncoder::flush() {
  if (const int error = avcodec_send_frame(ctx_.get(), nullptr); error < 0 && error != AVERROR_EOF) {
    return refuse_av(RefusalCode::EncodeFailed, std::format("avcodec_send_frame {} (fin)", ctx_->codec->name),
                     error);
  }
  return {};
}

std::expected<bool, Refusal> VideoEncoder::receive(AVPacket& packet) {
  const int error = avcodec_receive_packet(ctx_.get(), &packet);
  if (error == AVERROR(EAGAIN) || error == AVERROR_EOF) return false;
  if (error < 0) {
    return refuse_av(RefusalCode::EncodeFailed, std::format("avcodec_receive_packet {}", ctx_->codec->name),
                     error);
  }
  return true;
}

}  // namespace ajp::exporter
