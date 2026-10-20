#include "export/video_encoder.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <span>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include "core/log.h"

namespace ajp::exporter {
namespace {

// Les formats hôte qu'un encodeur accepte ; vide s'il ne le dit pas.
std::span<const AVPixelFormat> host_formats(const AVCodec& codec) {
  const void* formats = nullptr;
  int count = 0;
  if (avcodec_get_supported_config(nullptr, &codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &formats, &count) < 0 ||
      formats == nullptr)
    return {};
  return {static_cast<const AVPixelFormat*>(formats), static_cast<std::size_t>(count)};
}

// La disposition que lit un encodeur hôte : P010 puis YUV 4:2:0 10 bits planaire (FFV1), ou NV12.
std::optional<Layout> host_layout(const AVCodec& codec, bool ten_bit) {
  const std::span<const AVPixelFormat> formats = host_formats(codec);
  const auto accepts = [&](Layout layout) { return std::ranges::contains(formats, pixel_format(layout)); };
  if (ten_bit) {
    if (accepts(Layout::P010)) return Layout::P010;
    if (accepts(Layout::Yuv420p10)) return Layout::Yuv420p10;
    return std::nullopt;
  }
  if (accepts(Layout::Nv12)) return Layout::Nv12;
  return std::nullopt;
}

struct Opened {
  BufferPtr device;
  CodecContextPtr ctx;
  Layout layout = Layout::P010;
};

// Une tentative : le contexte rempli, les options posées, l'encodeur ouvert ; le texte du refus
// sinon.
std::expected<Opened, std::string> attempt(const AVCodec& codec, const EncoderCandidate& candidate,
                                           const std::vector<EncoderOption>& options,
                                           const EncoderSetup& setup) {
  Opened out;
  AVPixelFormat pix_fmt = AV_PIX_FMT_NONE;
  if (candidate.upload == Upload::Host) {
    const std::optional<Layout> layout = host_layout(codec, candidate.ten_bit);
    if (!layout) return std::unexpected(std::string("aucun format hôte en 4:2:0 de cette profondeur"));
    out.layout = *layout;
    pix_fmt = pixel_format(*layout);
  } else {
    out.layout = candidate.ten_bit ? Layout::P010 : Layout::Nv12;
    const AVHWDeviceType type =
        candidate.upload == Upload::Vaapi ? AV_HWDEVICE_TYPE_VAAPI : AV_HWDEVICE_TYPE_VULKAN;
    AVBufferRef* device = nullptr;
    if (const int err = av_hwdevice_ctx_create(&device, type, nullptr, nullptr, 0); err < 0)
      return std::unexpected(std::format("appareil : {}", av_error_text(err)));
    out.device.reset(device);
    BufferPtr frames(av_hwframe_ctx_alloc(out.device.get()));
    if (!frames) return std::unexpected(std::string("contexte d'images : mémoire"));
    auto* fc = reinterpret_cast<AVHWFramesContext*>(frames->data);
    fc->format = candidate.upload == Upload::Vaapi ? AV_PIX_FMT_VAAPI : AV_PIX_FMT_VULKAN;
    fc->sw_format = pixel_format(out.layout);
    fc->width = setup.width;
    fc->height = setup.height;
    fc->initial_pool_size = candidate.upload == Upload::Vaapi ? 16 : 0;
    if (const int err = av_hwframe_ctx_init(frames.get()); err < 0)
      return std::unexpected(
          std::format("surfaces {} : {}", av_get_pix_fmt_name(fc->sw_format), av_error_text(err)));
    pix_fmt = fc->format;
    out.ctx.reset(avcodec_alloc_context3(&codec));
    if (!out.ctx) return std::unexpected(std::string("contexte : mémoire"));
    out.ctx->hw_frames_ctx = av_buffer_ref(frames.get());
  }
  if (!out.ctx) out.ctx.reset(avcodec_alloc_context3(&codec));
  if (!out.ctx) return std::unexpected(std::string("contexte : mémoire"));
  AVCodecContext& c = *out.ctx;
  c.width = setup.width;
  c.height = setup.height;
  c.pix_fmt = pix_fmt;
  c.time_base = kMicroseconds;
  if (setup.fps_num > 0 && setup.fps_den > 0) c.framerate = AVRational{setup.fps_num, setup.fps_den};
  c.sample_aspect_ratio = setup.sample_aspect;
  const AvColor color = av_color(setup.color);
  c.color_primaries = color.primaries;
  c.color_trc = color.transfer;
  c.colorspace = color.matrix;
  c.color_range = color.range;
  c.chroma_sample_location = color.siting;
  // Une image clé toutes les 4 s environ : on se déplace vite dans le fichier.
  const double fps =
      setup.fps_num > 0 && setup.fps_den > 0 ? static_cast<double>(setup.fps_num) / setup.fps_den : 24.0;
  c.gop_size = std::clamp(static_cast<int>(std::lround(fps * 4.0)), 1, 600);
  c.flags |= AV_CODEC_FLAG_GLOBAL_HEADER;  // MKV : en-têtes de flux dans CodecPrivate
  const auto log = log_channel("export");
  for (const EncoderOption& option : options) {
    const int err = av_opt_set(&c, option.key.c_str(), option.value.c_str(), AV_OPT_SEARCH_CHILDREN);
    if (err < 0)
      log->debug("{} : option {}={} ignorée ({})", candidate.name, option.key, option.value,
                 av_error_text(err));
  }
  if (const int err = avcodec_open2(&c, &codec, nullptr); err < 0) return std::unexpected(av_error_text(err));
  return out;
}

}  // namespace

std::expected<VideoEncoder, Refusal> VideoEncoder::open(const EncoderSetup& setup) {
  const auto log = log_channel("export");
  std::vector<std::string> tried;
  for (const EncoderCandidate& candidate : encoder_candidates(setup.codec, setup.quality, setup.vendor)) {
    const AVCodec* codec = avcodec_find_encoder_by_name(candidate.name.c_str());
    if (codec == nullptr) {
      tried.push_back(std::format("{} absent", candidate.name));
      continue;
    }
    auto opened = attempt(*codec, candidate, candidate.options, setup);
    if (!opened && candidate.fallback != candidate.options) {
      log->info("{} refuse ses options complètes ({}) : options réduites", candidate.name, opened.error());
      opened = attempt(*codec, candidate, candidate.fallback, setup);
    }
    if (!opened) {
      tried.push_back(std::format("{} : {}", candidate.name, opened.error()));
      log->info("encodeur {} refusé : {}", candidate.name, opened.error());
      continue;
    }
    VideoEncoder encoder;
    encoder.device_ = std::move(opened->device);
    encoder.ctx_ = std::move(opened->ctx);
    encoder.layout_ = opened->layout;
    encoder.upload_ = candidate.upload;
    encoder.label_ = std::format("{} ({})", candidate.label, candidate.name);
    encoder.hw_frame_.reset(av_frame_alloc());
    encoder.packet_.reset(av_packet_alloc());
    if (!encoder.hw_frame_ || !encoder.packet_)
      return refuse(RefusalCode::EncodeFailed, "encodeur : mémoire");
    log->info("encodeur {} : {}x{} {}", encoder.label_, setup.width, setup.height,
              av_get_pix_fmt_name(pixel_format(encoder.layout_)));
    return encoder;
  }
  std::string detail = std::format("aucun encodeur {} ne s'ouvre", exporter::label(setup.codec));
  for (const std::string& line : tried) detail += std::format(" ; {}", line);
  return refuse(RefusalCode::EncoderUnavailable, std::move(detail));
}

std::expected<void, Refusal> VideoEncoder::drain(const PacketSink& sink) {
  while (true) {
    const int err = avcodec_receive_packet(ctx_.get(), packet_.get());
    if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return {};
    if (err < 0) return refuse_av(RefusalCode::EncodeFailed, "avcodec_receive_packet", err);
    auto sunk = sink(*packet_);
    av_packet_unref(packet_.get());
    if (!sunk) return sunk;
  }
}

std::expected<void, Refusal> VideoEncoder::encode(AVFrame* frame, const PacketSink& sink) {
  AVFrame* sent = frame;
  if (frame != nullptr && upload_ != Upload::Host) {
    av_frame_unref(hw_frame_.get());
    if (const int err = av_hwframe_get_buffer(ctx_->hw_frames_ctx, hw_frame_.get(), 0); err < 0)
      return refuse_av(RefusalCode::EncodeFailed, "av_hwframe_get_buffer", err);
    if (const int err = av_hwframe_transfer_data(hw_frame_.get(), frame, 0); err < 0)
      return refuse_av(RefusalCode::EncodeFailed, "av_hwframe_transfer_data", err);
    if (const int err = av_frame_copy_props(hw_frame_.get(), frame); err < 0)
      return refuse_av(RefusalCode::EncodeFailed, "av_frame_copy_props", err);
    sent = hw_frame_.get();
  }
  int err = avcodec_send_frame(ctx_.get(), sent);
  if (err == AVERROR(EAGAIN)) {  // sorties à lire d'abord, puis la même image
    if (auto drained = drain(sink); !drained) return drained;
    err = avcodec_send_frame(ctx_.get(), sent);
  }
  if (sent == hw_frame_.get()) av_frame_unref(hw_frame_.get());
  if (err < 0 && !(frame == nullptr && err == AVERROR_EOF))
    return refuse_av(RefusalCode::EncodeFailed, "avcodec_send_frame", err);
  return drain(sink);
}

}  // namespace ajp::exporter
