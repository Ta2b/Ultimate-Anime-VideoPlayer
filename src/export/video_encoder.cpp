#include "export/video_encoder.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/packet.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/mastering_display_metadata.h>
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
  Upload upload = Upload::Host;
  bool shared = false;  // l'appareil Vulkan du rendu (EncoderSetup::vulkan_device)
};

AVHWDeviceType device_type(Upload upload) {
  switch (upload) {
    case Upload::Cuda:
      return AV_HWDEVICE_TYPE_CUDA;
    case Upload::Vaapi:
      return AV_HWDEVICE_TYPE_VAAPI;
    case Upload::Vulkan:
      return AV_HWDEVICE_TYPE_VULKAN;
    case Upload::Host:
      break;
  }
  return AV_HWDEVICE_TYPE_NONE;
}

AVPixelFormat device_format(Upload upload) {
  switch (upload) {
    case Upload::Cuda:
      return AV_PIX_FMT_CUDA;
    case Upload::Vaapi:
      return AV_PIX_FMT_VAAPI;
    case Upload::Vulkan:
      return AV_PIX_FMT_VULKAN;
    case Upload::Host:
      break;
  }
  return AV_PIX_FMT_NONE;
}

// AV_CUDA_USE_PRIMARY_CONTEXT (hwcontext_cuda.h, qui exige les types CUDA) : le contexte primaire
// du GPU, celui des noyaux, de NVDEC et des tampons de l'export.
constexpr int kCudaUsePrimaryContext = 1;

// Images du pool CUDA allouées à l'ouverture. NVENC garde chaque image d'entrée jusqu'à la sortie
// de son paquet (lookahead et B-frames des options), et l'export en tient held de plus (ses
// tampons, la file d'écriture), plus deux (celle qu'on copie, celle qu'on encode). Allouées
// d'avance, aucun cuMemAlloc ne tombe pendant les premières images, quand les moteurs de la chaîne
// capturent leurs graphes CUDA.
int cuda_pool_frames(const std::vector<EncoderOption>& options, int export_held) {
  constexpr int kInFlight = 2;
  int held = std::max(export_held, 0);
  for (const EncoderOption& option : options) {
    if (option.key != "rc-lookahead" && option.key != "bf") continue;
    int value = 0;
    const char* end = option.value.data() + option.value.size();
    if (std::from_chars(option.value.data(), end, value).ec == std::errc() && value > 0) held += value;
  }
  return held + kInFlight;
}

// Le contexte d'images d'un encodeur matériel ; renderable : images Vulkan de l'appareil du rendu
// dans lesquelles libplacebo rend (COLOR_ATTACHMENT, en plus de ce que FFmpeg demande).
std::expected<BufferPtr, std::string> open_frames(AVBufferRef& device, const EncoderCandidate& candidate,
                                                  const std::vector<EncoderOption>& options, Layout layout,
                                                  const EncoderSetup& setup, bool renderable) {
  BufferPtr frames(av_hwframe_ctx_alloc(&device));
  if (!frames) return std::unexpected(std::string("contexte d'images : mémoire"));
  auto* fc = reinterpret_cast<AVHWFramesContext*>(frames->data);
  fc->format = device_format(candidate.upload);
  fc->sw_format = pixel_format(layout);
  fc->width = setup.width;
  fc->height = setup.height;
  fc->initial_pool_size = candidate.upload == Upload::Vaapi  ? 16
                          : candidate.upload == Upload::Cuda ? cuda_pool_frames(options, setup.held_frames)
                                                             : 0;
  if (renderable) {
    auto* vk = static_cast<AVVulkanFramesContext*>(fc->hwctx);
    vk->usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // FFmpeg y ajoute les siens
  }
  if (const int err = av_hwframe_ctx_init(frames.get()); err < 0)
    return std::unexpected(
        std::format("surfaces {} : {}", av_get_pix_fmt_name(fc->sw_format), av_error_text(err)));
  return frames;
}

// L'appareil et le contexte d'images d'un encodeur qui reçoit ses images sur l'appareil ; le
// texte du refus sinon. Un encodeur Vulkan prend l'appareil du rendu s'il est donné.
std::expected<Opened, std::string> open_device(const AVCodec& codec, const EncoderCandidate& candidate,
                                               const std::vector<EncoderOption>& options,
                                               const EncoderSetup& setup) {
  Opened out;
  out.upload = candidate.upload;
  out.layout = candidate.ten_bit ? Layout::P010 : Layout::Nv12;
  AVBufferRef* device = nullptr;
  if (candidate.upload == Upload::Vulkan && setup.vulkan_device != nullptr) {
    device = av_buffer_ref(setup.vulkan_device);
    if (device == nullptr) return std::unexpected(std::string("appareil partagé : mémoire"));
    out.shared = true;
  } else {
    const std::string ordinal = std::to_string(setup.cuda_ordinal);
    const bool cuda = candidate.upload == Upload::Cuda;
    const char* selector = cuda && setup.cuda_ordinal >= 0 ? ordinal.c_str() : nullptr;
    if (const int err = av_hwdevice_ctx_create(&device, device_type(candidate.upload), selector, nullptr,
                                               cuda ? kCudaUsePrimaryContext : 0);
        err < 0) {
      return std::unexpected(std::format("appareil : {}", av_error_text(err)));
    }
  }
  out.device.reset(device);
  auto frames = open_frames(*out.device, candidate, options, out.layout, setup, out.shared);
  if (!frames && out.shared) {
    // Sans rendu direct dans ses images, l'encodeur les charge depuis l'hôte, sur le même appareil.
    log_channel("export")->info("{} : images non rendables sur l'appareil du rendu ({}) ; chargées",
                                candidate.name, frames.error());
    frames = open_frames(*out.device, candidate, options, out.layout, setup, false);
  }
  if (!frames) return std::unexpected(std::move(frames.error()));
  out.ctx.reset(avcodec_alloc_context3(&codec));
  if (!out.ctx) return std::unexpected(std::string("contexte : mémoire"));
  out.ctx->hw_frames_ctx = av_buffer_ref(frames->get());
  out.ctx->pix_fmt = reinterpret_cast<AVHWFramesContext*>((*frames)->data)->format;
  return out;
}

// « lookahead_level=3 tf_level=4 », pour le journal.
std::string describe(const std::vector<EncoderOption>& options) {
  std::string text;
  for (const EncoderOption& option : options)
    text += std::format("{}{}={}", text.empty() ? "" : " ", option.key, option.value);
  return text;
}

// Une tentative : le contexte rempli, les options posées, l'encodeur ouvert ; le texte du refus
// sinon. Un encodeur CUDA dont l'appareil ne s'ouvre pas (FFmpeg sans CUDA) reçoit ses images de
// l'hôte, comme avant.
// HDR10 : l'affichage de mastering (ST 2086) et les niveaux de lumière de la source, joints au
// flux pour que le lecteur du fichier exporté ne perde pas le réglage de l'écran.
void hdr_side_data(AVCodecContext& c, const HdrMetadata& hdr) {
  const auto rational = [](float value) { return av_d2q(static_cast<double>(value), 1'000'000); };
  if (hdr.max_luminance > 0.0F) {
    AVPacketSideData* side = av_packet_side_data_new(&c.coded_side_data, &c.nb_coded_side_data,
                                                     AV_PKT_DATA_MASTERING_DISPLAY_METADATA,
                                                     sizeof(AVMasteringDisplayMetadata), 0);
    if (side != nullptr) {
      auto* mastering = reinterpret_cast<AVMasteringDisplayMetadata*>(side->data);
      for (std::size_t i = 0; i < 3; ++i) {
        mastering->display_primaries[i][0] = rational(hdr.primaries_xy[2 * i]);
        mastering->display_primaries[i][1] = rational(hdr.primaries_xy[2 * i + 1]);
      }
      mastering->white_point[0] = rational(hdr.white_xy[0]);
      mastering->white_point[1] = rational(hdr.white_xy[1]);
      mastering->max_luminance = rational(hdr.max_luminance);
      mastering->min_luminance = rational(hdr.min_luminance);
      mastering->has_primaries = 1;
      mastering->has_luminance = 1;
    }
  }
  if (hdr.max_cll > 0.0F || hdr.max_fall > 0.0F) {
    AVPacketSideData* side =
        av_packet_side_data_new(&c.coded_side_data, &c.nb_coded_side_data, AV_PKT_DATA_CONTENT_LIGHT_LEVEL,
                                sizeof(AVContentLightMetadata), 0);
    if (side != nullptr) {
      auto* light = reinterpret_cast<AVContentLightMetadata*>(side->data);
      light->MaxCLL = static_cast<unsigned>(std::lround(hdr.max_cll));
      light->MaxFALL = static_cast<unsigned>(std::lround(hdr.max_fall));
    }
  }
}

std::expected<Opened, std::string> attempt(const AVCodec& codec, const EncoderCandidate& candidate,
                                           const std::vector<EncoderOption>& options,
                                           const EncoderSetup& setup) {
  const auto log = log_channel("export");
  Opened out;
  if (candidate.upload != Upload::Host) {
    auto device = open_device(codec, candidate, options, setup);
    if (device) {
      out = std::move(*device);
    } else if (candidate.upload != Upload::Cuda) {
      return std::unexpected(std::move(device.error()));
    } else {
      log->info("{} : {} ; images chargées depuis l'hôte", candidate.name, device.error());
    }
  }
  if (!out.ctx) {
    const std::optional<Layout> layout = host_layout(codec, candidate.ten_bit);
    if (!layout) return std::unexpected(std::string("aucun format hôte en 4:2:0 de cette profondeur"));
    out.layout = *layout;
    out.upload = Upload::Host;
    out.ctx.reset(avcodec_alloc_context3(&codec));
    if (!out.ctx) return std::unexpected(std::string("contexte : mémoire"));
    out.ctx->pix_fmt = pixel_format(*layout);
  }
  AVCodecContext& c = *out.ctx;
  c.width = setup.width;
  c.height = setup.height;
  c.time_base = kMicroseconds;
  if (setup.fps_num > 0 && setup.fps_den > 0) c.framerate = AVRational{setup.fps_num, setup.fps_den};
  c.sample_aspect_ratio = setup.sample_aspect;
  const AvColor color = av_color(setup.color);
  c.color_primaries = color.primaries;
  c.color_trc = color.transfer;
  c.colorspace = color.matrix;
  c.color_range = color.range;
  c.chroma_sample_location = color.siting;
  if (setup.color.hdr && setup.color.transfer == Transfer::Pq) hdr_side_data(c, *setup.color.hdr);
  // Une image clé toutes les 4 s environ : on se déplace vite dans le fichier.
  const double fps =
      setup.fps_num > 0 && setup.fps_den > 0 ? static_cast<double>(setup.fps_num) / setup.fps_den : 24.0;
  c.gop_size = std::clamp(static_cast<int>(std::lround(fps * 4.0)), 1, 600);
  c.flags |= AV_CODEC_FLAG_GLOBAL_HEADER;  // MKV : en-têtes de flux dans CodecPrivate
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
    std::expected<Opened, std::string> opened = std::unexpected(std::string());
    for (const std::vector<EncoderOption>& recent : candidate.recent) {
      std::vector<EncoderOption> options = candidate.options;
      options.insert(options.end(), recent.begin(), recent.end());
      opened = attempt(*codec, candidate, options, setup);
      if (opened) {
        log->info("{} : options récentes {}", candidate.name, describe(recent));
        break;
      }
      log->debug("{} refuse les options {} ({})", candidate.name, describe(recent), opened.error());
    }
    if (!opened) opened = attempt(*codec, candidate, candidate.options, setup);
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
    encoder.upload_ = opened->upload;
    encoder.shared_ = opened->shared;
    encoder.label_ = std::format("{} ({})", candidate.label, candidate.name);
    encoder.hw_frame_.reset(av_frame_alloc());
    encoder.packet_.reset(av_packet_alloc());
    if (!encoder.hw_frame_ || !encoder.packet_)
      return refuse(RefusalCode::EncodeFailed, "encodeur : mémoire");
    log->info("encodeur {} : {}x{} {}{}", encoder.label_, setup.width, setup.height,
              av_get_pix_fmt_name(pixel_format(encoder.layout_)),
              encoder.upload_ == Upload::Cuda ? " en mémoire CUDA"
              : encoder.shared_               ? " sur l'appareil du rendu"
                                              : "");
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
  // Une image hôte pour un encodeur qui lit l'appareil : chargée sur une surface du contexte
  // d'images ; une image déjà au format de l'appareil part telle quelle.
  if (frame != nullptr && upload_ != Upload::Host && frame->format != ctx_->pix_fmt) {
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
