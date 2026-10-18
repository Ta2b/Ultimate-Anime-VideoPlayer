#include "media/frame_data.h"

#include <cstddef>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
}

#include "media/color.h"

namespace ajp::media {
namespace {

std::optional<FrameMemory> memory_of(const AVFrame& frame) {
  if (frame.hw_frames_ctx == nullptr) return FrameMemory::Host;
  const auto& frames = *reinterpret_cast<const AVHWFramesContext*>(frame.hw_frames_ctx->data);
  switch (frames.device_ctx->type) {
    case AV_HWDEVICE_TYPE_CUDA:
      return FrameMemory::Cuda;
    case AV_HWDEVICE_TYPE_VULKAN:
      return FrameMemory::Vulkan;
    case AV_HWDEVICE_TYPE_D3D11VA:
      return FrameMemory::D3d11;
    case AV_HWDEVICE_TYPE_VAAPI:
      return FrameMemory::VaApi;
    case AV_HWDEVICE_TYPE_VIDEOTOOLBOX:
      return FrameMemory::VideoToolbox;
    default:
      return std::nullopt;
  }
}

AVPixelFormat sw_format_of(const AVFrame& frame) {
  if (frame.hw_frames_ctx == nullptr) return static_cast<AVPixelFormat>(frame.format);
  return reinterpret_cast<const AVHWFramesContext*>(frame.hw_frames_ctx->data)->sw_format;
}

template <class Handle>
std::uintptr_t handle_value(Handle handle) {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<std::uintptr_t>(handle);
  } else {
    return static_cast<std::uintptr_t>(handle);
  }
}

// Plans d'une image matérielle, selon l'API :
// Cuda : un CUdeviceptr par plan, pas en octets par native_pitch ;
// Vulkan : plan 0 = la VkImage multiplan de l'AVVkFrame (une seule, sinon refusée) ;
// D3D11 : plan 0 = ID3D11Texture2D*, plan 1 = index de tranche ;
// VA-API : plan 0 = VASurfaceID ; VideoToolbox : plan 0 = CVPixelBufferRef.
class AvFrameData final : public FrameData {
 public:
  AvFrameData(FramePtr frame, FrameMemory memory, AVPixelFormat sw_format, ReadyEvent ready)
      : frame_(std::move(frame)), memory_(memory), sw_format_(sw_format), ready_(std::move(ready)) {}

  FrameMemory memory() const override { return memory_; }

  int plane_count() const override {
    switch (memory_) {
      case FrameMemory::Host:
      case FrameMemory::Cuda:
        return av_pix_fmt_count_planes(sw_format_);
      case FrameMemory::D3d11:
        return 2;
      case FrameMemory::Vulkan:
      case FrameMemory::VaApi:
      case FrameMemory::VideoToolbox:
        return 1;
    }
    return 0;
  }

  std::span<const std::byte> host_plane(int plane) const override {
    if (memory_ != FrameMemory::Host || plane < 0 || plane >= plane_count()) return {};
    const auto i = static_cast<std::size_t>(plane);
    const int rows = plane_rows(plane);
    const int row_bytes = av_image_get_linesize(sw_format_, frame_->width, plane);
    if (rows <= 0 || row_bytes <= 0) return {};
    const auto stride = static_cast<std::size_t>(frame_->linesize[i]);
    const std::size_t size =
        stride * static_cast<std::size_t>(rows - 1) + static_cast<std::size_t>(row_bytes);
    return std::as_bytes(std::span(frame_->data[i], size));
  }

  std::size_t host_stride(int plane) const override {
    if (memory_ != FrameMemory::Host || plane < 0 || plane >= plane_count()) return 0;
    return static_cast<std::size_t>(frame_->linesize[static_cast<std::size_t>(plane)]);
  }

  std::uintptr_t native_plane(int plane) const override {
    if (plane < 0 || plane >= plane_count()) return 0;
    const auto i = static_cast<std::size_t>(plane);
    switch (memory_) {
      case FrameMemory::Host:
        return 0;
      case FrameMemory::Cuda:
      case FrameMemory::D3d11:
        return reinterpret_cast<std::uintptr_t>(frame_->data[i]);
      case FrameMemory::Vulkan:
        return handle_value(reinterpret_cast<const AVVkFrame*>(frame_->data[0])->img[i]);
      case FrameMemory::VaApi:
      case FrameMemory::VideoToolbox:
        return reinterpret_cast<std::uintptr_t>(frame_->data[3]);
    }
    return 0;
  }

  std::size_t native_pitch(int plane) const override {
    if (memory_ != FrameMemory::Cuda || plane < 0 || plane >= plane_count()) return 0;
    return static_cast<std::size_t>(frame_->linesize[static_cast<std::size_t>(plane)]);
  }

  std::uintptr_t ready_event() const override { return ready_.handle; }
  std::uint64_t ready_value() const override { return ready_.value; }

 private:
  int plane_rows(int plane) const {
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(sw_format_);
    if (desc == nullptr) return 0;
    const bool chroma = plane == 1 || plane == 2;
    return chroma ? AV_CEIL_RSHIFT(frame_->height, desc->log2_chroma_h) : frame_->height;
  }

  FramePtr frame_;
  FrameMemory memory_;
  AVPixelFormat sw_format_;
  ReadyEvent ready_;
};

// Sémaphore timeline de l'image et sa valeur de fin de décodage, lus sous le verrou de
// l'AVVkFrame comme hwcontext_vulkan.h le demande. Une seule VkImage : une seule attente.
std::expected<ReadyEvent, Refusal> vulkan_ready(const AVFrame& frame) {
  auto* frames = reinterpret_cast<AVHWFramesContext*>(frame.hw_frames_ctx->data);
  auto* context = static_cast<AVVulkanFramesContext*>(frames->hwctx);
  auto* image = reinterpret_cast<AVVkFrame*>(frame.data[0]);
  if (image == nullptr) return refuse(RefusalCode::FormatUnsupported, "image Vulkan sans AVVkFrame");
  if (image->img[1] != VK_NULL_HANDLE) {
    return refuse(RefusalCode::FormatUnsupported,
                  "image Vulkan à plusieurs VkImage : une attente par image, hors du contrat de FrameData");
  }
  if (context->lock_frame != nullptr) context->lock_frame(frames, image);
  ReadyEvent ready{.handle = handle_value(image->sem[0]), .value = image->sem_value[0], .owner = nullptr};
  if (context->unlock_frame != nullptr) context->unlock_frame(frames, image);
  return ready;
}

}  // namespace

std::expected<DecodedFrame, Refusal> to_decoded_frame(FramePtr frame, std::int64_t pts_us,
                                                      std::int64_t duration_us, ReadyEvent ready) {
  const std::optional<FrameMemory> memory = memory_of(*frame);
  if (!memory) return refuse(RefusalCode::FormatUnsupported, "image d'un type d'appareil inconnu");
  const AVPixelFormat sw_format = sw_format_of(*frame);
  auto format = map_pixel_format(sw_format);
  if (!format) return std::unexpected(format.error());
  if (*memory == FrameMemory::Host) {
    for (int i = 0; i < av_pix_fmt_count_planes(sw_format); ++i) {
      if (frame->linesize[static_cast<std::size_t>(i)] <= 0) {
        return refuse(RefusalCode::FormatUnsupported, "image retournée (pas de ligne négatif)");
      }
    }
  }
  auto color = color_info(*frame, sw_format);
  if (!color) return std::unexpected(color.error());
  if (*memory == FrameMemory::Vulkan) {
    auto own = vulkan_ready(*frame);
    if (!own) return std::unexpected(own.error());
    ready = std::move(*own);
  }

  DecodedFrame decoded{
      .width = frame->width,
      .height = frame->height,
      .format = *format,
      .color = *color,
      .pts_us = pts_us,
      .duration_us = duration_us,
  };
  decoded.data = std::make_shared<const AvFrameData>(std::move(frame), *memory, sw_format, std::move(ready));
  return decoded;
}

}  // namespace ajp::media
