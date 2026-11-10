#include "media/frame_data.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

extern "C" {
#include <libswscale/swscale.h>

#include <libavutil/buffer.h>
#include <libavutil/cpu.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include "media/av_util.h"
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

// Ce qu'une image Vulkan apporte : son contexte d'images, qui porte le verrou, et l'AVVkFrame.
struct VulkanParts {
  AVHWFramesContext* frames = nullptr;
  AVVulkanFramesContext* context = nullptr;
  AVVkFrame* image = nullptr;

  void lock() const {
    if (context->lock_frame != nullptr) context->lock_frame(frames, image);
  }
  void unlock() const {
    if (context->unlock_frame != nullptr) context->unlock_frame(frames, image);
  }
};

VulkanParts vulkan_parts(const AVFrame& frame) {
  auto* frames = reinterpret_cast<AVHWFramesContext*>(frame.hw_frames_ctx->data);
  return {.frames = frames,
          .context = static_cast<AVVulkanFramesContext*>(frames->hwctx),
          .image = reinterpret_cast<AVVkFrame*>(frame.data[0])};
}

// Plans d'une image matérielle, selon l'API :
// Cuda : un CUdeviceptr par plan, pas en octets par native_pitch ;
// Vulkan : plan 0 = la VkImage multiplan de l'AVVkFrame (une seule, sinon refusée), lue
// entre begin_vulkan_access et end_vulkan_access ;
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

  // Le verrou reste pris jusqu'à end_vulkan_access, comme hwcontext_vulkan.h le demande
  // autour d'une soumission.
  std::expected<VulkanImageAccess, Refusal> begin_vulkan_access() const override {
    if (memory_ != FrameMemory::Vulkan) return FrameData::begin_vulkan_access();
    const VulkanParts vk = vulkan_parts(*frame_);
    vk.lock();
    const AVVkFrame& image = *vk.image;
    return VulkanImageAccess{
        .semaphore = handle_value(image.sem[0]),
        .wait_value = image.sem_value[0],
        .signal_value = image.sem_value[0] + 1,
        .state = {.layout = static_cast<std::uint32_t>(image.layout[0]),
                  .queue_family = image.queue_family[0]},
        .usage = static_cast<std::uint32_t>(vk.context->usage),
        .flags = static_cast<std::uint32_t>(vk.context->img_flags),
        .format = static_cast<std::uint32_t>(vk.context->format[0]),
    };
  }

  void end_vulkan_access(const VulkanImageState& after, bool submitted) const override {
    if (memory_ != FrameMemory::Vulkan) return;
    const VulkanParts vk = vulkan_parts(*frame_);
    if (submitted) {
      AVVkFrame& image = *vk.image;
      image.sem_value[0] += 1;
      image.layout[0] = static_cast<VkImageLayout>(after.layout);
      image.queue_family[0] = after.queue_family;
      image.access[0] = VK_ACCESS_2_NONE;  // la soumission signalée rend tout visible
    }
    vk.unlock();
  }

  std::uintptr_t native_pool_id() const override {
    if (frame_->hw_frames_ctx == nullptr) return 0;
    return reinterpret_cast<std::uintptr_t>(frame_->hw_frames_ctx->data);
  }

  // Une référence au contexte d'images de FFmpeg, qui porte le pool : le pointeur rendu est
  // native_pool_id.
  std::shared_ptr<const void> native_pool() const override {
    if (frame_->hw_frames_ctx == nullptr) return nullptr;
    AVBufferRef* ref = av_buffer_ref(frame_->hw_frames_ctx);
    if (ref == nullptr) return nullptr;
    return {ref->data, [ref](const void* /*data*/) mutable { av_buffer_unref(&ref); }};
  }

  const AVFrame& av() const { return *frame_; }

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
  const VulkanParts vk = vulkan_parts(frame);
  if (vk.image == nullptr) return refuse(RefusalCode::FormatUnsupported, "image Vulkan sans AVVkFrame");
  if (vk.image->img[1] != VK_NULL_HANDLE) {
    return refuse(RefusalCode::FormatUnsupported,
                  "image Vulkan à plusieurs VkImage : une attente par image, hors du contrat de FrameData");
  }
  vk.lock();
  ReadyEvent ready{
      .handle = handle_value(vk.image->sem[0]), .value = vk.image->sem_value[0], .owner = nullptr};
  vk.unlock();
  return ready;
}

}  // namespace

const AVFrame* av_frame_of(const FrameData& data) {
  const auto* own = dynamic_cast<const AvFrameData*>(&data);
  return own == nullptr ? nullptr : &own->av();
}

namespace {

struct SwsFree {
  void operator()(SwsContext* context) const { sws_freeContext(context); }
};

struct PoolFree {
  void operator()(AVBufferPool* pool) const { av_buffer_pool_uninit(&pool); }
};

// Threads de swscale pour une image : la conversion tient sur le thread source, qui décode aussi (A15-06).
constexpr int kConvertThreads = 4;
constexpr int kConvertAlign = 64;

// Où swscale pose la chroma qu'il réduit : à gauche en largeur, au centre en hauteur, ce que l'image
// convertie annonce (AVCHROMA_LOC_LEFT). Laissé à lui-même, il la posait au centre en largeur : un quart de
// point de chroma d'écart avec ce que l'ingestion lisait (A15-07). Une source déjà en 4:2:0 n'est pas
// rééchantillonnée, rien n'est posé.
bool place_chroma(SwsContext* ctx, AVPixelFormat from, AVPixelFormat to) {
  const AVPixFmtDescriptor* src = av_pix_fmt_desc_get(from);
  const AVPixFmtDescriptor* dst = av_pix_fmt_desc_get(to);
  if (src == nullptr || dst == nullptr) return false;
  const bool reduced = dst->log2_chroma_w > src->log2_chroma_w || dst->log2_chroma_h > src->log2_chroma_h;
  if (!reduced) return true;
  constexpr int kLeft = 0;
  constexpr int kCenter = 128;  // en 256es de point de luma
  return (src->log2_chroma_w == 0 || av_opt_set_int(ctx, "src_h_chr_pos", kLeft, 0) >= 0) &&
         av_opt_set_int(ctx, "dst_h_chr_pos", kLeft, 0) >= 0 &&
         av_opt_set_int(ctx, "dst_v_chr_pos", kCenter, 0) >= 0;
}

// Le contexte de conversion et le pool des images converties, gardés d'une image à l'autre tant que la
// taille et les formats ne changent pas : une image 4K 4:4:4 n'alloue plus 25 Mo neufs à chaque fois
// (A15-06). Un seul thread source convertit.
struct Converter {
  std::unique_ptr<SwsContext, SwsFree> context;
  std::unique_ptr<AVBufferPool, PoolFree> pool;
  int width = 0;
  int height = 0;
  AVPixelFormat from = AV_PIX_FMT_NONE;
  AVPixelFormat to = AV_PIX_FMT_NONE;
  int bytes = 0;

  bool ready_for(const AVFrame& frame, AVPixelFormat format, AVPixelFormat target) const {
    return context && pool && width == frame.width && height == frame.height && from == format &&
           to == target;
  }

  std::expected<void, Refusal> prepare(const AVFrame& frame, AVPixelFormat format, AVPixelFormat target) {
    if (ready_for(frame, format, target)) return {};
    pool.reset();
    context.reset(sws_alloc_context());
    if (!context) return refuse(RefusalCode::DecodeFailed, "conversion vers 4:2:0 : mémoire insuffisante");
    SwsContext* ctx = context.get();
    const bool set = av_opt_set_int(ctx, "srcw", frame.width, 0) >= 0 &&
                     av_opt_set_int(ctx, "srch", frame.height, 0) >= 0 &&
                     av_opt_set_int(ctx, "src_format", format, 0) >= 0 &&
                     av_opt_set_int(ctx, "dstw", frame.width, 0) >= 0 &&
                     av_opt_set_int(ctx, "dsth", frame.height, 0) >= 0 &&
                     av_opt_set_int(ctx, "dst_format", target, 0) >= 0 &&
                     // Spline et arrondi exact (A49-13) : quand la chroma est réduite (4:2:2, ou 4:4:4 sans
                     // full_host_formats), le bilinéaire était le moins bon filtre de swscale.
                     av_opt_set_int(ctx, "sws_flags", SWS_SPLINE | SWS_ACCURATE_RND, 0) >= 0 &&
                     av_opt_set_int(ctx, "threads", std::min(av_cpu_count(), kConvertThreads), 0) >= 0;
    if (!set || !place_chroma(ctx, format, target) || sws_init_context(ctx, nullptr, nullptr) < 0) {
      context.reset();
      return refuse(RefusalCode::FormatUnsupported, "conversion vers 4:2:0 : contexte swscale refusé");
    }
    bytes = av_image_get_buffer_size(target, frame.width, frame.height, kConvertAlign);
    if (bytes <= 0) return refuse(RefusalCode::FormatUnsupported, "conversion vers 4:2:0 : taille d'image");
    pool.reset(av_buffer_pool_init(static_cast<std::size_t>(bytes), nullptr));
    if (!pool) return refuse(RefusalCode::DecodeFailed, "conversion vers 4:2:0 : mémoire insuffisante");
    width = frame.width;
    height = frame.height;
    from = format;
    to = target;
    return {};
  }
};

// Le format que la chaîne lit pour une image hôte YUV qu'elle ne lit pas telle quelle. Sans full : du 4:2:0
// de même profondeur ou moins, 8 bits en yuv420p, au-delà en yuv420p10. Avec full (la chaîne lit Yuv444p16
// et P016 en mémoire hôte, A15-07 et A49-13) : le 4:4:4 garde sa chroma entière en 16 bits, et le 4:2:0 de
// plus de 10 bits garde sa profondeur en P016 ; le 4:2:2 reste réduit en 4:2:0, comme avant.
AVPixelFormat chain_target(const AVPixFmtDescriptor& desc, bool full) {
  const bool whole_chroma = desc.log2_chroma_w == 0 && desc.log2_chroma_h == 0;
  const bool quarter_chroma = desc.log2_chroma_w == 1 && desc.log2_chroma_h == 1;
  if (full && whole_chroma) return AV_PIX_FMT_YUV444P16LE;
  if (full && quarter_chroma && desc.comp[0].depth > 10) return AV_PIX_FMT_P016LE;
  return desc.comp[0].depth > 8 ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P;
}

// Une image hôte YUV que la chaîne ne lit pas (4:2:2, 4:4:4, 12 bits : H.264 High 4:4:4, VP9 profils 1
// et 3, ProRes) devient un format qu'elle lit (chain_target). Même matrice et même plage. Rend nullptr si
// l'image n'est pas de ce cas.
std::expected<FramePtr, Refusal> to_chain_yuv(const AVFrame& frame, AVPixelFormat format, bool full) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  if (desc == nullptr ||
      (desc->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_PAL)) != 0 ||
      desc->nb_components < 3)
    return FramePtr{};
  const AVPixelFormat target = chain_target(*desc, full);
  thread_local Converter converter;
  if (auto prepared = converter.prepare(frame, format, target); !prepared)
    return std::unexpected(prepared.error());
  FramePtr out(av_frame_alloc());
  if (!out) return refuse(RefusalCode::DecodeFailed, "conversion vers 4:2:0 : mémoire insuffisante");
  out->format = target;
  out->width = frame.width;
  out->height = frame.height;
  out->buf[0] = av_buffer_pool_get(converter.pool.get());
  if (out->buf[0] == nullptr) return refuse(RefusalCode::DecodeFailed, "conversion vers 4:2:0 : tampon");
  if (const int err = av_image_fill_arrays(out->data, out->linesize, out->buf[0]->data, target, frame.width,
                                           frame.height, kConvertAlign);
      err < 0)
    return refuse_av(RefusalCode::DecodeFailed, "conversion vers 4:2:0 : tampon", err);
  out->extended_data = out->data;
  if (const int err = av_frame_copy_props(out.get(), &frame); err < 0)
    return refuse_av(RefusalCode::DecodeFailed, "conversion vers 4:2:0 : propriétés", err);
  // Déjà en 4:2:0 (12 bits ramenés à 10 ou gardés en P016), la chroma n'est pas rééchantillonnée : elle
  // garde sa position. Sinon, celle que swscale produit (A15-07) ; en 4:4:4 elle ne sert pas.
  const bool subsampled = desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1;
  if (!subsampled) out->chroma_location = AVCHROMA_LOC_LEFT;
  // Les formats yuvj (plage pleine) sortent en plage limitée : swscale la convertit.
  if (std::string_view(desc->name).starts_with("yuvj")) out->color_range = AVCOL_RANGE_MPEG;
  // L'interface par images : la seule où swscale découpe la conversion entre ses threads.
  if (const int err = sws_scale_frame(converter.context.get(), out.get(), &frame); err < 0)
    return refuse_av(RefusalCode::DecodeFailed, "conversion vers 4:2:0 : sws_scale_frame", err);
  return out;
}

}  // namespace

std::expected<DecodedFrame, Refusal> to_decoded_frame(FramePtr frame, std::int64_t pts_us,
                                                      std::int64_t duration_us, ReadyEvent ready,
                                                      bool full_host_formats) {
  const std::optional<FrameMemory> memory = memory_of(*frame);
  if (!memory) return refuse(RefusalCode::FormatUnsupported, "image d'un type d'appareil inconnu");
  if (*memory == FrameMemory::Host && !map_pixel_format(static_cast<AVPixelFormat>(frame->format))) {
    auto converted = to_chain_yuv(*frame, static_cast<AVPixelFormat>(frame->format), full_host_formats);
    if (!converted) return std::unexpected(converted.error());
    if (*converted) frame = std::move(*converted);
  }
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
      .width = chain_dimension(frame->width),
      .height = chain_dimension(frame->height),
      .format = *format,
      .color = *color,
      .pts_us = pts_us,
      .duration_us = duration_us,
  };
  decoded.data = std::make_shared<const AvFrameData>(std::move(frame), *memory, sw_format, std::move(ready));
  return decoded;
}

}  // namespace ajp::media
