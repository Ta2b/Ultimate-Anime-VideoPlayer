// FrameExport, chemin Vulkan Video : les images de l'encodeur, sur l'appareil de l'écran
// (vulkan_share.h), rendues dans leurs plans. Une image du pool de FFmpeg est une VkImage à deux
// plans (NV12 ou P010) ; chaque plan s'enveloppe en texture libplacebo (pl_vulkan_wrap), rendable
// parce que l'encodeur a demandé COLOR_ATTACHMENT à la création. Autour du rendu, le sémaphore
// timeline de l'image, comme hwcontext_vulkan.h le prescrit : libplacebo attend sa valeur
// (release), rend, puis la signale (hold) ; l'image porte la valeur suivante et sa disposition,
// et l'encodeur l'attend à son tour. Les plans ont chacun leur disposition (aspects distincts) :
// les deux enveloppes d'une même image ne se gênent pas.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <format>
#include <optional>
#include <utility>

#include "app/play/placebo_export.h"

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

#include "app/play/placebo_impl.h"
#include "core/log.h"

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Les formats des deux plans d'une image 4:2:0 semi-planaire, par son format.
struct PlaneFormats {
  VkFormat luma;
  VkFormat chroma;
  bool packed;  // 10 bits dans 16 placés par la texture (rx10)
};

std::optional<PlaneFormats> plane_formats_of(VkFormat image) {
  switch (image) {
    case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
      return PlaneFormats{VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM, false};
    case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
      return PlaneFormats{VK_FORMAT_R10X6_UNORM_PACK16, VK_FORMAT_R10X6G10X6_UNORM_2PACK16, true};
    case VK_FORMAT_G16_B16R16_2PLANE_420_UNORM:
      return PlaneFormats{VK_FORMAT_R16_UNORM, VK_FORMAT_R16G16_UNORM, false};
    default:
      return std::nullopt;
  }
}

struct FrameParts {
  AVHWFramesContext* frames = nullptr;
  AVVulkanFramesContext* context = nullptr;
  AVVkFrame* image = nullptr;
};

FrameParts parts_of(AVBufferRef& frames, const AVFrame& frame) {
  auto* fc = reinterpret_cast<AVHWFramesContext*>(frames.data);
  return {.frames = fc,
          .context = static_cast<AVVulkanFramesContext*>(fc->hwctx),
          .image = reinterpret_cast<AVVkFrame*>(frame.data[0])};
}

}  // namespace

std::expected<void, Refusal> FrameExport::open_vulkan(pl_gpu gpu, AVBufferRef* frames) {
  if (setup_.layout == exporter::Layout::Yuv420p10)
    return refuse(RefusalCode::FeatureUnavailable, "disposition en trois plans");
  AVBufferRef* ref = av_buffer_ref(frames);
  if (ref == nullptr) return refuse(RefusalCode::GpuMemoryExhausted, "référence du contexte d'images");
  vulkan_ = VulkanFrames{.frames = ref};
  // Une image d'essai : ses plans doivent s'envelopper et se rendre ; elle retourne au pool.
  AVFrame* probe = av_frame_alloc();
  if (probe == nullptr) return refuse(RefusalCode::GpuMemoryExhausted, "image d'essai");
  if (av_hwframe_get_buffer(ref, probe, 0) < 0) {
    av_frame_free(&probe);
    return refuse(RefusalCode::VendorCallFailed, "av_hwframe_get_buffer");
  }
  auto planes = wrap_planes(*vulkan_, gpu, *probe);
  av_frame_free(&probe);
  if (!planes) return std::unexpected(planes.error());
  return {};
}

void FrameExport::close_vulkan(pl_gpu gpu) {
  if (!vulkan_) return;
  pl_gpu_finish(gpu);  // les rendus en cours écrivent dans les images du pool
  for (VulkanFrames::Wrap& wrap : vulkan_->wraps) {
    for (pl_tex& tex : wrap.planes) pl_tex_destroy(gpu, &tex);
  }
  for (Pending& pending : vulkan_->pending) av_frame_free(&pending.frame);
  av_buffer_unref(&vulkan_->frames);
  vulkan_.reset();
}

std::expected<std::array<pl_tex, 2>, Refusal> FrameExport::wrap_planes(VulkanFrames& v, pl_gpu gpu,
                                                                       const AVFrame& frame) {
  const FrameParts parts = parts_of(*v.frames, frame);
  const auto image = std::bit_cast<std::uint64_t>(parts.image->img[0]);
  const auto known = std::ranges::find(v.wraps, image, &VulkanFrames::Wrap::image);
  if (known != v.wraps.end()) return known->planes;
  const std::optional<PlaneFormats> formats = plane_formats_of(parts.context->format[0]);
  if (!formats || parts.context->format[1] != VK_FORMAT_UNDEFINED) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("image de l'encodeur au format Vulkan {}, deux plans 4:2:0 attendus",
                              static_cast<int>(parts.context->format[0])));
  }
  VulkanFrames::Wrap wrap{.image = image};
  const std::array<VkFormat, 2> plane_formats{formats->luma, formats->chroma};
  for (std::size_t p = 0; p < 2; ++p) {
    pl_vulkan_wrap_params params{};
    params.image = parts.image->img[0];
    params.aspect = p == 0 ? VK_IMAGE_ASPECT_PLANE_0_BIT : VK_IMAGE_ASPECT_PLANE_1_BIT;
    params.width = p == 0 ? frame.width : frame.width / 2;
    params.height = p == 0 ? frame.height : frame.height / 2;
    params.format = plane_formats[p];
    params.usage = parts.context->usage;
    wrap.planes[p] = pl_vulkan_wrap(gpu, &params);
    if (wrap.planes[p] == nullptr || !wrap.planes[p]->params.renderable) {
      const bool wrapped = wrap.planes[p] != nullptr;
      for (pl_tex& tex : wrap.planes) pl_tex_destroy(gpu, &tex);
      return refuse(RefusalCode::FeatureUnavailable,
                    wrapped ? std::format("plan {} de l'encodeur non rendable sur ce GPU", p)
                            : std::format("plan {} de l'encodeur sans texture libplacebo", p));
    }
  }
  v.wraps.push_back(wrap);
  return wrap.planes;
}

FrameExport::Converted FrameExport::convert_vulkan(VulkanFrames& v, const RingFrame& frame) {
  PlaceboScreen::Impl& s = *screen_.impl_;
  const pl_gpu gpu = s.vk->gpu;
  const auto given_back = [&](Refusal failure) {
    return Converted{.freed = s.ring.retire(gpu, frame), .failure = std::move(failure)};
  };
  if (v.pending.size() >= buffers_.size())
    return given_back(Refusal{RefusalCode::SequenceOrder, "export : aucune image libre"});
  AVFrame* out = av_frame_alloc();
  if (out == nullptr) return given_back(Refusal{RefusalCode::GpuMemoryExhausted, "export : image"});
  if (av_hwframe_get_buffer(v.frames, out, 0) < 0) {
    av_frame_free(&out);
    return given_back(Refusal{RefusalCode::VendorCallFailed, "export : av_hwframe_get_buffer"});
  }
  auto planes = wrap_planes(v, gpu, *out);
  if (!planes) {
    av_frame_free(&out);
    return given_back(std::move(planes.error()));
  }
  const FrameParts parts = parts_of(*v.frames, *out);
  AVVkFrame& image = *parts.image;
  if (parts.context->lock_frame != nullptr) parts.context->lock_frame(parts.frames, parts.image);
  // libplacebo attend le dernier usage de l'image (l'encodeur, ou rien) avant d'y rendre.
  for (const pl_tex tex : *planes) {
    pl_vulkan_release_params release{};
    release.tex = tex;
    release.layout = image.layout[0];
    release.qf = VK_QUEUE_FAMILY_IGNORED;
    release.semaphore = pl_vulkan_sem{.sem = image.sem[0], .value = image.sem_value[0]};
    pl_vulkan_release_ex(gpu, &release);
  }
  const bool packed = plane_formats_of(parts.context->format[0]).value_or(PlaneFormats{}).packed;
  Converted converted = render(frame, std::span<const pl_tex>(*planes), packed);
  // Reprise des plans, rendus ou non : chaque enveloppe signale une valeur de plus (deux signaux
  // d'une même soumission vont croissant), l'image passe à la dernière valeur réellement signalée.
  // Un plan dont la reprise échoue ne signale rien : la valeur n'avance pas pour lui, sinon
  // l'encodeur attendrait pour toujours un signal qui ne vient pas.
  VkImageLayout layout = image.layout[0];
  bool held = true;
  std::uint64_t signalled = image.sem_value[0];
  for (std::size_t p = 0; p < planes->size(); ++p) {
    pl_vulkan_hold_params hold{};
    hold.tex = (*planes)[p];
    hold.out_layout = &layout;
    hold.qf = VK_QUEUE_FAMILY_IGNORED;
    const std::uint64_t value = image.sem_value[0] + 1 + p;
    hold.semaphore = pl_vulkan_sem{.sem = image.sem[0], .value = value};
    if (pl_vulkan_hold_ex(gpu, &hold)) {
      signalled = value;
    } else {
      held = false;
    }
  }
  image.sem_value[0] = signalled;
  image.layout[0] = layout;
  image.access[0] = VK_ACCESS_2_NONE;  // la soumission signalée rend tout visible
  if (parts.context->unlock_frame != nullptr) parts.context->unlock_frame(parts.frames, parts.image);
  pl_gpu_flush(gpu);  // le signal part maintenant : l'encodeur n'attend pas la prochaine image
  if (!converted.failure && !held)
    converted.failure = Refusal{RefusalCode::VendorCallFailed, "export : pl_vulkan_hold_ex"};
  if (converted.failure) {
    av_frame_free(&out);
    return converted;
  }
  v.pending.push_back({.frame = out, .pts_us = frame.pts_us});
  return converted;
}

bool FrameExport::deliver_vulkan(VulkanFrames& v, const std::function<bool(exporter::Picture&&)>& push) {
  if (v.pending.empty()) return false;
  const Pending& next = v.pending.front();
  exporter::Picture picture;
  picture.pts_us = next.pts_us;
  picture.frame = next.frame;
  if (!push(std::move(picture))) return false;
  v.pending.pop_front();
  return true;
}

}  // namespace ajp::app::play
