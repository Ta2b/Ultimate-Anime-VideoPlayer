#include "media/hw_device.h"

#include <algorithm>
#include <bit>
#include <format>
#include <string>
#include <utility>

// hwcontext_cuda.h n'a besoin que de CUcontext et CUstream : on les déclare
// comme nv-codec-headers (dynlink_cuda.h), sans en-tête du kit CUDA.
#define CUDA_VERSION 7050
using CUcontext = struct CUctx_st*;
using CUstream = struct CUstream_st*;

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/hwcontext_vulkan.h>
}

namespace ajp::media {
namespace {

constexpr int kMaxVulkanDevices = 8;

AVHWDeviceType device_type(HwApi api) {
  switch (api) {
    case HwApi::Cuda:
      return AV_HWDEVICE_TYPE_CUDA;
    case HwApi::Vulkan:
      return AV_HWDEVICE_TYPE_VULKAN;
    case HwApi::D3d11va:
      return AV_HWDEVICE_TYPE_D3D11VA;
    case HwApi::Vaapi:
      return AV_HWDEVICE_TYPE_VAAPI;
    case HwApi::VideoToolbox:
      return AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
    case HwApi::Software:
      break;
  }
  return AV_HWDEVICE_TYPE_NONE;
}

std::unexpected<Refusal> refuse_hw(HwApi api, std::string_view why) {
  return refuse(RefusalCode::HwDecodeUnavailable, std::format("{} : {}", name(api), why));
}

// Premier décodeur du codec qui accepte un contexte d'appareil de ce type.
std::optional<std::pair<const AVCodec*, AVPixelFormat>> find_hw_decoder(AVCodecID id, AVHWDeviceType type) {
  void* it = nullptr;
  while (const AVCodec* codec = av_codec_iterate(&it)) {
    if (codec->id != id || av_codec_is_decoder(codec) == 0) continue;
    for (int i = 0;; ++i) {
      const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
      if (config == nullptr) break;
      if (config->device_type == type && (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0) {
        return std::pair{codec, config->pix_fmt};
      }
    }
  }
  return std::nullopt;
}

std::expected<BufferPtr, Refusal> create_device(HwApi api, const std::string& device, int flags) {
  AVBufferRef* ref = nullptr;
  const char* selector = device.empty() ? nullptr : device.c_str();
  if (const int err = av_hwdevice_ctx_create(&ref, device_type(api), selector, nullptr, flags); err < 0) {
    return refuse_hw(api, std::format("création de l'appareil « {} » : {}", device, av_error_text(err)));
  }
  return BufferPtr(ref);
}

// Contexte primaire de l'appareil dont l'UUID égale celui du rendu : il partage
// ses allocations avec nos noyaux et TensorRT (mpv crée au contraire un
// contexte dédié, hwdec_cuda_vk.c:341).
std::expected<HwDevice, Refusal> open_cuda(HwDevice hw, const DecoderOptions& options) {
  if (!options.render_device) return refuse_hw(HwApi::Cuda, "UUID de l'appareil de rendu inconnu");
  auto driver = CudaDriver::load();
  if (!driver) return std::unexpected(driver.error());
  auto ordinal = (*driver)->ordinal_of(*options.render_device);
  if (!ordinal) return std::unexpected(ordinal.error());
  auto device = create_device(HwApi::Cuda, std::to_string(*ordinal), AV_CUDA_USE_PRIMARY_CONTEXT);
  if (!device) return std::unexpected(device.error());
  auto stream = CudaDecodeStream::attach(std::move(*driver), **device);
  if (!stream) return std::unexpected(stream.error());
  hw.device = std::move(*device);
  hw.stream = std::move(*stream);
  return hw;
}

// FFmpeg ne choisit un appareil Vulkan que par index ou par nom : on essaie
// les index jusqu'à trouver l'UUID du rendu.
std::expected<HwDevice, Refusal> open_vulkan(HwDevice hw, const DecoderOptions& options) {
  if (!options.render_device) {
    auto device = create_device(HwApi::Vulkan, {}, 0);
    if (!device) return std::unexpected(device.error());
    hw.device = std::move(*device);
    return hw;
  }
  int tried = 0;
  for (; tried < kMaxVulkanDevices; ++tried) {
    auto device = create_device(HwApi::Vulkan, std::to_string(tried), 0);
    if (!device) {
      if (tried == 0) return std::unexpected(device.error());  // aucun appareil Vulkan : la vraie cause
      break;
    }
    if (vulkan_device_uuid(**device) == options.render_device) {
      hw.device = std::move(*device);
      return hw;
    }
  }
  return refuse_hw(HwApi::Vulkan, std::format("aucun des {} appareils n'a l'UUID du rendu", tried));
}

}  // namespace

std::string_view name(HwApi api) {
  switch (api) {
    case HwApi::Cuda:
      return "cuda";
    case HwApi::Vulkan:
      return "vulkan";
    case HwApi::D3d11va:
      return "d3d11va";
    case HwApi::Vaapi:
      return "vaapi";
    case HwApi::VideoToolbox:
      return "videotoolbox";
    case HwApi::Software:
      return "logiciel";
  }
  return "inconnue";
}

std::vector<HwApi> default_attempts() {
#if defined(_WIN32)
  return {HwApi::Cuda, HwApi::D3d11va, HwApi::Vulkan, HwApi::Software};
#elif defined(__APPLE__)
  return {HwApi::VideoToolbox, HwApi::Software};
#else
  return {HwApi::Cuda, HwApi::Vaapi, HwApi::Vulkan, HwApi::Software};
#endif
}

bool is_vulkan_video_codec(AVCodecID codec) {
  return codec == AV_CODEC_ID_H264 || codec == AV_CODEC_ID_HEVC || codec == AV_CODEC_ID_AV1 ||
         codec == AV_CODEC_ID_VP9;
}

std::expected<HwDevice, Refusal> open_hw_device(AVCodecID codec, HwApi api, const DecoderOptions& options) {
  const AVHWDeviceType type = device_type(api);
  if (type == AV_HWDEVICE_TYPE_NONE) return refuse_hw(api, "pas une API matérielle");
  // Avant toute recherche : le refus est le même que FFmpeg ait ou non ce
  // décodeur (ffv1_vulkan n'existe que si configure a trouvé un glslc récent).
  if (api == HwApi::Vulkan && !is_vulkan_video_codec(codec) && !options.vulkan_compute) {
    return refuse_hw(api, std::format("{} hors Vulkan Video ; son décodeur par compute shaders "
                                      "demande DecoderOptions::vulkan_compute",
                                      avcodec_get_name(codec)));
  }
  const auto decoder = find_hw_decoder(codec, type);
  if (!decoder) return refuse_hw(api, std::format("aucun décodeur {} ne l'accepte", avcodec_get_name(codec)));

  HwDevice hw{.api = api, .codec = decoder->first, .format = decoder->second};
  if (options.devices != nullptr) {
    if (AVBufferRef* shared = options.devices->device(api); shared != nullptr) {
      hw.device.reset(shared);
      if (reinterpret_cast<const AVHWDeviceContext*>(shared->data)->type != type) {
        return refuse_hw(api, "l'appareil partagé est d'un autre type");
      }
      return hw;  // partagé : son propriétaire règle flux et synchronisation
    }
  }
  switch (api) {
    case HwApi::Cuda:
      return open_cuda(std::move(hw), options);
    case HwApi::Vulkan:
      return open_vulkan(std::move(hw), options);
    default: {
      // D3D11VA, VA-API, VideoToolbox : adaptateur par défaut (voir open_questions).
      auto device = create_device(api, {}, 0);
      if (!device) return std::unexpected(device.error());
      hw.device = std::move(*device);
      return hw;
    }
  }
}

std::optional<DeviceUuid> vulkan_device_uuid(const AVBufferRef& device) {
  const auto& ctx = *reinterpret_cast<const AVHWDeviceContext*>(device.data);
  if (ctx.type != AV_HWDEVICE_TYPE_VULKAN) return std::nullopt;
  const auto& vk = *static_cast<const AVVulkanDeviceContext*>(ctx.hwctx);
  const auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
      vk.get_proc_addr(vk.inst, "vkGetPhysicalDeviceProperties2"));
  if (get_properties == nullptr) return std::nullopt;
  VkPhysicalDeviceIDProperties ids{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                         .pNext = &ids};
  get_properties(vk.phys_dev, &properties);
  DeviceUuid uuid{};
  std::ranges::copy(ids.deviceUUID, uuid.begin());
  return uuid;
}

std::expected<std::unique_ptr<CudaDecodeStream>, Refusal> CudaDecodeStream::attach(
    std::shared_ptr<CudaDriver> driver, AVBufferRef& device) {
  auto& cuda = *static_cast<AVCUDADeviceContext*>(reinterpret_cast<AVHWDeviceContext*>(device.data)->hwctx);
  const auto context = reinterpret_cast<std::uintptr_t>(cuda.cuda_ctx);
  auto stream = driver->create_stream(context);
  if (!stream) return std::unexpected(stream.error());
  // NVDEC lit ce flux à la création du décodeur (decoder->stream) : la copie de
  // sécurité et la libération de la surface s'y font, sans toucher au flux 0.
  cuda.stream = std::bit_cast<CUstream>(*stream);
  return std::unique_ptr<CudaDecodeStream>(new CudaDecodeStream(std::move(driver), device, context, *stream));
}

CudaDecodeStream::CudaDecodeStream(std::shared_ptr<CudaDriver> driver, AVBufferRef& device,
                                   std::uintptr_t context, std::uintptr_t stream)
    : driver_(std::move(driver)), device_(av_buffer_ref(&device)), context_(context), stream_(stream) {}

CudaDecodeStream::~CudaDecodeStream() {
  // Le contexte d'appareil peut survivre au décodeur (images encore tenues) :
  // il ne doit plus désigner un flux détruit.
  if (device_) {
    auto& cuda =
        *static_cast<AVCUDADeviceContext*>(reinterpret_cast<AVHWDeviceContext*>(device_->data)->hwctx);
    cuda.stream = nullptr;
  }
  driver_->destroy_stream(context_, stream_);
}

std::expected<ReadyEvent, Refusal> CudaDecodeStream::record() const {
  return driver_->record(context_, stream_);
}

}  // namespace ajp::media
