#include "app/play/vulkan_share.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::string av_text(int error) {
  std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
  av_strerror(error, text.data(), text.size());
  return text.data();
}

// FFmpeg verrouille chaque file par ces fonctions, comme le backend Vulkan : libplacebo tient le
// verrou, et user_opaque porte le pl_vulkan.
void lock_queue(AVHWDeviceContext* ctx, std::uint32_t family, std::uint32_t index) {
  const auto vk = static_cast<pl_vulkan>(ctx->user_opaque);
  vk->lock_queue(vk, family, index);
}
void unlock_queue(AVHWDeviceContext* ctx, std::uint32_t family, std::uint32_t index) {
  const auto vk = static_cast<pl_vulkan>(ctx->user_opaque);
  vk->unlock_queue(vk, family, index);
}

}  // namespace

std::expected<std::unique_ptr<VulkanShare>, Refusal> VulkanShare::create(pl_vk_inst inst, pl_vulkan vk) {
  AVBufferRef* ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
  if (ref == nullptr) return refuse(RefusalCode::GpuMemoryExhausted, "av_hwdevice_ctx_alloc");
  std::unique_ptr<VulkanShare> share(new VulkanShare());
  share->device_ = ref;
  auto* ctx = reinterpret_cast<AVHWDeviceContext*>(ref->data);
  auto* hw = static_cast<AVVulkanDeviceContext*>(ctx->hwctx);
  ctx->user_opaque = const_cast<pl_vulkan_t*>(vk);
  hw->get_proc_addr = vk->get_proc_addr;
  hw->inst = vk->instance;
  hw->phys_dev = vk->phys_device;
  hw->act_dev = vk->device;
  // Les fonctionnalités activées, telles que libplacebo les garde : la chaîne vit avec lui.
  hw->device_features = *vk->features;
  hw->enabled_inst_extensions = inst->extensions;
  hw->nb_enabled_inst_extensions = inst->num_extensions;
  hw->enabled_dev_extensions = vk->extensions;
  hw->nb_enabled_dev_extensions = vk->num_extensions;
  // Dépréciés par FFmpeg 7.1 au profit de VK_KHR_internally_synchronized_queues, mais toujours
  // honorés avant libavutil 62 : sans eux, FFmpeg verrouillerait ses propres mutex, et libplacebo
  // soumettrait sur les mêmes files à côté.
#if FF_API_VULKAN_SYNC_QUEUES
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  hw->lock_queue = lock_queue;
  hw->unlock_queue = unlock_queue;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
#else
#error "libavutil sans lock_queue/unlock_queue : FFmpeg et libplacebo soumettraient sur les mêmes files sans verrou (VK_KHR_internally_synchronized_queues à activer)"
#endif

  // Les familles de files dont libplacebo a créé les files, avec leurs capacités ; FFmpeg complète
  // les opérations vidéo de celles qui décodent ou encodent.
  const auto get_families = reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
  if (get_families == nullptr)
    return refuse(RefusalCode::VendorCallFailed, "vkGetPhysicalDeviceQueueFamilyProperties introuvable");
  std::uint32_t count = 0;
  get_families(vk->phys_device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  get_families(vk->phys_device, &count, families.data());
  hw->nb_qf = 0;
  // pl_vulkan.queues est déprécié depuis libplacebo 6.271, mais reste la seule liste des files de
  // libplacebo (graphique, calcul, transfert). Les familles vidéo n'y sont pas : elles suivent.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  constexpr int kMaxFamilies = static_cast<int>(sizeof(hw->qf) / sizeof(hw->qf[0]));
  for (int i = 0; i < vk->num_queues && hw->nb_qf < kMaxFamilies; ++i) {
    const pl_vulkan_queue& queue = vk->queues[i];
    if (queue.index >= count || queue.count == 0) continue;
    hw->qf[hw->nb_qf++] =
        AVVulkanDeviceQueueFamily{.idx = static_cast<int>(queue.index),
                                  .num = static_cast<int>(queue.count),
                                  .flags = static_cast<VkQueueFlagBits>(families[queue.index].queueFlags),
                                  .video_caps = VK_VIDEO_CODEC_OPERATION_NONE_KHR};
  }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
  // Les familles vidéo, créées avec toutes leurs files (kVideoQueueFlags) : sans elles, FFmpeg ne
  // trouve aucune file de décodage (« Decoding of hevc is not supported by this device »).
  for (std::uint32_t family = 0; family < count && hw->nb_qf < kMaxFamilies; ++family) {
    const VkQueueFamilyProperties& props = families[family];
    if ((props.queueFlags & kVideoQueueFlags) == 0 || props.queueCount == 0) continue;
    const auto listed = std::ranges::any_of(
        std::span(hw->qf, static_cast<std::size_t>(hw->nb_qf)),
        [&](const AVVulkanDeviceQueueFamily& qf) { return qf.idx == static_cast<int>(family); });
    if (listed) continue;
    hw->qf[hw->nb_qf++] = AVVulkanDeviceQueueFamily{.idx = static_cast<int>(family),
                                                    .num = static_cast<int>(props.queueCount),
                                                    .flags = static_cast<VkQueueFlagBits>(props.queueFlags),
                                                    .video_caps = VK_VIDEO_CODEC_OPERATION_NONE_KHR};
  }
  if (hw->nb_qf == 0) return refuse(RefusalCode::VendorCallFailed, "aucune file de libplacebo à partager");
  if (const int err = av_hwdevice_ctx_init(ref); err < 0)
    return refuse(RefusalCode::VendorCallFailed, std::format("av_hwdevice_ctx_init : {}", av_text(err)));
  return share;
}

VulkanShare::~VulkanShare() {
  av_buffer_unref(&device_);
}

AVBufferRef* VulkanShare::device(media::HwApi api) const {
  if (api != media::HwApi::Vulkan || device_ == nullptr) return nullptr;
  return av_buffer_ref(device_);
}

}  // namespace ajp::app::play
