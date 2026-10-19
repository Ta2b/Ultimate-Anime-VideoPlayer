// Accès Vulkan à une image décodée : chaque soumission suit l'état noté dans l'AVVkFrame
// (valeur du sémaphore, disposition, file), sous son verrou, comme hwcontext_vulkan.h le
// demande. Une image FFmpeg sur lavapipe, sans décodage : le protocole ne dépend pas du codec.
#include <chrono>
#include <cstdint>
#include <future>
#include <optional>
#include <utility>

#include <catch2/catch_test_macros.hpp>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

#include "media/frame_data.h"
#include "media/hw_device.h"

using namespace ajp::media;
using ajp::FrameMemory;
using ajp::RefusalCode;
using ajp::VulkanImageState;
using namespace std::chrono_literals;

namespace {

// Une image du pool Vulkan de FFmpeg, d'une seule VkImage, ou rien si l'appareil la refuse.
struct VulkanFrame {
  BufferPtr frames;
  FramePtr frame;
};

std::optional<VulkanFrame> vulkan_frame(AVBufferRef& device, AVPixelFormat sw_format) {
  BufferPtr frames(av_hwframe_ctx_alloc(&device));
  if (!frames) return std::nullopt;
  auto* context = reinterpret_cast<AVHWFramesContext*>(frames->data);
  context->format = AV_PIX_FMT_VULKAN;
  context->sw_format = sw_format;
  context->width = 64;
  context->height = 64;
  if (av_hwframe_ctx_init(frames.get()) < 0) return std::nullopt;
  FramePtr frame(av_frame_alloc());
  if (!frame || av_hwframe_get_buffer(frames.get(), frame.get(), 0) < 0) return std::nullopt;
  if (reinterpret_cast<const AVVkFrame*>(frame->data[0])->img[1] != VK_NULL_HANDLE) return std::nullopt;
  return VulkanFrame{std::move(frames), std::move(frame)};
}

// Ce que ferait la soumission de la lecture : signaler la valeur promise, ici depuis l'hôte.
// Sans ce signal, FFmpeg attendrait la valeur pour toujours en libérant son pool.
bool signal_from_host(const AVHWFramesContext& frames, VkSemaphore semaphore, std::uint64_t value) {
  const auto& device = *static_cast<const AVVulkanDeviceContext*>(frames.device_ctx->hwctx);
  const auto device_proc =
      reinterpret_cast<PFN_vkGetDeviceProcAddr>(device.get_proc_addr(device.inst, "vkGetDeviceProcAddr"));
  if (device_proc == nullptr) return false;
  const auto signal =
      reinterpret_cast<PFN_vkSignalSemaphore>(device_proc(device.act_dev, "vkSignalSemaphore"));
  if (signal == nullptr) return false;
  const VkSemaphoreSignalInfo info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = semaphore, .value = value};
  return signal(device.act_dev, &info) == VK_SUCCESS;
}

}  // namespace

TEST_CASE("frame_data : une image hôte n'ouvre aucun accès Vulkan", "[media][frame_data]") {
  FramePtr frame(av_frame_alloc());
  REQUIRE(frame);
  frame->format = AV_PIX_FMT_NV12;
  frame->width = 64;
  frame->height = 64;
  REQUIRE(av_frame_get_buffer(frame.get(), 0) == 0);
  auto decoded = to_decoded_frame(std::move(frame), 0, 40'000);
  REQUIRE(decoded);
  REQUIRE(decoded->data->memory() == FrameMemory::Host);
  auto access = decoded->data->begin_vulkan_access();
  REQUIRE_FALSE(access);
  CHECK(access.error().code == RefusalCode::FormatUnsupported);
  decoded->data->end_vulkan_access({}, true);  // sans accès ouvert : rien à rendre
}

TEST_CASE("frame_data : l'accès Vulkan suit l'état de l'AVVkFrame sous son verrou",
          "[media][frame_data][vulkan]") {
  auto device = open_hw_device(AV_CODEC_ID_HEVC, HwApi::Vulkan, DecoderOptions{});
  if (!device) SKIP("aucun appareil Vulkan : " << device.error().detail);
  std::optional<VulkanFrame> pooled = vulkan_frame(*device->device, AV_PIX_FMT_NV12);
  if (!pooled) pooled = vulkan_frame(*device->device, AV_PIX_FMT_RGBA);
  if (!pooled) SKIP("aucune image Vulkan d'une seule VkImage sur cet appareil");

  auto* frames = reinterpret_cast<AVHWFramesContext*>(pooled->frames->data);
  auto* context = static_cast<AVVulkanFramesContext*>(frames->hwctx);
  AVVkFrame* image = reinterpret_cast<AVVkFrame*>(pooled->frame->data[0]);
  const std::uint64_t before = image->sem_value[0];
  // Ce que le décodeur aurait laissé : une disposition de décodage, notée dans l'image.
  image->layout[0] = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

  auto decoded = to_decoded_frame(std::move(pooled->frame), 0, 40'000);
  REQUIRE(decoded);
  const ajp::FrameData& data = *decoded->data;
  REQUIRE(data.memory() == FrameMemory::Vulkan);

  auto access = data.begin_vulkan_access();
  REQUIRE(access);
  CHECK(access->semaphore == reinterpret_cast<std::uintptr_t>(image->sem[0]));
  CHECK(access->wait_value == before);
  CHECK(access->signal_value == before + 1);
  CHECK(access->state.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  CHECK(access->state.queue_family == image->queue_family[0]);

  // Le verrou de l'image reste pris jusqu'à end_vulkan_access : le décodeur attend.
  auto decoder = std::async(std::launch::async, [&] {
    context->lock_frame(frames, image);
    const std::uint64_t seen = image->sem_value[0];
    context->unlock_frame(frames, image);
    return seen;
  });
  CHECK(decoder.wait_for(50ms) == std::future_status::timeout);

  data.end_vulkan_access({.layout = VK_IMAGE_LAYOUT_GENERAL, .queue_family = access->state.queue_family},
                         true);
  const bool signalled = signal_from_host(*frames, image->sem[0], access->signal_value);
  if (!signalled) image->sem_value[0] = before;  // sinon la libération du pool attendrait sans fin
  REQUIRE(signalled);
  CHECK(decoder.get() == before + 1);  // le décodeur voit la valeur que la lecture signale
  CHECK(image->layout[0] == VK_IMAGE_LAYOUT_GENERAL);

  // Une soumission abandonnée ne change rien : aucune valeur promise jamais signalée.
  auto abandoned = data.begin_vulkan_access();
  REQUIRE(abandoned);
  CHECK(abandoned->wait_value == before + 1);
  CHECK(abandoned->state.layout == VK_IMAGE_LAYOUT_GENERAL);
  data.end_vulkan_access(VulkanImageState{.layout = VK_IMAGE_LAYOUT_UNDEFINED}, false);
  CHECK(image->sem_value[0] == before + 1);
  CHECK(image->layout[0] == VK_IMAGE_LAYOUT_GENERAL);
}
