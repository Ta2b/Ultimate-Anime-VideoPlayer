#include "render/device_description.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "render/c_structs.h"
#include "render/vk_util.h"

namespace ajp::render::detail {
namespace {

bool has_extension(const std::vector<std::string>& extensions, std::string_view name) {
  return std::ranges::find(extensions, name) != extensions.end();
}

// pl_vulkan_t::extensions peut contenir des doublons : on garde le premier de chaque nom.
std::vector<std::string> unique_names(const char* const* names, int count) {
  std::vector<std::string> out;
  for (int i = 0; i < count; ++i) {
    const std::string_view name = names[i];
    if (!has_extension(out, name)) out.emplace_back(name);
  }
  return out;
}

// driverInfo dit la version lisible (« 615.78.08 », « Mesa 26.2.4 ») ; sans lui, on décode
// driverVersion à la manière de NVIDIA (10.8.8.6 bits) ou de VK_MAKE_API_VERSION.
std::string driver_version(const VkPhysicalDeviceProperties& props,
                           const VkPhysicalDeviceDriverProperties& driver) {
  const std::string_view info(driver.driverInfo, strnlen(driver.driverInfo, VK_MAX_DRIVER_INFO_SIZE));
  if (!info.empty()) return std::string(info);
  const std::uint32_t v = props.driverVersion;
  if (props.vendorID == pci::kNvidia) {
    return std::format("{}.{}.{}", v >> 22U, (v >> 14U) & 0xFFU, (v >> 6U) & 0xFFU);
  }
  return std::format("{}.{}.{}", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
}

struct FamilyInfo {
  VkQueueFlags flags = 0;
  std::uint32_t count = 0;
  VkVideoCodecOperationFlagsKHR video_caps = 0;
};

std::vector<FamilyInfo> query_families(PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
                                       VkPhysicalDevice physical, bool video_queue) {
  const auto get_families = instance_function<PFN_vkGetPhysicalDeviceQueueFamilyProperties2>(
      gipa, instance, "vkGetPhysicalDeviceQueueFamilyProperties2");
  std::uint32_t count = 0;
  get_families(physical, &count, nullptr);
  std::vector<VkQueueFamilyVideoPropertiesKHR> video(
      count, {.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR});
  std::vector<VkQueueFamilyProperties2> props(count, {.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2});
  // La structure vidéo n'est chaînée que si VK_KHR_video_queue est activée.
  if (video_queue) {
    for (std::size_t i = 0; i < props.size(); ++i) props[i].pNext = &video[i];
  }
  get_families(physical, &count, props.data());

  std::vector<FamilyInfo> out(count);
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i].flags = props[i].queueFamilyProperties.queueFlags;
    out[i].count = props[i].queueFamilyProperties.queueCount;
    out[i].video_caps = video_queue ? video[i].videoCodecOperations : 0;
  }
  return out;
}

}  // namespace

std::expected<GpuDevice, Refusal> describe_device(pl_vk_inst instance, pl_vulkan vk,
                                                  const QueueLocks& locks) {
  const PFN_vkGetInstanceProcAddr gipa = vk->get_proc_addr;
  const auto get_props = instance_function<PFN_vkGetPhysicalDeviceProperties2>(
      gipa, vk->instance, "vkGetPhysicalDeviceProperties2");
  if (get_props == nullptr) {
    return std::unexpected(
        Refusal{RefusalCode::FeatureUnavailable, "vkGetPhysicalDeviceProperties2 absente"});
  }
  auto driver = zeroed<VkPhysicalDeviceDriverProperties>();
  driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
  VkPhysicalDeviceIDProperties ids{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
                                   .pNext = &driver};
  VkPhysicalDeviceProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &ids};
  get_props(vk->phys_device, &props);

  GpuDevice out;
  out.api = GpuApi::Vulkan;
  out.pci_vendor = props.properties.vendorID;
  out.pci_device = props.properties.deviceID;
  out.name = std::string(props.properties.deviceName,
                         strnlen(props.properties.deviceName, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE));
  std::ranges::copy(ids.deviceUUID, out.uuid.begin());
#ifdef _WIN32
  if (ids.deviceLUIDValid == VK_TRUE) {
    DeviceLuid luid{};
    std::ranges::copy(ids.deviceLUID, luid.begin());
    out.luid = luid;
  }
#endif
  out.driver_version = driver_version(props.properties, driver);

  VulkanDevice dev;
  dev.instance = vk->instance;
  dev.physical_device = vk->phys_device;
  dev.device = vk->device;
  dev.get_instance_proc_addr = reinterpret_cast<VoidFunction>(gipa);
  dev.api_version = instance->api_version;
  dev.instance_extensions = unique_names(instance->extensions, instance->num_extensions);
  dev.device_extensions = unique_names(vk->extensions, vk->num_extensions);
  dev.enabled_features = vk->features;
  dev.graphics = QueueRef{.family = vk->queue_graphics.index, .index = 0};
  dev.compute = QueueRef{.family = vk->queue_compute.index, .index = 0};
  dev.lock_queue = locks.lock;
  dev.unlock_queue = locks.unlock;
  dev.queue_context = locks.context;

  const bool video_queue = has_extension(dev.device_extensions, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
  const bool video_decode =
      video_queue && has_extension(dev.device_extensions, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
  const std::vector<FamilyInfo> families = query_families(gipa, vk->instance, vk->phys_device, video_queue);
  // Même règle que libplacebo (device_init) : toutes les files des familles graphique, calcul
  // et transfert, et de celles qui portent une file demandée en plus (le décodage vidéo).
  for (std::uint32_t i = 0; i < families.size(); ++i) {
    const FamilyInfo& family = families[i];
    const bool used = i == vk->queue_graphics.index || i == vk->queue_compute.index ||
                      i == vk->queue_transfer.index || (family.flags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) != 0;
    if (!used) continue;
    dev.families.push_back(QueueFamily{
        .index = i, .count = family.count, .flags = family.flags, .video_caps = family.video_caps});
    if (video_decode && !dev.video_decode && (family.flags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) != 0) {
      dev.video_decode = QueueRef{.family = i, .index = 0};
    }
  }
  out.vulkan = std::move(dev);
  return out;
}

}  // namespace ajp::render::detail
