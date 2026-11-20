#include "app/play/placebo_present.h"

#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <spdlog/spdlog.h>

namespace kirari::app::play {
namespace {

bool enabled(pl_vulkan vk, const char* extension) {
  for (int i = 0; i < vk->num_extensions; ++i) {
    if (std::string_view(vk->extensions[i]) == extension) return true;
  }
  return false;
}

// La structure de type `type` dans la chaîne des fonctionnalités activées, ou nullptr.
const VkBaseInStructure* find_feature(const VkPhysicalDeviceFeatures2* features, VkStructureType type) {
  for (auto* s = reinterpret_cast<const VkBaseInStructure*>(features); s != nullptr; s = s->pNext) {
    if (s->sType == type) return s;
  }
  return nullptr;
}

}  // namespace

std::vector<const char*> present_extensions(PresentWanted wanted) {
  std::vector<const char*> out;
  if (wanted == PresentWanted::Open || wanted == PresentWanted::Wayland) return out;
#if defined(VK_KHR_present_id2) && defined(VK_KHR_present_wait2) && defined(VK_EXT_present_timing)
  out.push_back(VK_KHR_PRESENT_ID_2_EXTENSION_NAME);
  out.push_back(VK_KHR_PRESENT_WAIT_2_EXTENSION_NAME);
  // Aussi à wait : sans wp_presentation (Windows, X11), present timing est son niveau qui date (B10-01).
  out.push_back(VK_EXT_PRESENT_TIMING_EXTENSION_NAME);
#endif
#ifdef VK_KHR_calibrated_timestamps
  out.push_back(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
#endif
  return out;
}

PresentCaps read_present_caps(pl_vulkan vk, VkSurfaceKHR surface) {
  PresentCaps caps;
#if defined(VK_KHR_present_id2) && defined(VK_KHR_present_wait2) && defined(VK_EXT_present_timing)
  // L'appareil : extension et fonctionnalité activées.
  const auto* id2 = reinterpret_cast<const VkPhysicalDevicePresentId2FeaturesKHR*>(
      find_feature(vk->features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR));
  const auto* wait2 = reinterpret_cast<const VkPhysicalDevicePresentWait2FeaturesKHR*>(
      find_feature(vk->features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR));
  const auto* timing = reinterpret_cast<const VkPhysicalDevicePresentTimingFeaturesEXT*>(
      find_feature(vk->features, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT));
  const bool dev_id2 = enabled(vk, VK_KHR_PRESENT_ID_2_EXTENSION_NAME) && id2 != nullptr && id2->presentId2;
  const bool dev_wait2 =
      enabled(vk, VK_KHR_PRESENT_WAIT_2_EXTENSION_NAME) && wait2 != nullptr && wait2->presentWait2;
  const bool dev_timing =
      enabled(vk, VK_EXT_PRESENT_TIMING_EXTENSION_NAME) && timing != nullptr && timing->presentTiming;
  const bool dev_absolute = dev_timing && timing->presentAtAbsoluteTime;
  // La surface : vkGetPhysicalDeviceSurfaceCapabilities2KHR (VK_KHR_get_surface_capabilities2, que
  // libplacebo active sur l'instance).
  const auto caps2 = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR"));
  if (caps2 != nullptr && surface != VK_NULL_HANDLE) {
    VkPresentTimingSurfaceCapabilitiesEXT surf_timing{
        .sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT};
    VkSurfaceCapabilitiesPresentWait2KHR surf_wait2{
        .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_WAIT_2_KHR,
        .pNext = dev_timing ? &surf_timing : nullptr};
    VkSurfaceCapabilitiesPresentId2KHR surf_id2{
        .sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR, .pNext = &surf_wait2};
    VkSurfaceCapabilities2KHR out{.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR, .pNext = &surf_id2};
    const VkPhysicalDeviceSurfaceInfo2KHR info{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR,
                                               .surface = surface};
    if (caps2(vk->phys_device, &info, &out) == VK_SUCCESS) {
      caps.present_id2 = dev_id2 && surf_id2.presentId2Supported;
      caps.present_wait2 = dev_wait2 && surf_wait2.presentWait2Supported;
      caps.timing = dev_timing && surf_timing.presentTimingSupported;
      caps.timing_absolute = dev_absolute && surf_timing.presentAtAbsoluteTimeSupported;
      caps.present_stages = caps.timing ? surf_timing.presentStageQueries : 0;
    }
  }
#endif
#ifdef VK_KHR_calibrated_timestamps
  caps.calibrated = enabled(vk, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
#endif
  return caps;
}

VkSwapchainCreateFlagsKHR present_flags(PresentLevel level, const PresentCaps& caps) {
  VkSwapchainCreateFlagsKHR flags = 0;
#if defined(VK_KHR_present_id2) && defined(VK_KHR_present_wait2) && defined(VK_EXT_present_timing)
  if (level == PresentLevel::Open || level == PresentLevel::Wayland) return 0;
  if (caps.present_id2) flags |= VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR;
  if (caps.present_wait2) flags |= VK_SWAPCHAIN_CREATE_PRESENT_WAIT_2_BIT_KHR;
  if (level == PresentLevel::Timing || level == PresentLevel::TimingAbsolute)
    flags |= VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT;
#else
  (void)level;
  (void)caps;
#endif
  return flags;
}

// Les espaces colorimétriques que la surface offre : dit, au journal, si un écran HDR peut recevoir du HDR10
// (A2B10G10R10 + HDR10_ST2084) ou si le pilote ne propose que du sRGB.
void log_surface_color_spaces(spdlog::logger& log, pl_vulkan vk, VkSurfaceKHR surface) {
  const auto formats_of = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
      vk->get_proc_addr(vk->instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
  if (formats_of == nullptr || surface == VK_NULL_HANDLE) return;
  std::uint32_t count = 0;
  if (formats_of(vk->phys_device, surface, &count, nullptr) != VK_SUCCESS || count == 0) return;
  std::vector<VkSurfaceFormatKHR> formats(count);
  if (formats_of(vk->phys_device, surface, &count, formats.data()) != VK_SUCCESS) return;
  std::string text;
  bool hdr10 = false;
  for (const VkSurfaceFormatKHR& format : formats) {
    hdr10 = hdr10 || format.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT;
    text += std::format(" [format {} espace {}]", static_cast<int>(format.format),
                        static_cast<int>(format.colorSpace));
  }
  log.info("surface Vulkan : {} configurations, HDR10 (ST2084) {}{}",
           formats.size(),  // i18n-ok : journal
           hdr10 ? "offert" : "absent", text);
}

// Le niveau retenu et ce qui l'a permis, au journal.
void log_present_level(spdlog::logger& log, PresentLevel level, const PresentCaps& c) {
  log.info(
      "présentation : niveau {} (present_id2 {}, present_wait2 {}, present timing {}, "  // i18n-ok : journal
      "date absolue {}, horodatages calibrés {}, wp_presentation {})",
      name(level), c.present_id2, c.present_wait2, c.timing, c.timing_absolute, c.calibrated,
      c.wayland_presentation);
}

}  // namespace kirari::app::play
