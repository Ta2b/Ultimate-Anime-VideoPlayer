// Ce que placebo_open.cpp demande à Vulkan pour le retour de présentation (chantier 6.2) et ce qu'il en
// relit : les extensions et fonctionnalités de present_id2, present_wait2, present_timing et des horodatages
// calibrés, les capacités de la surface, les drapeaux de la swapchain. Le choix lui-même est pur
// (present_level.h). Rien n'est demandé à --present=open : l'appareil est créé comme avant.
#pragma once

#include <vector>

#include <libplacebo/vulkan.h>

#include "app/play/present_level.h"

namespace ajp::app::play {

// Les fonctionnalités de présentation à chaîner devant `next` (VkPhysicalDeviceFeatures2::pNext) ; libplacebo
// n'active que celles que l'appareil a. Vide (head() rend next) aux en-têtes Vulkan qui ne les connaissent
// pas.
struct PresentFeatures {
#if defined(VK_KHR_present_id2) && defined(VK_KHR_present_wait2) && defined(VK_EXT_present_timing)
  VkPhysicalDevicePresentTimingFeaturesEXT timing{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT,
      .presentTiming = VK_TRUE,
      .presentAtAbsoluteTime = VK_TRUE};
  VkPhysicalDevicePresentWait2FeaturesKHR wait2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR, .presentWait2 = VK_TRUE};
  VkPhysicalDevicePresentId2FeaturesKHR id2{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR, .presentId2 = VK_TRUE};
  void* head(void* next) {
    id2.pNext = next;
    wait2.pNext = &id2;
    timing.pNext = &wait2;
    return &timing;
  }
#else
  void* head(void* next) { return next; }
#endif
};

// Les extensions de présentation demandées en plus (aucune à --present=open).
std::vector<const char*> present_extensions(PresentWanted wanted);

// Ce que l'appareil a activé et ce que la surface offre.
PresentCaps read_present_caps(pl_vulkan vk, VkSurfaceKHR surface);

// Les drapeaux de création de la swapchain au niveau retenu (0 à Open et Wayland).
VkSwapchainCreateFlagsKHR present_flags(PresentLevel level, const PresentCaps& caps);

}  // namespace ajp::app::play
