// Tables des fonctions Vulkan du socle, remplies depuis un vkGetInstanceProcAddr : globales
// (sans instance), d'instance, puis d'appareil par vkGetDeviceProcAddr. Une liste par niveau
// déclare chaque fonction une seule fois ; une fonction absente est refusée au chargement.
#pragma once

#include <expected>

#include "core/types.h"
#include "gpu/vulkan/compute/vulkan.h"

// clang-format off
#define KIRARI_VK_GLOBAL_FUNCTIONS(X)        \
  X(vkCreateInstance)                     \
  X(vkEnumerateInstanceVersion)           \
  X(vkEnumerateInstanceExtensionProperties) \
  X(vkEnumerateInstanceLayerProperties)

#define KIRARI_VK_INSTANCE_FUNCTIONS(X)            \
  X(vkDestroyInstance)                          \
  X(vkEnumeratePhysicalDevices)                 \
  X(vkGetPhysicalDeviceProperties)              \
  X(vkGetPhysicalDeviceProperties2)             \
  X(vkGetPhysicalDeviceFeatures2)               \
  X(vkGetPhysicalDeviceQueueFamilyProperties)   \
  X(vkGetPhysicalDeviceMemoryProperties)        \
  X(vkGetPhysicalDeviceMemoryProperties2)       \
  X(vkGetPhysicalDeviceFormatProperties)        \
  X(vkGetPhysicalDeviceExternalSemaphoreProperties) \
  X(vkGetPhysicalDeviceExternalBufferProperties) \
  X(vkEnumerateDeviceExtensionProperties)       \
  X(vkCreateDevice)                             \
  X(vkGetDeviceProcAddr)

#define KIRARI_VK_DEVICE_FUNCTIONS(X)        \
  X(vkDestroyDevice)                      \
  X(vkGetDeviceQueue)                     \
  X(vkDeviceWaitIdle)                     \
  X(vkQueueSubmit)                        \
  X(vkAllocateMemory)                     \
  X(vkFreeMemory)                         \
  X(vkMapMemory)                          \
  X(vkUnmapMemory)                        \
  X(vkCreateBuffer)                       \
  X(vkDestroyBuffer)                      \
  X(vkGetBufferMemoryRequirements)        \
  X(vkBindBufferMemory)                   \
  X(vkCreateImage)                        \
  X(vkDestroyImage)                       \
  X(vkGetImageMemoryRequirements)         \
  X(vkBindImageMemory)                    \
  X(vkCreateImageView)                    \
  X(vkDestroyImageView)                   \
  X(vkCreateShaderModule)                 \
  X(vkDestroyShaderModule)                \
  X(vkCreatePipelineCache)                \
  X(vkGetPipelineCacheData)               \
  X(vkDestroyPipelineCache)               \
  X(vkCreateComputePipelines)             \
  X(vkDestroyPipeline)                    \
  X(vkCreatePipelineLayout)               \
  X(vkDestroyPipelineLayout)              \
  X(vkCreateDescriptorSetLayout)          \
  X(vkDestroyDescriptorSetLayout)         \
  X(vkCreateDescriptorPool)               \
  X(vkDestroyDescriptorPool)              \
  X(vkAllocateDescriptorSets)             \
  X(vkUpdateDescriptorSets)               \
  X(vkCreateCommandPool)                  \
  X(vkDestroyCommandPool)                 \
  X(vkAllocateCommandBuffers)             \
  X(vkFreeCommandBuffers)                 \
  X(vkBeginCommandBuffer)                 \
  X(vkEndCommandBuffer)                   \
  X(vkCmdBindPipeline)                    \
  X(vkCmdBindDescriptorSets)              \
  X(vkCmdPushConstants)                   \
  X(vkCmdDispatch)                        \
  X(vkCmdPipelineBarrier)                 \
  X(vkCmdFillBuffer)                      \
  X(vkCmdCopyBuffer)                      \
  X(vkCmdCopyImageToBuffer)               \
  X(vkCmdCopyBufferToImage)               \
  X(vkCreateSampler)                      \
  X(vkDestroySampler)                     \
  X(vkCreateSemaphore)                    \
  X(vkDestroySemaphore)                   \
  X(vkWaitSemaphores)                     \
  X(vkSignalSemaphore)                    \
  X(vkGetSemaphoreCounterValue)
// clang-format on

#define KIRARI_VK_DECLARE(name) PFN_##name name = nullptr;

namespace kirari::gpu::vk {

struct GlobalFunctions {
  KIRARI_VK_GLOBAL_FUNCTIONS(KIRARI_VK_DECLARE)
};

struct InstanceFunctions {
  KIRARI_VK_INSTANCE_FUNCTIONS(KIRARI_VK_DECLARE)
};

struct DeviceFunctions {
  KIRARI_VK_DEVICE_FUNCTIONS(KIRARI_VK_DECLARE)
};

// Refus LibraryUnavailable qui nomme la première fonction absente.
std::expected<GlobalFunctions, Refusal> load_global(PFN_vkGetInstanceProcAddr get_instance_proc_addr);
std::expected<InstanceFunctions, Refusal> load_instance(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                                                        VkInstance instance);
std::expected<DeviceFunctions, Refusal> load_device(const InstanceFunctions& instance, VkDevice device);

}  // namespace kirari::gpu::vk

#undef KIRARI_VK_DECLARE
