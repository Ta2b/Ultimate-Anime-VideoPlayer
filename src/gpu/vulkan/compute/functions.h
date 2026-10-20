// Tables des fonctions Vulkan du socle, remplies depuis un vkGetInstanceProcAddr : globales
// (sans instance), d'instance, puis d'appareil par vkGetDeviceProcAddr. Une liste par niveau
// déclare chaque fonction une seule fois ; une fonction absente est refusée au chargement.
#pragma once

#include <expected>

#include "core/types.h"
#include "gpu/vulkan/compute/vulkan.h"

// clang-format off
#define AJP_VK_GLOBAL_FUNCTIONS(X)        \
  X(vkCreateInstance)                     \
  X(vkEnumerateInstanceVersion)           \
  X(vkEnumerateInstanceExtensionProperties) \
  X(vkEnumerateInstanceLayerProperties)

#define AJP_VK_INSTANCE_FUNCTIONS(X)            \
  X(vkDestroyInstance)                          \
  X(vkEnumeratePhysicalDevices)                 \
  X(vkGetPhysicalDeviceProperties)              \
  X(vkGetPhysicalDeviceProperties2)             \
  X(vkGetPhysicalDeviceFeatures2)               \
  X(vkGetPhysicalDeviceQueueFamilyProperties)   \
  X(vkGetPhysicalDeviceMemoryProperties)        \
  X(vkGetPhysicalDeviceFormatProperties)        \
  X(vkGetPhysicalDeviceExternalSemaphoreProperties) \
  X(vkEnumerateDeviceExtensionProperties)       \
  X(vkCreateDevice)                             \
  X(vkGetDeviceProcAddr)

#define AJP_VK_DEVICE_FUNCTIONS(X)        \
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
  X(vkCreateSemaphore)                    \
  X(vkDestroySemaphore)                   \
  X(vkWaitSemaphores)                     \
  X(vkSignalSemaphore)                    \
  X(vkGetSemaphoreCounterValue)
// clang-format on

#define AJP_VK_DECLARE(name) PFN_##name name = nullptr;

namespace ajp::gpu::vk {

struct GlobalFunctions {
  AJP_VK_GLOBAL_FUNCTIONS(AJP_VK_DECLARE)
};

struct InstanceFunctions {
  AJP_VK_INSTANCE_FUNCTIONS(AJP_VK_DECLARE)
};

struct DeviceFunctions {
  AJP_VK_DEVICE_FUNCTIONS(AJP_VK_DECLARE)
};

// Refus LibraryUnavailable qui nomme la première fonction absente.
std::expected<GlobalFunctions, Refusal> load_global(PFN_vkGetInstanceProcAddr get_instance_proc_addr);
std::expected<InstanceFunctions, Refusal> load_instance(PFN_vkGetInstanceProcAddr get_instance_proc_addr,
                                                        VkInstance instance);
std::expected<DeviceFunctions, Refusal> load_device(const InstanceFunctions& instance, VkDevice device);

}  // namespace ajp::gpu::vk

#undef AJP_VK_DECLARE
