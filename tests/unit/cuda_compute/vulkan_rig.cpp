#include "vulkan_rig.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <dlfcn.h>
#include <ranges>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu::cuda::test {

#define AJP_VK_INSTANCE_FUNCTIONS(X)          \
  X(vkDestroyInstance)                        \
  X(vkEnumeratePhysicalDevices)               \
  X(vkGetPhysicalDeviceProperties2)           \
  X(vkGetPhysicalDeviceQueueFamilyProperties) \
  X(vkGetPhysicalDeviceMemoryProperties)      \
  X(vkEnumerateDeviceExtensionProperties)     \
  X(vkCreateDevice)                           \
  X(vkGetDeviceProcAddr)

#define AJP_VK_DEVICE_FUNCTIONS(X) \
  X(vkDestroyDevice)               \
  X(vkGetDeviceQueue)              \
  X(vkCreateCommandPool)           \
  X(vkDestroyCommandPool)          \
  X(vkAllocateCommandBuffers)      \
  X(vkFreeCommandBuffers)          \
  X(vkBeginCommandBuffer)          \
  X(vkEndCommandBuffer)            \
  X(vkCmdPipelineBarrier)          \
  X(vkCmdCopyImageToBuffer)        \
  X(vkCmdFillBuffer)               \
  X(vkQueueSubmit)                 \
  X(vkDeviceWaitIdle)              \
  X(vkCreateFence)                 \
  X(vkDestroyFence)                \
  X(vkWaitForFences)               \
  X(vkResetFences)                 \
  X(vkCreateImage)                 \
  X(vkDestroyImage)                \
  X(vkGetImageMemoryRequirements)  \
  X(vkCreateBuffer)                \
  X(vkDestroyBuffer)               \
  X(vkGetBufferMemoryRequirements) \
  X(vkAllocateMemory)              \
  X(vkFreeMemory)                  \
  X(vkBindImageMemory)             \
  X(vkBindBufferMemory)            \
  X(vkMapMemory)                   \
  X(vkUnmapMemory)                 \
  X(vkGetMemoryFdKHR)              \
  X(vkCreateSemaphore)             \
  X(vkDestroySemaphore)            \
  X(vkGetSemaphoreFdKHR)

#define AJP_VK_MEMBER(name) PFN_##name name = nullptr;
struct VulkanFunctions {
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
  PFN_vkCreateInstance vkCreateInstance = nullptr;
  AJP_VK_INSTANCE_FUNCTIONS(AJP_VK_MEMBER)
  AJP_VK_DEVICE_FUNCTIONS(AJP_VK_MEMBER)
};
#undef AJP_VK_MEMBER

namespace {

constexpr std::array<const char*, 2> kDeviceExtensions = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
                                                          VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};

template <class F>
F load(PFN_vkVoidFunction f) {
  return reinterpret_cast<F>(f);
}

}  // namespace

std::unique_ptr<VulkanRig> VulkanRig::open(const Uuid& uuid, std::string& reason) {
  std::unique_ptr<VulkanRig> rig(new VulkanRig());
  rig->library_ = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (rig->library_ == nullptr) {
    reason = "chargeur Vulkan absent (libvulkan.so.1)";
    return nullptr;
  }
  rig->fn_ = std::make_unique<VulkanFunctions>();
  VulkanFunctions& f = *rig->fn_;
  f.vkGetInstanceProcAddr =
      reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(rig->library_, "vkGetInstanceProcAddr"));
  REQUIRE(f.vkGetInstanceProcAddr != nullptr);
  f.vkCreateInstance =
      load<PFN_vkCreateInstance>(f.vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));

  const VkApplicationInfo app{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                              .pApplicationName = "ajp_cuda_compute_tests",
                              .apiVersion = VK_API_VERSION_1_2};
  const VkInstanceCreateInfo instance_info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                           .pApplicationInfo = &app};
  if (f.vkCreateInstance(&instance_info, nullptr, &rig->instance_) != VK_SUCCESS) {
    reason = "vkCreateInstance refusé";
    return nullptr;
  }
#define AJP_VK_LOAD_INSTANCE(name) f.name = load<PFN_##name>(f.vkGetInstanceProcAddr(rig->instance_, #name));
  AJP_VK_INSTANCE_FUNCTIONS(AJP_VK_LOAD_INSTANCE)
#undef AJP_VK_LOAD_INSTANCE

  std::uint32_t count = 0;
  f.vkEnumeratePhysicalDevices(rig->instance_, &count, nullptr);
  std::vector<VkPhysicalDevice> physicals(count);
  f.vkEnumeratePhysicalDevices(rig->instance_, &count, physicals.data());
  for (VkPhysicalDevice p : physicals) {
    VkPhysicalDeviceIDProperties id{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &id};
    f.vkGetPhysicalDeviceProperties2(p, &props);
    if (std::memcmp(id.deviceUUID, uuid.data(), uuid.size()) == 0 &&
        props.properties.apiVersion >= VK_API_VERSION_1_2) {
      rig->physical_ = p;
    }
  }
  if (rig->physical_ == VK_NULL_HANDLE) {
    reason = "aucun appareil Vulkan 1.2 de même UUID que l'appareil CUDA (lavapipe seul ?)";
    return nullptr;
  }

  f.vkEnumerateDeviceExtensionProperties(rig->physical_, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  f.vkEnumerateDeviceExtensionProperties(rig->physical_, nullptr, &count, extensions.data());
  for (const char* wanted : kDeviceExtensions) {
    if (std::ranges::none_of(extensions, [wanted](const VkExtensionProperties& e) {
          return std::string_view(e.extensionName) == wanted;
        })) {
      reason = std::string("extension Vulkan absente : ") + wanted;
      return nullptr;
    }
  }

  f.vkGetPhysicalDeviceQueueFamilyProperties(rig->physical_, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  f.vkGetPhysicalDeviceQueueFamilyProperties(rig->physical_, &count, families.data());
  const auto family = std::ranges::find_if(families, [](const VkQueueFamilyProperties& q) {
    return (q.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != 0;
  });
  REQUIRE(family != families.end());
  rig->family_ = static_cast<std::uint32_t>(family - families.begin());

  const float priority = 1.0F;
  const VkDeviceQueueCreateInfo queue_info{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                           .queueFamilyIndex = rig->family_,
                                           .queueCount = 1,
                                           .pQueuePriorities = &priority};
  VkPhysicalDeviceVulkan12Features features12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                              .timelineSemaphore = VK_TRUE};
  const VkDeviceCreateInfo device_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = static_cast<std::uint32_t>(kDeviceExtensions.size()),
      .ppEnabledExtensionNames = kDeviceExtensions.data()};
  REQUIRE(f.vkCreateDevice(rig->physical_, &device_info, nullptr, &rig->device_) == VK_SUCCESS);
#define AJP_VK_LOAD_DEVICE(name) f.name = load<PFN_##name>(f.vkGetDeviceProcAddr(rig->device_, #name));
  AJP_VK_DEVICE_FUNCTIONS(AJP_VK_LOAD_DEVICE)
#undef AJP_VK_LOAD_DEVICE
  f.vkGetDeviceQueue(rig->device_, rig->family_, 0, &rig->queue_);
  const VkCommandPoolCreateInfo pool_info{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                          .queueFamilyIndex = rig->family_};
  REQUIRE(f.vkCreateCommandPool(rig->device_, &pool_info, nullptr, &rig->pool_) == VK_SUCCESS);
  const VkFenceCreateInfo fence_info{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  REQUIRE(f.vkCreateFence(rig->device_, &fence_info, nullptr, &rig->fence_) == VK_SUCCESS);
  return rig;
}

VulkanRig::~VulkanRig() {
  if (fn_ && device_ != VK_NULL_HANDLE) {
    fn_->vkDeviceWaitIdle(device_);
    if (pending_ != VK_NULL_HANDLE) fn_->vkFreeCommandBuffers(device_, pool_, 1, &pending_);
    fn_->vkDestroyFence(device_, fence_, nullptr);
    fn_->vkDestroyCommandPool(device_, pool_, nullptr);
    fn_->vkDestroyDevice(device_, nullptr);
  }
  if (fn_ && instance_ != VK_NULL_HANDLE) fn_->vkDestroyInstance(instance_, nullptr);
  if (library_ != nullptr) dlclose(library_);
}

std::uint32_t VulkanRig::memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags) const {
  VkPhysicalDeviceMemoryProperties props{};
  fn_->vkGetPhysicalDeviceMemoryProperties(physical_, &props);
  for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    if ((bits & (1U << i)) != 0 && (props.memoryTypes[i].propertyFlags & flags) == flags) return i;
  }
  FAIL("aucun type de mémoire Vulkan convenable");
  return 0;
}

Exported VulkanRig::export_image(int width, int height) {
  const VulkanFunctions& f = *fn_;
  Exported e;
  const VkExternalMemoryImageCreateInfo external{.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                                                 .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  const VkImageCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &external,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R16G16B16A16_UNORM,
      .extent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  REQUIRE(f.vkCreateImage(device_, &info, nullptr, &e.image) == VK_SUCCESS);
  VkMemoryRequirements req{};
  f.vkGetImageMemoryRequirements(device_, e.image, &req);
  const VkMemoryDedicatedAllocateInfo dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                                .image = e.image};
  const VkExportMemoryAllocateInfo export_info{.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                               .pNext = &dedicated,
                                               .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  const VkMemoryAllocateInfo alloc{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &export_info,
      .allocationSize = req.size,
      .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  REQUIRE(f.vkAllocateMemory(device_, &alloc, nullptr, &e.memory) == VK_SUCCESS);
  REQUIRE(f.vkBindImageMemory(device_, e.image, e.memory, 0) == VK_SUCCESS);
  e.size = req.size;
  const VkMemoryGetFdInfoKHR fd_info{.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                     .memory = e.memory,
                                     .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  REQUIRE(f.vkGetMemoryFdKHR(device_, &fd_info, &e.fd) == VK_SUCCESS);
  return e;
}

Exported VulkanRig::export_buffer(VkDeviceSize bytes) {
  const VulkanFunctions& f = *fn_;
  Exported e;
  const VkExternalMemoryBufferCreateInfo external{
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .pNext = &external,
                                .size = bytes,
                                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  REQUIRE(f.vkCreateBuffer(device_, &info, nullptr, &e.buffer) == VK_SUCCESS);
  VkMemoryRequirements req{};
  f.vkGetBufferMemoryRequirements(device_, e.buffer, &req);
  const VkMemoryDedicatedAllocateInfo dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                                .buffer = e.buffer};
  const VkExportMemoryAllocateInfo export_info{.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                               .pNext = &dedicated,
                                               .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  const VkMemoryAllocateInfo alloc{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &export_info,
      .allocationSize = req.size,
      .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  REQUIRE(f.vkAllocateMemory(device_, &alloc, nullptr, &e.memory) == VK_SUCCESS);
  REQUIRE(f.vkBindBufferMemory(device_, e.buffer, e.memory, 0) == VK_SUCCESS);
  e.size = req.size;
  const VkMemoryGetFdInfoKHR fd_info{.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
                                     .memory = e.memory,
                                     .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  REQUIRE(f.vkGetMemoryFdKHR(device_, &fd_info, &e.fd) == VK_SUCCESS);
  return e;
}

VkSemaphore VulkanRig::export_timeline(int& fd) {
  const VulkanFunctions& f = *fn_;
  const VkExportSemaphoreCreateInfo export_info{
      .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
  const VkSemaphoreTypeCreateInfo type{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                       .pNext = &export_info,
                                       .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                       .initialValue = 0};
  const VkSemaphoreCreateInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type};
  VkSemaphore semaphore = VK_NULL_HANDLE;
  REQUIRE(f.vkCreateSemaphore(device_, &info, nullptr, &semaphore) == VK_SUCCESS);
  const VkSemaphoreGetFdInfoKHR fd_info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                        .semaphore = semaphore,
                                        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
  REQUIRE(f.vkGetSemaphoreFdKHR(device_, &fd_info, &fd) == VK_SUCCESS);
  return semaphore;
}

VkCommandBuffer VulkanRig::begin() {
  const VkCommandBufferAllocateInfo info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = pool_,
                                         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1};
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  REQUIRE(fn_->vkAllocateCommandBuffers(device_, &info, &cmd) == VK_SUCCESS);
  const VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  REQUIRE(fn_->vkBeginCommandBuffer(cmd, &begin_info) == VK_SUCCESS);
  return cmd;
}

void VulkanRig::submit(VkCommandBuffer cmd, VkSemaphore semaphore, std::uint64_t wait, std::uint64_t signal,
                       bool block) {
  const VulkanFunctions& f = *fn_;
  REQUIRE(f.vkEndCommandBuffer(cmd) == VK_SUCCESS);
  const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  const bool waits = semaphore != VK_NULL_HANDLE && wait != 0;
  const bool signals = semaphore != VK_NULL_HANDLE && signal != 0;
  const VkTimelineSemaphoreSubmitInfo timeline{.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                               .waitSemaphoreValueCount = waits ? 1U : 0U,
                                               .pWaitSemaphoreValues = &wait,
                                               .signalSemaphoreValueCount = signals ? 1U : 0U,
                                               .pSignalSemaphoreValues = &signal};
  const VkSubmitInfo info{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .pNext = &timeline,
                          .waitSemaphoreCount = waits ? 1U : 0U,
                          .pWaitSemaphores = &semaphore,
                          .pWaitDstStageMask = &stage,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &cmd,
                          .signalSemaphoreCount = signals ? 1U : 0U,
                          .pSignalSemaphores = &semaphore};
  REQUIRE(f.vkQueueSubmit(queue_, 1, &info, fence_) == VK_SUCCESS);
  pending_ = cmd;
  if (block) finish();
}

void VulkanRig::finish() {
  if (pending_ == VK_NULL_HANDLE) return;
  REQUIRE(fn_->vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX) == VK_SUCCESS);
  REQUIRE(fn_->vkResetFences(device_, 1, &fence_) == VK_SUCCESS);
  fn_->vkFreeCommandBuffers(device_, pool_, 1, &pending_);
  pending_ = VK_NULL_HANDLE;
}

void VulkanRig::release_to_external(VkImage image) {
  VkCommandBuffer cmd = begin();
  const VkImageMemoryBarrier barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                     .srcAccessMask = 0,
                                     .dstAccessMask = 0,
                                     .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                     .srcQueueFamilyIndex = family_,
                                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL,
                                     .image = image,
                                     .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
  fn_->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                            0, nullptr, 0, nullptr, 1, &barrier);
  submit(cmd, VK_NULL_HANDLE, 0, 0, true);
}

std::vector<std::byte> VulkanRig::read_image(VkImage image, int width, int height, VkSemaphore semaphore,
                                             std::uint64_t wait, std::uint64_t signal) {
  const VulkanFunctions& f = *fn_;
  const auto bytes = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 8;
  const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .size = bytes,
                                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VkBuffer staging = VK_NULL_HANDLE;
  REQUIRE(f.vkCreateBuffer(device_, &info, nullptr, &staging) == VK_SUCCESS);
  VkMemoryRequirements req{};
  f.vkGetBufferMemoryRequirements(device_, staging, &req);
  const VkMemoryAllocateInfo alloc{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = memory_type(
          req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  VkDeviceMemory memory = VK_NULL_HANDLE;
  REQUIRE(f.vkAllocateMemory(device_, &alloc, nullptr, &memory) == VK_SUCCESS);
  REQUIRE(f.vkBindBufferMemory(device_, staging, memory, 0) == VK_SUCCESS);

  VkCommandBuffer cmd = begin();
  const VkImageMemoryBarrier acquire{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                     .srcAccessMask = 0,
                                     .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                                     .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                     .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                     .srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL,
                                     .dstQueueFamilyIndex = family_,
                                     .image = image,
                                     .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
  f.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &acquire);
  const VkBufferImageCopy region{
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1}};
  f.vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);
  submit(cmd, semaphore, wait, signal, true);

  void* mapped = nullptr;
  REQUIRE(f.vkMapMemory(device_, memory, 0, bytes, 0, &mapped) == VK_SUCCESS);
  std::vector<std::byte> out(bytes);
  std::memcpy(out.data(), mapped, bytes);
  f.vkUnmapMemory(device_, memory);
  f.vkDestroyBuffer(device_, staging, nullptr);
  f.vkFreeMemory(device_, memory, nullptr);
  return out;
}

void VulkanRig::fill_buffer(VkBuffer buffer, VkDeviceSize bytes, std::uint32_t value, VkSemaphore semaphore,
                            std::uint64_t signal) {
  VkCommandBuffer cmd = begin();
  fn_->vkCmdFillBuffer(cmd, buffer, 0, bytes, value);
  const VkBufferMemoryBarrier release{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                      .dstAccessMask = 0,
                                      .srcQueueFamilyIndex = family_,
                                      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL,
                                      .buffer = buffer,
                                      .offset = 0,
                                      .size = VK_WHOLE_SIZE};
  fn_->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                            nullptr, 1, &release, 0, nullptr);
  submit(cmd, semaphore, 0, signal, false);
}

void VulkanRig::destroy(Exported& e) {
  if (e.image != VK_NULL_HANDLE) fn_->vkDestroyImage(device_, e.image, nullptr);
  if (e.buffer != VK_NULL_HANDLE) fn_->vkDestroyBuffer(device_, e.buffer, nullptr);
  if (e.memory != VK_NULL_HANDLE) fn_->vkFreeMemory(device_, e.memory, nullptr);
  e = {};
}

void VulkanRig::destroy(VkSemaphore semaphore) {
  fn_->vkDestroySemaphore(device_, semaphore, nullptr);
}

}  // namespace ajp::gpu::cuda::test
