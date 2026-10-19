#include "vk_import.h"

#include <algorithm>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace ajp::render::test {
namespace {

template <class Pfn>
Pfn load(PFN_vkGetDeviceProcAddr gdpa, VkDevice device, const char* name) {
  return reinterpret_cast<Pfn>(gdpa(device, name));
}

template <class Handle>
Handle from_u64(std::uint64_t value) {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(value));
  } else {
    return static_cast<Handle>(value);
  }
}

}  // namespace

VulkanCalls::VulkanCalls(const GpuDevice& gpu_device) {
  const VulkanDevice& vk = *gpu_device.vulkan;
  instance = static_cast<VkInstance>(vk.instance);
  physical = static_cast<VkPhysicalDevice>(vk.physical_device);
  device = static_cast<VkDevice>(vk.device);
  const auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(vk.get_instance_proc_addr);
  const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
  memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
      gipa(instance, "vkGetPhysicalDeviceMemoryProperties"));
  create_image = load<PFN_vkCreateImage>(gdpa, device, "vkCreateImage");
  destroy_image = load<PFN_vkDestroyImage>(gdpa, device, "vkDestroyImage");
  image_requirements = load<PFN_vkGetImageMemoryRequirements>(gdpa, device, "vkGetImageMemoryRequirements");
  allocate = load<PFN_vkAllocateMemory>(gdpa, device, "vkAllocateMemory");
  free = load<PFN_vkFreeMemory>(gdpa, device, "vkFreeMemory");
  bind = load<PFN_vkBindImageMemory>(gdpa, device, "vkBindImageMemory");
  create_semaphore = load<PFN_vkCreateSemaphore>(gdpa, device, "vkCreateSemaphore");
  destroy_semaphore = load<PFN_vkDestroySemaphore>(gdpa, device, "vkDestroySemaphore");
  import_semaphore = load<PFN_vkImportSemaphoreFdKHR>(gdpa, device, "vkImportSemaphoreFdKHR");
  counter_value = load<PFN_vkGetSemaphoreCounterValue>(gdpa, device, "vkGetSemaphoreCounterValue");
  signal_semaphore = load<PFN_vkSignalSemaphore>(gdpa, device, "vkSignalSemaphore");
}

std::uint64_t VulkanCalls::counter(std::uint64_t semaphore) const {
  std::uint64_t value = 0;
  counter_value(device, from_u64<VkSemaphore>(semaphore), &value);
  return value;
}

void VulkanCalls::signal(std::uint64_t semaphore, std::uint64_t value) const {
  const VkSemaphoreSignalInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                                   .semaphore = from_u64<VkSemaphore>(semaphore),
                                   .value = value};
  signal_semaphore(device, &info);
}

ImportedSlot::~ImportedSlot() {
  pl_gpu_finish(gpu_);
  pl_tex_destroy(gpu_, &tex);
  if (image != VK_NULL_HANDLE) vk_.destroy_image(vk_.device, image, nullptr);
  if (memory != VK_NULL_HANDLE) vk_.free(vk_.device, memory, nullptr);
  if (semaphore != VK_NULL_HANDLE) vk_.destroy_semaphore(vk_.device, semaphore, nullptr);
}

void ImportedSlot::write(pl_gpu gpu, std::uint64_t wait_value, const std::array<float, 4>& color,
                         std::uint64_t signal_value) {
  const pl_vulkan_release_params release{
      .tex = tex,
      .layout = first_ ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
      .qf = VK_QUEUE_FAMILY_IGNORED,
      .semaphore = {.sem = semaphore, .value = wait_value}};
  pl_vulkan_release_ex(gpu, &release);
  pl_tex_clear(gpu, tex, color.data());
  const pl_vulkan_hold_params hold{.tex = tex,
                                   .layout = VK_IMAGE_LAYOUT_GENERAL,
                                   .qf = VK_QUEUE_FAMILY_IGNORED,
                                   .semaphore = {.sem = semaphore, .value = signal_value}};
  pl_vulkan_hold_ex(gpu, &hold);
  pl_gpu_flush(gpu);
  first_ = false;
}

std::expected<std::unique_ptr<ImportedSlot>, std::string> import_slot(const VulkanCalls& vk,
                                                                      const RenderDevice& device,
                                                                      const OutputRing& ring,
                                                                      std::uint32_t index) {
  const RingSlotDesc& desc = ring.slots()[index];
  auto slot = std::make_unique<ImportedSlot>(vk, device.gpu());

  // Mêmes paramètres que libplacebo (gpu_tex.c) : partage entre ses familles de files.
  const pl_vulkan pv = device.vulkan();
  std::vector<std::uint32_t> families;
  for (const std::uint32_t family :
       {pv->queue_graphics.index, pv->queue_compute.index, pv->queue_transfer.index}) {
    if (std::find(families.begin(), families.end(), family) == families.end()) families.push_back(family);
  }
  const VkExternalMemoryImageCreateInfo external{.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                                                 .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
  const VkImageCreateInfo info{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &external,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R16G16B16A16_UNORM,
      .extent = {.width = static_cast<std::uint32_t>(desc.width),
                 .height = static_cast<std::uint32_t>(desc.height),
                 .depth = 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = ring.image_usage(),
      .sharingMode = families.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = static_cast<std::uint32_t>(families.size()),
      .pQueueFamilyIndices = families.data(),
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  if (vk.create_image(vk.device, &info, nullptr, &slot->image) != VK_SUCCESS)
    return std::unexpected("vkCreateImage");

  VkMemoryRequirements reqs{};
  vk.image_requirements(vk.device, slot->image, &reqs);
  VkPhysicalDeviceMemoryProperties props{};
  vk.memory_properties(vk.physical, &props);
  std::uint32_t type = 0;
  while (type < props.memoryTypeCount && ((reqs.memoryTypeBits >> type) & 1U) == 0) ++type;
  if (type == props.memoryTypeCount) return std::unexpected("aucun type de mémoire");

  const VkMemoryDedicatedAllocateInfo dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                                .image = slot->image};
  const VkImportMemoryFdInfoKHR import{.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
                                       .pNext = ring.dedicated(index) ? &dedicated : nullptr,
                                       .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
                                       .fd = dup(static_cast<int>(desc.memory.value))};
  const VkMemoryAllocateInfo alloc{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                   .pNext = &import,
                                   .allocationSize = desc.memory_size,
                                   .memoryTypeIndex = type};
  if (vk.allocate(vk.device, &alloc, nullptr, &slot->memory) != VK_SUCCESS) {
    close(import.fd);
    return std::unexpected("vkAllocateMemory (import)");
  }
  if (vk.bind(vk.device, slot->image, slot->memory, desc.memory_offset) != VK_SUCCESS) {
    return std::unexpected("vkBindImageMemory");
  }

  const VkSemaphoreTypeCreateInfo timeline{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                           .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
  const VkSemaphoreCreateInfo sem_info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timeline};
  if (vk.create_semaphore(vk.device, &sem_info, nullptr, &slot->semaphore) != VK_SUCCESS) {
    return std::unexpected("vkCreateSemaphore");
  }
  const VkImportSemaphoreFdInfoKHR sem_import{.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
                                              .semaphore = slot->semaphore,
                                              .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
                                              .fd = dup(static_cast<int>(desc.semaphore_handle.value))};
  if (vk.import_semaphore(vk.device, &sem_import) != VK_SUCCESS) {
    close(sem_import.fd);
    return std::unexpected("vkImportSemaphoreFdKHR");
  }

  // L'image importée n'a encore aucune disposition : le premier tour part d'UNDEFINED.
  pl_vulkan_wrap_params wrap{};
  wrap.image = slot->image;
  wrap.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
  wrap.width = desc.width;
  wrap.height = desc.height;
  wrap.format = VK_FORMAT_R16G16B16A16_UNORM;
  wrap.usage = ring.image_usage();
  wrap.debug_tag = "test écrivain";
  slot->tex = pl_vulkan_wrap(device.gpu(), &wrap);
  if (slot->tex == nullptr) return std::unexpected("pl_vulkan_wrap");
  return slot;
}

}  // namespace ajp::render::test
