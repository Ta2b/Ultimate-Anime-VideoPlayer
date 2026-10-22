#include "gpu/vulkan/backend/shared_tensor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string_view>
#include <utility>

#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/compute/buffer.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// Les types Win32 de Vulkan après windows.h.
#include <vulkan/vulkan_win32.h>
#else
#include <unistd.h>
#endif

namespace ajp::gpu::vk {
namespace {

using net::MemoryHandle;

// L'usage des tampons partagés, le même pour la question d'export et pour leur création.
constexpr VkBufferUsageFlags kSharedUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VkExternalMemoryHandleTypeFlagBits vk_handle(MemoryHandle handle) {
  switch (handle) {
    case MemoryHandle::DmaBuf:
      return VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    case MemoryHandle::OpaqueWin32:
      return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    case MemoryHandle::OpaqueFd:
      break;
  }
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
}

bool enabled(const VkContext& context, std::string_view extension) {
  return std::ranges::contains(context.caps().extensions, extension);
}

#if defined(_WIN32)

PFN_vkGetMemoryWin32HandleKHR memory_win32_function(const VkContext& context) {
  return reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
      context.instance_functions().vkGetDeviceProcAddr(context.device(), "vkGetMemoryWin32HandleKHR"));
}

// Les extensions de ce type activées, et sa fonction d'export : seul le HANDLE NT sous Windows.
bool can_export(const VkContext& context, MemoryHandle handle) {
  return handle == MemoryHandle::OpaqueWin32 &&
         enabled(context, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME) &&
         memory_win32_function(context) != nullptr;
}

// Le descripteur de memory sous ce type, que l'appelant donne au runtime ou ferme.
std::expected<net::HandleValue, Refusal> export_handle(const VkContext& context, VkDeviceMemory memory,
                                                       MemoryHandle handle) {
  const VkMemoryGetWin32HandleInfoKHR get{.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
                                          .memory = memory,
                                          .handleType = vk_handle(handle)};
  HANDLE value = nullptr;
  if (auto ok =
          check(memory_win32_function(context)(context.device(), &get, &value), "vkGetMemoryWin32HandleKHR");
      !ok)
    return std::unexpected(ok.error());
  return reinterpret_cast<net::HandleValue>(value);
}

void close_handle(net::HandleValue value) {
  CloseHandle(reinterpret_cast<HANDLE>(value));
}

#else

constexpr VkExternalSemaphoreHandleTypeFlagBits kSemaphoreHandle =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

PFN_vkGetMemoryFdKHR memory_fd_function(const VkContext& context) {
  return reinterpret_cast<PFN_vkGetMemoryFdKHR>(
      context.instance_functions().vkGetDeviceProcAddr(context.device(), "vkGetMemoryFdKHR"));
}

PFN_vkGetSemaphoreFdKHR semaphore_fd_function(const VkContext& context) {
  return reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
      context.instance_functions().vkGetDeviceProcAddr(context.device(), "vkGetSemaphoreFdKHR"));
}

// Les extensions de ce type activées, et sa fonction d'export : descripteur opaque ou dma-buf
// sous Linux.
bool can_export(const VkContext& context, MemoryHandle handle) {
  if (handle == MemoryHandle::OpaqueWin32 || !enabled(context, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) ||
      memory_fd_function(context) == nullptr)
    return false;
  return handle != MemoryHandle::DmaBuf || enabled(context, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
}

std::expected<net::HandleValue, Refusal> export_handle(const VkContext& context, VkDeviceMemory memory,
                                                       MemoryHandle handle) {
  const VkMemoryGetFdInfoKHR get{
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = memory, .handleType = vk_handle(handle)};
  int fd = -1;
  if (auto ok = check(memory_fd_function(context)(context.device(), &get, &fd), "vkGetMemoryFdKHR"); !ok)
    return std::unexpected(ok.error());
  return fd;
}

void close_handle(net::HandleValue value) {
  ::close(static_cast<int>(value));
}

#endif

// Un type DEVICE_LOCAL permis, invisible de l'hôte de préférence.
std::optional<std::uint32_t> device_type(const VkContext& context, std::uint32_t allowed) {
  VkPhysicalDeviceMemoryProperties properties{};
  context.instance_functions().vkGetPhysicalDeviceMemoryProperties(context.physical_device(), &properties);
  std::optional<std::uint32_t> best;
  for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    const VkMemoryPropertyFlags flags = properties.memoryTypes[i].propertyFlags;
    if ((allowed & (1U << i)) == 0 || (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) continue;
    if (!best || (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) best = i;
    if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) break;
  }
  return best;
}

class SharedTensorBuffer final : public TensorBuffer {
 public:
  SharedTensorBuffer(VkContext& context, VkBuffer buffer, VkDeviceMemory memory, TensorType type,
                     const Shape& shape, std::shared_ptr<HostViews> views)
      : context_(context),
        buffer_(buffer),
        memory_(memory),
        tensor_{.type = type, .shape = shape, .handle = detail::handle_bits(buffer), .offset = 0},
        views_(std::move(views)) {}
  SharedTensorBuffer(const SharedTensorBuffer&) = delete;
  SharedTensorBuffer& operator=(const SharedTensorBuffer&) = delete;
  SharedTensorBuffer(SharedTensorBuffer&&) = delete;
  SharedTensorBuffer& operator=(SharedTensorBuffer&&) = delete;
  ~SharedTensorBuffer() override {
    if (imported_) {
      const std::scoped_lock lock(views_->mutex);
      views_->views.erase(tensor_.handle);
    }
    imported_.reset();  // le runtime rend sa vue avant que la mémoire ne disparaisse
    // Ce que la file a reçu finit avant ; un GPU bloqué garde tampon et mémoire.
    if (!context_.settle()) return;
    context_.fn().vkDestroyBuffer(context_.device(), buffer_, nullptr);
    context_.destroyed().note(handle_word(buffer_));
    context_.fn().vkFreeMemory(context_.device(), memory_, nullptr);
  }
  const Tensor& tensor() const override { return tensor_; }

  void adopt(std::unique_ptr<net::ImportedMemory> imported, std::uint64_t bytes) {
    imported_ = std::move(imported);
    const std::scoped_lock lock(views_->mutex);
    views_->views[tensor_.handle] = {.data = nullptr, .bytes = bytes, .device = imported_->device()};
  }

  // Vulkan et le runtime voient-ils la même mémoire, et pas chacun une copie ? Vulkan écrit un mot
  // à l'octet 0, que le runtime relit en écrivant le sien à l'octet 4 (ImportedMemory::exchange) ;
  // Vulkan relit ce dernier par une copie vers un tampon de l'hôte. Chaque côté attend la fin de
  // l'autre sur l'hôte, comme les étapes de la chaîne. Un runtime qui n'échange pas n'est pas
  // vérifié. Mots remis à zéro ensuite.
  Done verify(MemoryAllocator& memory) {
    constexpr std::uint32_t kFromVulkan = 0x5A3C'A5C3U;
    constexpr std::uint32_t kFromRuntime = 0xC3A5'3C5AU;
    const DeviceFunctions& fn = context_.fn();
    const VkMemoryBarrier to_runtime{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                     .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                     .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    auto written = context_.run_once([&](VkCommandBuffer cb) {
      fn.vkCmdFillBuffer(cb, buffer_, 0, 4, kFromVulkan);
      fn.vkCmdFillBuffer(cb, buffer_, 4, 4, 0);
      fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1,
                              &to_runtime, 0, nullptr, 0, nullptr);
    });
    if (!written) return written;
    auto seen = imported_->exchange(kFromRuntime);
    if (!seen && seen.error().code != RefusalCode::FeatureUnavailable) return std::unexpected(seen.error());
    if (seen && *seen != kFromVulkan) {
      return detail::refuse_backend(
          RefusalCode::FeatureUnavailable,
          std::format("le runtime ne voit pas la mémoire de la chaîne (lu {:#010x})", *seen));
    }
    // Relu par l'hôte s'il y a eu échange, puis remis à zéro, dans la même soumission.
    std::optional<Buffer> readback;
    if (seen) {
      auto made = Buffer::create(context_, memory, 8, MemoryUse::Readback);
      if (!made) return std::unexpected(made.error());
      readback.emplace(std::move(*made));
    }
    const VkMemoryBarrier from_runtime{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    const VkMemoryBarrier copied{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                 .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                                 .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT};
    const VkMemoryBarrier to_host{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                  .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                  .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT};
    auto read = context_.run_once([&](VkCommandBuffer cb) {
      fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                              &from_runtime, 0, nullptr, 0, nullptr);
      if (readback) {
        const VkBufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = 8};
        fn.vkCmdCopyBuffer(cb, buffer_, readback->handle(), 1, &region);
        fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                                &copied, 0, nullptr, 0, nullptr);
      }
      fn.vkCmdFillBuffer(cb, buffer_, 0, 8, 0);
      fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &to_host,
                              0, nullptr, 0, nullptr);
    });
    if (!read) return read;
    if (!readback) return {};
    std::uint32_t back = 0;
    std::memcpy(&back, readback->mapped() + 4, 4);
    if (back != kFromRuntime) {
      return detail::refuse_backend(
          RefusalCode::FeatureUnavailable,
          std::format("la chaîne ne voit pas ce que le runtime écrit (lu {:#010x})", back));
    }
    return {};
  }

 private:
  VkContext& context_;
  VkBuffer buffer_;
  VkDeviceMemory memory_;
  Tensor tensor_;
  std::shared_ptr<HostViews> views_;
  std::unique_ptr<net::ImportedMemory> imported_;
};

}  // namespace

bool exports_memory(const VkContext& context, MemoryHandle handle) {
  if (!can_export(context, handle)) return false;
  const VkPhysicalDeviceExternalBufferInfo info{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
      .usage = kSharedUsage,
      .handleType = vk_handle(handle)};
  VkExternalBufferProperties properties{.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
  context.instance_functions().vkGetPhysicalDeviceExternalBufferProperties(context.physical_device(), &info,
                                                                           &properties);
  return (properties.externalMemoryProperties.externalMemoryFeatures &
          VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
}

#if defined(_WIN32)

bool exports_semaphores(const VkContext& /*context*/) {
  return false;
}

std::expected<int, Refusal> semaphore_fd(const VkContext& /*context*/, VkSemaphore /*semaphore*/) {
  return detail::refuse_backend(RefusalCode::FeatureUnavailable, "interop de sémaphore : Linux seulement");
}

#else

bool exports_semaphores(const VkContext& context) {
  if (!enabled(context, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) ||
      semaphore_fd_function(context) == nullptr)
    return false;
  const VkSemaphoreTypeCreateInfo timeline{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                           .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
  const VkPhysicalDeviceExternalSemaphoreInfo info{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
      .pNext = &timeline,
      .handleType = kSemaphoreHandle};
  VkExternalSemaphoreProperties properties{.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
  context.instance_functions().vkGetPhysicalDeviceExternalSemaphoreProperties(context.physical_device(),
                                                                              &info, &properties);
  return (properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) != 0;
}

std::expected<int, Refusal> semaphore_fd(const VkContext& context, VkSemaphore semaphore) {
  const VkSemaphoreGetFdInfoKHR get{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                                    .semaphore = semaphore,
                                    .handleType = kSemaphoreHandle};
  int fd = -1;
  if (auto ok = check(semaphore_fd_function(context)(context.device(), &get, &fd), "vkGetSemaphoreFdKHR");
      !ok)
    return std::unexpected(ok.error());
  return fd;
}

#endif

std::expected<std::unique_ptr<TensorBuffer>, Refusal> shared_tensor(
    VkContext& context, MemoryAllocator& memory, net::Runtime& runtime, TensorType type, const Shape& shape,
    std::uint64_t bytes, std::shared_ptr<HostViews> views, bool verify) {
  const MemoryHandle handle = runtime.memory_handle();
  if (!exports_memory(context, handle))
    return detail::refuse_backend(RefusalCode::FeatureUnavailable,
                                  "interop : descripteur non exporté pour ce tampon");
  const DeviceFunctions& fn = context.fn();
  const VkExternalMemoryBufferCreateInfo external{
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = vk_handle(handle)};
  const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .pNext = &external,
                                .size = bytes,
                                .usage = kSharedUsage,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VkBuffer buffer = VK_NULL_HANDLE;
  if (auto ok =
          check(fn.vkCreateBuffer(context.device(), &info, nullptr, &buffer), "vkCreateBuffer (exporté)");
      !ok)
    return std::unexpected(ok.error());
  VkMemoryRequirements requirements{};
  fn.vkGetBufferMemoryRequirements(context.device(), buffer, &requirements);
  const auto type_index = device_type(context, requirements.memoryTypeBits);
  if (!type_index) {
    fn.vkDestroyBuffer(context.device(), buffer, nullptr);
    return detail::refuse_backend(RefusalCode::FeatureUnavailable,
                                  "interop : aucun type de mémoire d'appareil");
  }
  const VkMemoryDedicatedAllocateInfo dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                                .buffer = buffer};
  const VkExportMemoryAllocateInfo export_info{.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                               .pNext = &dedicated,
                                               .handleTypes = vk_handle(handle)};
  const VkMemoryAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                      .pNext = &export_info,
                                      .allocationSize = requirements.size,
                                      .memoryTypeIndex = *type_index};
  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  if (auto ok = check(fn.vkAllocateMemory(context.device(), &allocate, nullptr, &device_memory),
                      std::format("vkAllocateMemory ({} o, exportée)", requirements.size));
      !ok) {
    fn.vkDestroyBuffer(context.device(), buffer, nullptr);
    return std::unexpected(ok.error());
  }
  // Le tampon possède désormais tampon et mémoire : chaque refus suivant les rend.
  auto shared =
      std::make_unique<SharedTensorBuffer>(context, buffer, device_memory, type, shape, std::move(views));
  if (auto ok =
          check(fn.vkBindBufferMemory(context.device(), buffer, device_memory, 0), "vkBindBufferMemory");
      !ok)
    return std::unexpected(ok.error());
  auto exported = export_handle(context, device_memory, handle);
  if (!exported) return std::unexpected(exported.error());
  auto imported = runtime.import_memory(*exported, requirements.size, type, shape);
  if (!imported) {
    close_handle(*exported);  // le runtime ne le prend qu'en cas de succès
    return std::unexpected(imported.error());
  }
  shared->adopt(std::move(*imported), bytes);
  if (verify) {
    if (auto ok = shared->verify(memory); !ok) return std::unexpected(ok.error());
  }
  return shared;
}

}  // namespace ajp::gpu::vk
