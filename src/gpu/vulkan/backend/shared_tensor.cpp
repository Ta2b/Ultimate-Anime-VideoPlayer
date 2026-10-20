#include "gpu/vulkan/backend/shared_tensor.h"

#include <algorithm>
#include <format>
#include <optional>
#include <string_view>
#include <utility>

#include "gpu/vulkan/backend/kernel_ops_detail.h"

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace ajp::gpu::vk {

#if defined(_WIN32)

bool exports_memory(const VkContext& /*context*/) {
  return false;
}

bool exports_semaphores(const VkContext& /*context*/) {
  return false;
}

std::expected<int, Refusal> semaphore_fd(const VkContext& /*context*/, VkSemaphore /*semaphore*/) {
  return detail::refuse_backend(RefusalCode::FeatureUnavailable, "interop de sémaphore : Linux seulement");
}

std::expected<std::unique_ptr<TensorBuffer>, Refusal> shared_tensor(
    const VkContext& /*context*/, net::Runtime& /*runtime*/, TensorType /*type*/, const Shape& /*shape*/,
    std::uint64_t /*bytes*/, std::shared_ptr<HostViews> /*views*/) {
  return detail::refuse_backend(RefusalCode::FeatureUnavailable, "interop de mémoire : Linux seulement");
}

#else

namespace {

constexpr VkExternalMemoryHandleTypeFlagBits kHandle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
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
  SharedTensorBuffer(const VkContext& context, VkBuffer buffer, VkDeviceMemory memory, TensorType type,
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
    context_.fn().vkDestroyBuffer(context_.device(), buffer_, nullptr);
    context_.fn().vkFreeMemory(context_.device(), memory_, nullptr);
  }
  const Tensor& tensor() const override { return tensor_; }

  void adopt(std::unique_ptr<net::ImportedMemory> imported, std::uint64_t bytes) {
    imported_ = std::move(imported);
    const std::scoped_lock lock(views_->mutex);
    views_->views[tensor_.handle] = {.data = nullptr, .bytes = bytes, .device = imported_->device()};
  }

 private:
  const VkContext& context_;
  VkBuffer buffer_;
  VkDeviceMemory memory_;
  Tensor tensor_;
  std::shared_ptr<HostViews> views_;
  std::unique_ptr<net::ImportedMemory> imported_;
};

}  // namespace

bool exports_memory(const VkContext& context) {
  return std::ranges::contains(context.caps().extensions,
                               std::string_view(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) &&
         memory_fd_function(context) != nullptr;
}

bool exports_semaphores(const VkContext& context) {
  if (!std::ranges::contains(context.caps().extensions,
                             std::string_view(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME)) ||
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

std::expected<std::unique_ptr<TensorBuffer>, Refusal> shared_tensor(const VkContext& context,
                                                                    net::Runtime& runtime, TensorType type,
                                                                    const Shape& shape, std::uint64_t bytes,
                                                                    std::shared_ptr<HostViews> views) {
  const DeviceFunctions& fn = context.fn();
  const VkExternalMemoryBufferCreateInfo external{
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = kHandle};
  const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                .pNext = &external,
                                .size = bytes,
                                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
  const VkExportMemoryAllocateInfo export_info{
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .pNext = &dedicated, .handleTypes = kHandle};
  const VkMemoryAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                      .pNext = &export_info,
                                      .allocationSize = requirements.size,
                                      .memoryTypeIndex = *type_index};
  VkDeviceMemory memory = VK_NULL_HANDLE;
  if (auto ok = check(fn.vkAllocateMemory(context.device(), &allocate, nullptr, &memory),
                      std::format("vkAllocateMemory ({} o, exportée)", requirements.size));
      !ok) {
    fn.vkDestroyBuffer(context.device(), buffer, nullptr);
    return std::unexpected(ok.error());
  }
  // Le tampon possède désormais tampon et mémoire : chaque refus suivant les rend.
  auto shared = std::make_unique<SharedTensorBuffer>(context, buffer, memory, type, shape, std::move(views));
  if (auto ok = check(fn.vkBindBufferMemory(context.device(), buffer, memory, 0), "vkBindBufferMemory"); !ok)
    return std::unexpected(ok.error());
  const VkMemoryGetFdInfoKHR get{
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = memory, .handleType = kHandle};
  int fd = -1;
  if (auto ok = check(memory_fd_function(context)(context.device(), &get, &fd), "vkGetMemoryFdKHR"); !ok)
    return std::unexpected(ok.error());
  auto imported = runtime.import_memory(fd, requirements.size);
  if (!imported) {
    ::close(fd);  // le runtime ne le prend qu'en cas de succès
    return std::unexpected(imported.error());
  }
  shared->adopt(std::move(*imported), bytes);
  return shared;
}

#endif

}  // namespace ajp::gpu::vk
