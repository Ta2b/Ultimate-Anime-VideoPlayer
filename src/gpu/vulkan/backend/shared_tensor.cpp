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
namespace {

// Tampon de stockage dans une mémoire hôte du runtime : Vulkan rend son tampon et sa mémoire
// importée avant que le runtime ne libère la sienne.
class HostImportedBuffer final : public TensorBuffer {
 public:
  HostImportedBuffer(VkContext& context, std::unique_ptr<net::HostMemory> host, TensorType type,
                     const Shape& shape, std::uint64_t bytes, std::shared_ptr<HostViews> views)
      : context_(context),
        host_(std::move(host)),
        bytes_(bytes),
        type_(type),
        shape_(shape),
        views_(std::move(views)) {}
  HostImportedBuffer(const HostImportedBuffer&) = delete;
  HostImportedBuffer& operator=(const HostImportedBuffer&) = delete;
  HostImportedBuffer(HostImportedBuffer&&) = delete;
  HostImportedBuffer& operator=(HostImportedBuffer&&) = delete;
  ~HostImportedBuffer() override {
    if (tensor_.handle != 0) {
      const std::scoped_lock lock(views_->mutex);
      views_->views.erase(tensor_.handle);
    }
    // Ce que la file a reçu finit avant ; un GPU bloqué garde tampon, mémoire importée et mémoire
    // hôte.
    if (!context_.settle()) {
      [[maybe_unused]] const net::HostMemory* kept = host_.release();
      return;
    }
    if (buffer_ != VK_NULL_HANDLE) {
      context_.fn().vkDestroyBuffer(context_.device(), buffer_, nullptr);
      context_.destroyed().note(handle_word(buffer_));
    }
    if (memory_ != VK_NULL_HANDLE) context_.fn().vkFreeMemory(context_.device(), memory_, nullptr);
    host_.reset();
  }
  const Tensor& tensor() const override { return tensor_; }

  // Tampon et mémoire importée, puis l'inscription dans les projections.
  Done import(std::uint64_t allocation_size) {
    const DeviceFunctions& fn = context_.fn();
    constexpr auto kHandle = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    const VkExternalMemoryBufferCreateInfo external{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = kHandle};
    const VkBufferCreateInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .pNext = &external,
                                  .size = allocation_size,
                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    if (auto ok =
            check(fn.vkCreateBuffer(context_.device(), &info, nullptr, &buffer_), "vkCreateBuffer (hôte)");
        !ok)
      return ok;
    const auto host_properties = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        context_.instance_functions().vkGetDeviceProcAddr(context_.device(),
                                                          "vkGetMemoryHostPointerPropertiesEXT"));
    if (host_properties == nullptr)
      return detail::refuse_backend(RefusalCode::FeatureUnavailable,
                                    "vkGetMemoryHostPointerPropertiesEXT absente");
    VkMemoryHostPointerPropertiesEXT pointer{.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (auto ok = check(host_properties(context_.device(), kHandle, host_->data(), &pointer),
                        "vkGetMemoryHostPointerPropertiesEXT");
        !ok)
      return ok;
    VkMemoryRequirements requirements{};
    fn.vkGetBufferMemoryRequirements(context_.device(), buffer_, &requirements);
    VkPhysicalDeviceMemoryProperties properties{};
    context_.instance_functions().vkGetPhysicalDeviceMemoryProperties(context_.physical_device(),
                                                                      &properties);
    // Un type permis par le tampon et par l'adresse, cohérent avec l'hôte de préférence.
    std::optional<std::uint32_t> type;
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if ((requirements.memoryTypeBits & pointer.memoryTypeBits & (1U << i)) == 0) continue;
      const bool coherent =
          (properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
      if (!type || coherent) type = i;
      if (coherent) break;
    }
    if (!type || requirements.size > allocation_size)
      return detail::refuse_backend(RefusalCode::FeatureUnavailable,
                                    "mémoire hôte du runtime : aucun type importable");
    const VkImportMemoryHostPointerInfoEXT host_import{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = kHandle,
        .pHostPointer = host_->data()};
    const VkMemoryAllocateInfo allocate{.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                        .pNext = &host_import,
                                        .allocationSize = allocation_size,
                                        .memoryTypeIndex = *type};
    if (auto ok = check(fn.vkAllocateMemory(context_.device(), &allocate, nullptr, &memory_),
                        std::format("vkAllocateMemory ({} o, hôte importée)", allocation_size));
        !ok)
      return ok;
    if (auto ok = check(fn.vkBindBufferMemory(context_.device(), buffer_, memory_, 0), "vkBindBufferMemory");
        !ok)
      return ok;
    tensor_ = {.type = type_, .shape = shape_, .handle = detail::handle_bits(buffer_), .offset = 0};
    const std::scoped_lock lock(views_->mutex);
    views_->views[tensor_.handle] = {.data = host_->data(), .bytes = bytes_};
    return {};
  }

 private:
  VkContext& context_;
  std::unique_ptr<net::HostMemory> host_;
  std::uint64_t bytes_;
  TensorType type_;
  Shape shape_;
  std::shared_ptr<HostViews> views_;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  Tensor tensor_{};
};

}  // namespace

std::uint64_t host_import_alignment(const VkContext& context) {
  if (!std::ranges::contains(context.caps().extensions,
                             std::string_view(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)))
    return 0;
  VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                         .pNext = &host};
  context.instance_functions().vkGetPhysicalDeviceProperties2(context.physical_device(), &properties);
  return host.minImportedHostPointerAlignment;
}

std::expected<std::unique_ptr<TensorBuffer>, Refusal> host_imported_tensor(
    VkContext& context, net::Runtime& runtime, TensorType type, const Shape& shape, std::uint64_t bytes,
    std::uint64_t alignment, std::shared_ptr<HostViews> views) {
  if (alignment == 0)
    return detail::refuse_backend(RefusalCode::FeatureUnavailable, "VK_EXT_external_memory_host absente");
  // L'adresse et la taille importées sont des multiples de l'alignement de l'appareil.
  const std::uint64_t allocation_size = (bytes + alignment - 1) / alignment * alignment;
  auto host = runtime.allocate_host(allocation_size, alignment);
  if (!host) return std::unexpected(host.error());
  if (reinterpret_cast<std::uintptr_t>((*host)->data()) % alignment != 0)
    return detail::refuse_backend(RefusalCode::FeatureUnavailable,
                                  std::format("mémoire hôte du runtime non alignée sur {} o", alignment));
  auto buffer =
      std::make_unique<HostImportedBuffer>(context, std::move(*host), type, shape, bytes, std::move(views));
  if (auto ok = buffer->import(allocation_size); !ok) return std::unexpected(ok.error());
  return buffer;
}

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
    VkContext& /*context*/, net::Runtime& /*runtime*/, TensorType /*type*/, const Shape& /*shape*/,
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

 private:
  VkContext& context_;
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

std::expected<std::unique_ptr<TensorBuffer>, Refusal> shared_tensor(VkContext& context, net::Runtime& runtime,
                                                                    TensorType type, const Shape& shape,
                                                                    std::uint64_t bytes,
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
