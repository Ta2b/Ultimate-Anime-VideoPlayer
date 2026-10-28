// Les tenseurs d'un runtime dans sa propre mémoire hôte, importée par Vulkan (shared_tensor.h,
// host_imported_tensor) : USM d'OpenVINO et IOSurface de Core ML sur un GPU intégré, mémoire
// épinglée de MIGraphX sous Windows.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <string_view>
#include <utility>

#include "gpu/vulkan/backend/kernel_ops_detail.h"
#include "gpu/vulkan/backend/shared_tensor.h"

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
    // Un type permis par le tampon et par l'adresse, et cohérent avec l'hôte : aucun flush ni
    // invalidate n'existe aux frontières des étapes du runtime, un type non cohérent est refusé
    // (le tensor est alors relu par copie).
    std::optional<std::uint32_t> type;
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if ((requirements.memoryTypeBits & pointer.memoryTypeBits & (1U << i)) == 0) continue;
      if ((properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) continue;
      type = i;
      break;
    }
    if (!type || requirements.size > allocation_size)
      return detail::refuse_backend(RefusalCode::FeatureUnavailable,
                                    "mémoire hôte du runtime : aucun type importable");
    coherent_ = (properties.memoryTypes[*type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
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

  // Le GPU voit-il la mémoire de l'hôte elle-même, et pas une copie (un pilote qui accepterait
  // l'adresse sans la partager) ? Il recopie un mot que le CPU a écrit et en remplit un autre, que
  // le CPU relit. Sur une mémoire cohérente seulement ; remise à zéro ensuite.
  Done verify() {
    constexpr std::uint32_t kFromHost = 0xA5C3'5A3CU;
    constexpr std::uint32_t kFromDevice = 0x3C5A'C3A5U;
    if (!coherent_ || bytes_ < 12) return {};
    std::byte* const data = host_->data();
    const auto word = [data](std::size_t at) {
      std::uint32_t value = 0;
      std::memcpy(&value, data + (at * 4), 4);
      return value;
    };
    std::memcpy(data, &kFromHost, 4);
    std::memset(data + 4, 0, 8);
    const DeviceFunctions& fn = context_.fn();
    auto ran = context_.run_once([&](VkCommandBuffer cb) {
      const VkMemoryBarrier from_host{
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
      fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                              &from_host, 0, nullptr, 0, nullptr);
      const VkBufferCopy region{.srcOffset = 0, .dstOffset = 4, .size = 4};
      fn.vkCmdCopyBuffer(cb, buffer_, buffer_, 1, &region);
      fn.vkCmdFillBuffer(cb, buffer_, 8, 4, kFromDevice);
      const VkMemoryBarrier to_host{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                    .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
      fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host,
                              0, nullptr, 0, nullptr);
    });
    if (!ran) return ran;
    const bool shared = word(1) == kFromHost && word(2) == kFromDevice;
    std::memset(data, 0, 12);
    if (!shared)
      return detail::refuse_backend(RefusalCode::FeatureUnavailable,
                                    "mémoire hôte du runtime : Vulkan n'en voit qu'une copie");
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
  bool coherent_ = false;
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
    std::uint64_t alignment, std::shared_ptr<HostViews> views, bool verify) {
  if (alignment == 0)
    return detail::refuse_backend(RefusalCode::FeatureUnavailable, "VK_EXT_external_memory_host absente");
  // L'adresse et la taille importées sont des multiples de l'alignement de l'appareil.
  const std::uint64_t allocation_size = (bytes + alignment - 1) / alignment * alignment;
  auto host = runtime.allocate_host(allocation_size, alignment, type, shape);
  if (!host) return std::unexpected(host.error());
  if (reinterpret_cast<std::uintptr_t>((*host)->data()) % alignment != 0)
    return detail::refuse_backend(RefusalCode::FeatureUnavailable,
                                  std::format("mémoire hôte du runtime non alignée sur {} o", alignment));
  auto buffer =
      std::make_unique<HostImportedBuffer>(context, std::move(*host), type, shape, bytes, std::move(views));
  if (auto ok = buffer->import(allocation_size); !ok) return std::unexpected(ok.error());
  if (verify) {
    if (auto ok = buffer->verify(); !ok) return std::unexpected(ok.error());
  }
  return buffer;
}

}  // namespace ajp::gpu::vk
