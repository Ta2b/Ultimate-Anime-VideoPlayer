#include "gpu/vulkan/compute/library.h"

#include <format>
#include <system_error>
#include <type_traits>
#include <utility>

#include "gpu/vulkan/compute/refuse.h"

namespace ajp::gpu::vk {
namespace {

#if defined(_WIN32)
constexpr const char* kSystemLoader = "vulkan-1.dll";
#elif defined(__APPLE__)
constexpr const char* kSystemLoader = "libvulkan.1.dylib";
#else
constexpr const char* kSystemLoader = "libvulkan.so.1";
#endif

}  // namespace

std::expected<VulkanLibrary, Refusal> VulkanLibrary::open_system() {
  auto library = DynamicLibrary::open_system(kSystemLoader);
  if (!library) return std::unexpected(std::move(library.error()));
  return wrap(std::move(*library));
}

std::expected<VulkanLibrary, Refusal> VulkanLibrary::open(const std::filesystem::path& path) {
  // DynamicLibrary refuse un chemin relatif : celui d'un outil se lit depuis le dossier courant.
  std::error_code error;
  const std::filesystem::path absolute = std::filesystem::absolute(path, error);
  auto library = DynamicLibrary::open(error ? path : absolute);
  if (!library) return std::unexpected(std::move(library.error()));
  return wrap(std::move(*library));
}

std::expected<VulkanLibrary, Refusal> VulkanLibrary::wrap(DynamicLibrary library) {
  auto entry = library.symbol<std::remove_pointer_t<PFN_vkGetInstanceProcAddr>>("vkGetInstanceProcAddr");
  if (!entry) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("{} n'exporte pas vkGetInstanceProcAddr", library_path_text(library.path())));
  }
  return VulkanLibrary(std::move(library), *entry);
}

}  // namespace ajp::gpu::vk
