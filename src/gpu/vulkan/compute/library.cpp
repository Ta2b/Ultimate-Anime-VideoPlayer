#include "gpu/vulkan/compute/library.h"

#include <format>
#include <string>
#include <utility>

#include "gpu/vulkan/compute/refuse.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ajp::gpu::vk {
namespace {

#if defined(_WIN32)
constexpr const char* kSystemLoader = "vulkan-1.dll";
#elif defined(__APPLE__)
constexpr const char* kSystemLoader = "libvulkan.1.dylib";
#else
constexpr const char* kSystemLoader = "libvulkan.so.1";
#endif

std::string shown(const std::filesystem::path& path) {
  const std::u8string text = path.u8string();
  return {text.begin(), text.end()};
}

}  // namespace

std::expected<VulkanLibrary, Refusal> VulkanLibrary::open_system() {
  return load(kSystemLoader, true);
}

std::expected<VulkanLibrary, Refusal> VulkanLibrary::open(const std::filesystem::path& path) {
  return load(path, false);
}

// search : le nom seul est confié au chargeur dynamique du système (bibliothèque système) ;
// sinon le chemin est ouvert tel quel.
std::expected<VulkanLibrary, Refusal> VulkanLibrary::load(const std::filesystem::path& path, bool search) {
  const std::string name = shown(path);
#ifdef _WIN32
  const DWORD flags = search ? LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
                             : LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
  HMODULE module = LoadLibraryExW(path.c_str(), nullptr, flags);
  if (module == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("{} ne se charge pas (erreur Windows {})", name, GetLastError()));
  }
  void* handle = static_cast<void*>(module);
  auto* address = reinterpret_cast<void*>(GetProcAddress(module, "vkGetInstanceProcAddr"));
#else
  (void)search;  // dlopen cherche lui-même un nom sans barre oblique
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* error = dlerror();
    return refuse(RefusalCode::LibraryUnavailable, std::format("{} ne se charge pas : {}", name,
                                                               error != nullptr ? error : "erreur inconnue"));
  }
  void* address = dlsym(handle, "vkGetInstanceProcAddr");
#endif
  VulkanLibrary library(handle, reinterpret_cast<PFN_vkGetInstanceProcAddr>(address));
  if (address == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable,
                  std::format("{} n'exporte pas vkGetInstanceProcAddr", name));
  }
  return library;
}

VulkanLibrary::VulkanLibrary(VulkanLibrary&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)),
      get_instance_proc_addr_(std::exchange(other.get_instance_proc_addr_, nullptr)) {}

VulkanLibrary& VulkanLibrary::operator=(VulkanLibrary&& other) noexcept {
  if (this != &other) {
    close();
    handle_ = std::exchange(other.handle_, nullptr);
    get_instance_proc_addr_ = std::exchange(other.get_instance_proc_addr_, nullptr);
  }
  return *this;
}

VulkanLibrary::~VulkanLibrary() {
  close();
}

void VulkanLibrary::close() {
  if (handle_ == nullptr) return;
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(handle_));
#else
  dlclose(handle_);
#endif
  handle_ = nullptr;
  get_instance_proc_addr_ = nullptr;
}

}  // namespace ajp::gpu::vk
