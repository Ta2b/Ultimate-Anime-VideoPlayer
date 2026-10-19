#include "app/vulkan_probe.h"

#include <algorithm>
#include <format>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace ajp::app {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

std::string result_name(VkResult result) {
  switch (result) {
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER (aucun pilote Vulkan installé)";
    default:
      return std::format("VkResult {}", static_cast<int>(result));
  }
}

std::unexpected<Refusal> vk_failed(std::string_view call, VkResult result) {
  return refuse(RefusalCode::VulkanCallFailed, std::format("{} : {}", call, result_name(result)));
}

// Tableau de taille fixe, terminé par un zéro s'il ne remplit pas tout.
template <std::size_t N>
std::string text_of(const char (&chars)[N]) {
  const std::span<const char, N> all(chars);
  return {all.begin(), std::ranges::find(all, '\0')};
}

// Remplit une liste par le motif Vulkan « compter, puis lire », VK_INCOMPLETE compris.
template <class T, class Enumerate>
std::expected<std::vector<T>, Refusal> enumerate(std::string_view call, Enumerate&& fill) {
  std::vector<T> items;
  VkResult result = VK_INCOMPLETE;
  while (result == VK_INCOMPLETE) {
    std::uint32_t count = 0;
    result = fill(&count, static_cast<T*>(nullptr));
    if (result != VK_SUCCESS) return vk_failed(call, result);
    items.resize(count);
    result = fill(&count, items.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) return vk_failed(call, result);
    items.resize(count);
  }
  return items;
}

bool has_extension(const std::vector<VkExtensionProperties>& extensions, std::string_view name) {
  return std::ranges::any_of(
      extensions, [&](const VkExtensionProperties& e) { return text_of(e.extensionName) == name; });
}

// Instance détruite à la sortie de la sonde, avant la fermeture du chargeur.
class Instance {
 public:
  Instance(VkInstance handle, PFN_vkDestroyInstance destroy) : handle_(handle), destroy_(destroy) {}
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  ~Instance() {
    if (destroy_ != nullptr) destroy_(handle_, nullptr);
  }
  VkInstance get() const { return handle_; }

 private:
  VkInstance handle_;
  PFN_vkDestroyInstance destroy_;
};

struct DeviceFunctions {
  PFN_vkEnumeratePhysicalDevices enumerate_devices = nullptr;
  PFN_vkGetPhysicalDeviceProperties2 properties2 = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties memory = nullptr;
  PFN_vkEnumerateDeviceExtensionProperties extensions = nullptr;
};

DeviceType type_of(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return DeviceType::Discrete;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return DeviceType::Integrated;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return DeviceType::Virtual;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return DeviceType::Cpu;
    default:
      return DeviceType::Other;
  }
}

VulkanDevice read_device(const DeviceFunctions& vk, VkPhysicalDevice physical) {
  // Sans la liste des extensions, l'appareil reste décrit, sans unités matricielles ni pilote.
  const std::vector<VkExtensionProperties> extensions =
      enumerate<VkExtensionProperties>("vkEnumerateDeviceExtensionProperties",
                                       [&](std::uint32_t* count, VkExtensionProperties* items) {
                                         return vk.extensions(physical, nullptr, count, items);
                                       })
          .value_or(std::vector<VkExtensionProperties>{});

  VkPhysicalDeviceIDProperties ids{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
  VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                         .pNext = &ids};
  vk.properties2(physical, &properties);
  const VkPhysicalDeviceProperties& p = properties.properties;

  VulkanDevice device{.name = text_of(p.deviceName),
                      .vendor_id = p.vendorID,
                      .device_id = p.deviceID,
                      .type = type_of(p.deviceType),
                      .api_version = p.apiVersion,
                      .driver_version = p.driverVersion};
  std::ranges::copy(ids.deviceUUID, device.uuid.begin());
  device.uuid_known = std::ranges::any_of(device.uuid, [](std::uint8_t b) { return b != 0; });

  // Structure de Vulkan 1.2 : ne s'enchaîne que si l'appareil la connaît.
  if (p.apiVersion >= VK_API_VERSION_1_2 || has_extension(extensions, "VK_KHR_driver_properties")) {
    VkPhysicalDeviceDriverProperties driver{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 with_driver{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                            .pNext = &driver};
    vk.properties2(physical, &with_driver);
    device.driver_name = text_of(driver.driverName);
    device.driver_info = text_of(driver.driverInfo);
  }

  VkPhysicalDeviceMemoryProperties memory{};
  vk.memory(physical, &memory);
  for (std::uint32_t i = 0; i < memory.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; ++i) {
    const VkMemoryHeap& heap = memory.memoryHeaps[i];
    if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
      device.vram_bytes = std::max(device.vram_bytes, heap.size);
  }
  device.cooperative_matrix = has_extension(extensions, "VK_KHR_cooperative_matrix") ||
                              has_extension(extensions, "VK_NV_cooperative_matrix");
  return device;
}

std::expected<std::unique_ptr<LoadedLibrary>, Refusal> open_loader(const LibraryLoader& loader, core::Os os) {
  std::string tried;
  for (const std::string& name : vulkan_loader_names(os)) {
    auto library = loader.open(name);
    if (library) return library;
    tried += std::format("{}{}", tried.empty() ? "" : " ; ", library.error().detail);
  }
  return refuse(RefusalCode::LibraryUnavailable, std::format("chargeur Vulkan absent ({})", tried));
}

}  // namespace

std::vector<std::string> vulkan_loader_names(core::Os os) {
  switch (os) {
    case core::Os::Windows:
      return {"vulkan-1.dll"};
    case core::Os::MacOs:
      return {"libvulkan.1.dylib", "libMoltenVK.dylib"};
    case core::Os::Linux:
      return {"libvulkan.so.1"};
  }
  return {};
}

std::expected<std::vector<VulkanDevice>, Refusal> probe_vulkan(const LibraryLoader& loader, core::Os os) {
  auto library = open_loader(loader, os);
  if (!library) return std::unexpected(library.error());
  const auto get =
      (*library)->symbol<std::remove_pointer_t<PFN_vkGetInstanceProcAddr>>("vkGetInstanceProcAddr");
  if (get == nullptr)
    return refuse(RefusalCode::LibraryUnavailable, "vkGetInstanceProcAddr absent du chargeur Vulkan");
  const auto global = [&](const char* name) { return get(nullptr, name); };

  const auto version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(global("vkEnumerateInstanceVersion"));
  const auto create = reinterpret_cast<PFN_vkCreateInstance>(global("vkCreateInstance"));
  const auto list_extensions = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
      global("vkEnumerateInstanceExtensionProperties"));
  if (version == nullptr)
    return refuse(RefusalCode::VulkanCallFailed, "chargeur Vulkan 1.0 : 1.1 au moins est requis");
  if (create == nullptr || list_extensions == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable, "vkCreateInstance absent du chargeur Vulkan");
  }

  // MoltenVK ne s'énumère qu'avec l'extension de portabilité.
  auto extensions = enumerate<VkExtensionProperties>("vkEnumerateInstanceExtensionProperties",
                                                     [&](std::uint32_t* count, VkExtensionProperties* items) {
                                                       return list_extensions(nullptr, count, items);
                                                     });
  if (!extensions) return std::unexpected(extensions.error());
  std::vector<const char*> enabled;
  VkInstanceCreateFlags flags = 0;
  if (os == core::Os::MacOs && has_extension(*extensions, "VK_KHR_portability_enumeration")) {
    enabled.push_back("VK_KHR_portability_enumeration");
    flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  }
  const VkApplicationInfo application{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                      .pApplicationName = "ajp",
                                      .applicationVersion = 1,
                                      .pEngineName = "ajp",
                                      .engineVersion = 1,
                                      .apiVersion = VK_API_VERSION_1_1};
  const VkInstanceCreateInfo info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                  .flags = flags,
                                  .pApplicationInfo = &application,
                                  .enabledExtensionCount = static_cast<std::uint32_t>(enabled.size()),
                                  .ppEnabledExtensionNames = enabled.data()};
  VkInstance handle = VK_NULL_HANDLE;
  if (const VkResult result = create(&info, nullptr, &handle); result != VK_SUCCESS) {
    return vk_failed("vkCreateInstance", result);
  }
  const Instance instance(handle, reinterpret_cast<PFN_vkDestroyInstance>(get(handle, "vkDestroyInstance")));

  const auto local = [&](const char* name) { return get(instance.get(), name); };
  const DeviceFunctions vk{
      .enumerate_devices =
          reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(local("vkEnumeratePhysicalDevices")),
      .properties2 =
          reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(local("vkGetPhysicalDeviceProperties2")),
      .memory = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
          local("vkGetPhysicalDeviceMemoryProperties")),
      .extensions = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
          local("vkEnumerateDeviceExtensionProperties"))};
  if (vk.enumerate_devices == nullptr || vk.properties2 == nullptr || vk.memory == nullptr ||
      vk.extensions == nullptr) {
    return refuse(RefusalCode::LibraryUnavailable, "fonctions d'appareil absentes de l'instance Vulkan");
  }

  auto physical = enumerate<VkPhysicalDevice>("vkEnumeratePhysicalDevices",
                                              [&](std::uint32_t* count, VkPhysicalDevice* items) {
                                                return vk.enumerate_devices(instance.get(), count, items);
                                              });
  if (!physical) return std::unexpected(physical.error());
  std::vector<VulkanDevice> devices;
  for (VkPhysicalDevice device : *physical) devices.push_back(read_device(vk, device));
  return devices;
}

core::GpuVendor vendor_of(std::uint32_t pci_vendor) {
  switch (pci_vendor) {
    case pci::kNvidia:
      return core::GpuVendor::Nvidia;
    case pci::kAmd:
      return core::GpuVendor::Amd;
    case pci::kIntel:
      return core::GpuVendor::Intel;
    case pci::kApple:
      return core::GpuVendor::Apple;
    default:
      return core::GpuVendor::Other;
  }
}

std::string driver_version_text(const VulkanDevice& device, core::Os os) {
  if (!device.driver_info.empty()) return device.driver_info;
  const std::uint32_t v = device.driver_version;
  if (device.vendor_id == pci::kNvidia)
    return std::format("{}.{}.{:02}", v >> 22, (v >> 14) & 0xFF, (v >> 6) & 0xFF);
  if (device.vendor_id == pci::kIntel && os == core::Os::Windows)
    return std::format("{}.{}", v >> 14, v & 0x3FFF);
  return std::format("{}.{}.{}", v >> 22, (v >> 12) & 0x3FF, v & 0xFFF);
}

std::string uuid_text(const DeviceUuid& uuid) {
  std::string text;
  for (std::size_t i = 0; i < uuid.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) text += '-';
    text += std::format("{:02x}", uuid[i]);
  }
  return text;
}

}  // namespace ajp::app
