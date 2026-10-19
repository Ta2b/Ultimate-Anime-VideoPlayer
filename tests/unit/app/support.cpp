#include "support.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <utility>

namespace ajp::app::test {
namespace {

// --- Fonctions Vulkan factices : elles lisent la configuration de FakeVulkanScope. ---

// Seul état partagé des tests : les fonctions factices n'ont pas d'autre moyen de le recevoir.
FakeVulkan* current = nullptr;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

template <class T>
VkResult fill(std::uint32_t* count, T* items, const std::vector<T>& all) {
  const auto size = static_cast<std::uint32_t>(all.size());
  if (items == nullptr) {
    *count = size;
    return VK_SUCCESS;
  }
  const std::uint32_t written = std::min(*count, size);
  std::copy_n(all.begin(), written, items);
  *count = written;
  return written < size ? VK_INCOMPLETE : VK_SUCCESS;
}

template <std::size_t N>
void copy_text(char (&dest)[N], std::string_view text) {
  const std::size_t length = std::min(text.size(), N - 1);
  std::copy_n(text.data(), length, std::begin(dest));
  std::fill(std::begin(dest) + static_cast<std::ptrdiff_t>(length), std::end(dest), '\0');
}

std::vector<VkExtensionProperties> extension_list(const std::vector<std::string>& names) {
  std::vector<VkExtensionProperties> list(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    copy_text(list[i].extensionName, names[i]);
    list[i].specVersion = 1;
  }
  return list;
}

const FakeDevice& device_of(VkPhysicalDevice physical) {
  return *reinterpret_cast<const FakeDevice*>(physical);
}

VKAPI_ATTR VkResult VKAPI_CALL enumerate_instance_version(std::uint32_t* version) {
  *version = VK_MAKE_API_VERSION(0, 1, 4, 0);
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL enumerate_instance_extensions(const char*, std::uint32_t* count,
                                                             VkExtensionProperties* items) {
  return fill(count, items, extension_list(current->instance_extensions));
}

VKAPI_ATTR VkResult VKAPI_CALL create_instance(const VkInstanceCreateInfo*, const VkAllocationCallbacks*,
                                               VkInstance* instance) {
  if (current->create_result != VK_SUCCESS) return current->create_result;
  ++current->created;
  *instance = reinterpret_cast<VkInstance>(current);
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL destroy_instance(VkInstance, const VkAllocationCallbacks*) {
  ++current->destroyed;
}

VKAPI_ATTR VkResult VKAPI_CALL enumerate_devices(VkInstance, std::uint32_t* count, VkPhysicalDevice* items) {
  std::vector<VkPhysicalDevice> handles;
  for (FakeDevice& device : current->devices) handles.push_back(reinterpret_cast<VkPhysicalDevice>(&device));
  return fill(count, items, handles);
}

VKAPI_ATTR void VKAPI_CALL properties2(VkPhysicalDevice physical, VkPhysicalDeviceProperties2* out) {
  const FakeDevice& device = device_of(physical);
  VkPhysicalDeviceProperties& p = out->properties;
  p.apiVersion = device.api_version;
  p.driverVersion = device.driver_version;
  p.vendorID = device.vendor_id;
  p.deviceID = device.device_id;
  p.deviceType = device.type;
  copy_text(p.deviceName, device.name);
  for (auto* next = static_cast<VkBaseOutStructure*>(out->pNext); next != nullptr; next = next->pNext) {
    if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES) {
      auto* ids = reinterpret_cast<VkPhysicalDeviceIDProperties*>(next);
      std::ranges::copy(device.uuid, std::begin(ids->deviceUUID));
    } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES) {
      auto* driver = reinterpret_cast<VkPhysicalDeviceDriverProperties*>(next);
      copy_text(driver->driverName, device.driver_name);
      copy_text(driver->driverInfo, device.driver_info);
    }
  }
}

VKAPI_ATTR void VKAPI_CALL memory(VkPhysicalDevice physical, VkPhysicalDeviceMemoryProperties* out) {
  *out = {};
  out->memoryHeapCount = 2;
  out->memoryHeaps[0] = {.size = 64ULL << 30, .flags = 0};  // mémoire de l'hôte, jamais comptée
  out->memoryHeaps[1] = {.size = device_of(physical).vram, .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
}

VKAPI_ATTR VkResult VKAPI_CALL device_extensions(VkPhysicalDevice physical, const char*, std::uint32_t* count,
                                                 VkExtensionProperties* items) {
  return fill(count, items, extension_list(device_of(physical).extensions));
}

template <class Fn>
PFN_vkVoidFunction as_void(Fn* function) {
  return reinterpret_cast<PFN_vkVoidFunction>(function);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance_proc_addr(VkInstance instance, const char* name) {
  const std::string_view wanted(name);
  if (instance == VK_NULL_HANDLE) {
    if (wanted == "vkEnumerateInstanceVersion")
      return current->vulkan_1_0 ? nullptr : as_void(&enumerate_instance_version);
    if (wanted == "vkEnumerateInstanceExtensionProperties") return as_void(&enumerate_instance_extensions);
    if (wanted == "vkCreateInstance") return as_void(&create_instance);
    return nullptr;
  }
  if (wanted == "vkDestroyInstance") return as_void(&destroy_instance);
  if (wanted == "vkEnumeratePhysicalDevices") return as_void(&enumerate_devices);
  if (wanted == "vkGetPhysicalDeviceProperties2") return as_void(&properties2);
  if (wanted == "vkGetPhysicalDeviceMemoryProperties") return as_void(&memory);
  if (wanted == "vkEnumerateDeviceExtensionProperties") return as_void(&device_extensions);
  return nullptr;
}

// --- Pilote CUDA factice. ---

int cuda_13_4(int* version) {
  *version = 13040;
  return 0;
}
int cuda_12_8(int* version) {
  *version = 12080;
  return 0;
}
int cuda_broken(int* version) {
  *version = 0;
  return 3;  // CUDA_ERROR_NOT_INITIALIZED
}

class FakeLibrary final : public LoadedLibrary {
 public:
  explicit FakeLibrary(std::map<std::string, void*> symbols) : symbols_(std::move(symbols)) {}
  void* raw_symbol(const char* name) const override {
    const auto found = symbols_.find(name);
    return found == symbols_.end() ? nullptr : found->second;
  }

 private:
  std::map<std::string, void*> symbols_;
};

std::array<std::uint8_t, VK_UUID_SIZE> uuid(std::uint8_t first) {
  std::array<std::uint8_t, VK_UUID_SIZE> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(first + i);
  return bytes;
}

FakeDevice device(std::string name, std::uint32_t vendor, std::uint32_t id, VkPhysicalDeviceType type,
                  std::uint64_t vram_gib, bool matrix) {
  FakeDevice d{.name = std::move(name),
               .vendor_id = vendor,
               .device_id = id,
               .type = type,
               .uuid = uuid(static_cast<std::uint8_t>(id)),
               .vram = vram_gib << 30,
               .extensions = {"VK_KHR_driver_properties", "VK_KHR_swapchain"}};
  if (matrix) d.extensions.emplace_back("VK_KHR_cooperative_matrix");
  return d;
}

}  // namespace

TempDir::TempDir() {
  std::random_device random;
  path_ =
      fs::temp_directory_path() / ("ajp-app-" + std::to_string(random()) + "-" + std::to_string(random()));
  fs::create_directories(path_);
}

TempDir::~TempDir() {
  std::error_code ignored;
  fs::remove_all(path_, ignored);
}

void write_file(const fs::path& file, std::string_view bytes) {
  fs::create_directories(file.parent_path());
  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string read_file(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

std::string test_bytes(std::size_t size, std::uint32_t seed) {
  std::minstd_rand random(seed);
  std::string bytes(size, '\0');
  for (char& byte : bytes) byte = static_cast<char>(random() & 0xFF);
  return bytes;
}

void FakeLoader::add(std::string name, std::map<std::string, void*> symbols) {
  libraries_.insert_or_assign(std::move(name), std::move(symbols));
}

std::expected<std::unique_ptr<LoadedLibrary>, Refusal> FakeLoader::open(std::string_view name) const {
  const auto found = libraries_.find(name);
  if (found == libraries_.end()) {
    return std::unexpected(
        Refusal{RefusalCode::LibraryUnavailable, std::string(name) + " : absente (factice)"});
  }
  return std::make_unique<FakeLibrary>(found->second);
}

FakeVulkanScope::FakeVulkanScope(FakeLoader& loader, FakeVulkan& fake) {
  current = &fake;
  const std::map<std::string, void*> symbols{
      {"vkGetInstanceProcAddr", reinterpret_cast<void*>(&get_instance_proc_addr)}};
  for (const char* name : {"libvulkan.so.1", "vulkan-1.dll", "libvulkan.1.dylib"}) loader.add(name, symbols);
}

FakeVulkanScope::~FakeVulkanScope() {
  current = nullptr;
}

void add_fake_cuda(FakeLoader& loader, FakeCuda cuda) {
  void* function = nullptr;
  switch (cuda) {
    case FakeCuda::Cuda13_4:
      function = reinterpret_cast<void*>(&cuda_13_4);
      break;
    case FakeCuda::Cuda12_8:
      function = reinterpret_cast<void*>(&cuda_12_8);
      break;
    case FakeCuda::Broken:
      function = reinterpret_cast<void*>(&cuda_broken);
      break;
    case FakeCuda::NoSymbol:
      break;
  }
  std::map<std::string, void*> symbols;
  if (function != nullptr) symbols.emplace("cuDriverGetVersion", function);
  for (const char* name : {"libcuda.so.1", "nvcuda.dll"}) loader.add(name, symbols);
}

FakeDevice rtx_4070_ti() {
  FakeDevice d =
      device("NVIDIA GeForce RTX 4070 Ti", 0x10DE, 0x2782, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 12, true);
  d.api_version = VK_MAKE_API_VERSION(0, 1, 4, 312);
  d.driver_version = (615U << 22) | (78U << 14) | (8U << 6);
  d.driver_name = "NVIDIA";
  d.driver_info = "615.78.08";
  d.extensions.emplace_back("VK_NV_cooperative_matrix");
  return d;
}

FakeDevice gtx_1660() {
  FakeDevice d =
      device("NVIDIA GeForce GTX 1660 SUPER", 0x10DE, 0x21C4, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 6, false);
  d.driver_name = "NVIDIA";
  d.driver_info = "580.95.05";
  return d;
}

FakeDevice radeon_7900() {
  FakeDevice d = device("AMD Radeon RX 7900 XTX (RADV NAVI31)", 0x1002, 0x744C,
                        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 24, true);
  d.driver_name = "radv";
  d.driver_info = "Mesa 25.2.4";
  return d;
}

FakeDevice radeon_6800() {
  FakeDevice d = device("AMD Radeon RX 6800 (RADV NAVI21)", 0x1002, 0x73BF,
                        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 16, false);
  d.driver_name = "radv";
  d.driver_info = "Mesa 25.2.4";
  return d;
}

FakeDevice arc_a770() {
  FakeDevice d = device("Intel(R) Arc(tm) A770 Graphics (DG2)", 0x8086, 0x56A0,
                        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 16, true);
  d.driver_name = "Intel open-source Mesa driver";
  d.driver_info = "Mesa 25.2.4";
  return d;
}

FakeDevice uhd_770() {
  FakeDevice d = device("Intel(R) UHD Graphics 770 (ADL-S GT1)", 0x8086, 0x4680,
                        VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 8, false);
  d.api_version = VK_API_VERSION_1_1;  // sans VkPhysicalDeviceDriverProperties ni extension : pilote inconnu
  d.extensions = {};
  d.driver_version = VK_MAKE_API_VERSION(0, 24, 1, 7);
  return d;
}

FakeDevice llvmpipe() {
  FakeDevice d =
      device("llvmpipe (LLVM 20.1.2, 256 bits)", 0x10005, 0, VK_PHYSICAL_DEVICE_TYPE_CPU, 32, false);
  d.driver_name = "llvmpipe";
  d.driver_info = "Mesa 26.2.4 (LLVM 20.1.2)";
  return d;
}

}  // namespace ajp::app::test
