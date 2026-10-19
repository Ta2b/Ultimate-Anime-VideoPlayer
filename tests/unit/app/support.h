// Outils des tests app : dossiers temporaires, et un système factice derrière
// LibraryLoader (chargeur Vulkan qui décrit des GPU inventés, pilote CUDA factice).
// Tout se génère pendant le test : aucun binaire n'entre dans le dépôt.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include "app/library_loader.h"

namespace ajp::app::test {

namespace fs = std::filesystem;

// Dossier neuf, supprimé avec son contenu à la fin du test.
class TempDir {
 public:
  TempDir();
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  ~TempDir();
  const fs::path& path() const { return path_; }
  fs::path operator/(const fs::path& name) const { return path_ / name; }

 private:
  fs::path path_;
};

void write_file(const fs::path& file, std::string_view bytes);
std::string read_file(const fs::path& file);
// Octets pseudo-aléatoires reproductibles, pour les corps servis par le serveur de test.
std::string test_bytes(std::size_t size, std::uint32_t seed = 1);

// --- Système factice. ---

class FakeLoader final : public LibraryLoader {
 public:
  void add(std::string name, std::map<std::string, void*> symbols);
  std::expected<std::unique_ptr<LoadedLibrary>, Refusal> open(std::string_view name) const override;

 private:
  std::map<std::string, std::map<std::string, void*>, std::less<>> libraries_;
};

struct FakeDevice {
  std::string name;
  std::uint32_t vendor_id = 0;
  std::uint32_t device_id = 0;
  VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  std::uint32_t api_version = VK_API_VERSION_1_3;
  std::uint32_t driver_version = 0;
  std::string driver_name;
  std::string driver_info;
  std::array<std::uint8_t, VK_UUID_SIZE> uuid{};
  std::uint64_t vram = 0;
  std::vector<std::string> extensions;
};

struct FakeVulkan {
  std::vector<FakeDevice> devices;
  std::vector<std::string> instance_extensions;
  VkResult create_result = VK_SUCCESS;
  bool vulkan_1_0 = false;  // chargeur sans vkEnumerateInstanceVersion
  int created = 0;
  int destroyed = 0;
};

// Rend fake visible comme libvulkan.so.1 (et vulkan-1.dll, libvulkan.1.dylib) de loader le
// temps de la portée. Les fonctions Vulkan factices sont de simples fonctions : elles lisent
// cette configuration, une seule à la fois.
class FakeVulkanScope {
 public:
  FakeVulkanScope(FakeLoader& loader, FakeVulkan& fake);
  FakeVulkanScope(const FakeVulkanScope&) = delete;
  FakeVulkanScope& operator=(const FakeVulkanScope&) = delete;
  ~FakeVulkanScope();
};

// Pilote CUDA factice (libcuda.so.1, nvcuda.dll) dont cuDriverGetVersion rend 13040 (CUDA 13.4),
// 12080 (CUDA 12.8), ou une erreur.
enum class FakeCuda : std::uint8_t { Cuda13_4, Cuda12_8, Broken, NoSymbol };
void add_fake_cuda(FakeLoader& loader, FakeCuda cuda);

FakeDevice rtx_4070_ti();
FakeDevice gtx_1660();     // NVIDIA sans unités matricielles
FakeDevice radeon_7900();  // AMD RDNA3, unités matricielles
FakeDevice radeon_6800();  // AMD RDNA2, sans
FakeDevice arc_a770();     // Intel dédié, XMX
FakeDevice uhd_770();      // Intel intégré, sans XMX
FakeDevice llvmpipe();     // rendu logiciel

}  // namespace ajp::app::test
