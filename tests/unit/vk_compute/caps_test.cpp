// Capacités d'un appareil lues sur des fonctions Vulkan factices : la chaîne de propriétés ne dépend de la
// version de l'API que pour la taille des sous-groupes, pas pour l'adresse PCI.
#include "gpu/vulkan/compute/caps.h"

#include <cstdint>
#include <cstring>

#include <catch2/catch_test_macros.hpp>

#include "gpu/vulkan/compute/functions.h"

using namespace ajp::gpu::vk;

namespace {

std::uint32_t g_fake_api = VK_API_VERSION_1_2;

VKAPI_ATTR void VKAPI_CALL fake_properties(VkPhysicalDevice, VkPhysicalDeviceProperties* out) {
  *out = {};
  out->apiVersion = g_fake_api;
  std::strncpy(static_cast<char*>(out->deviceName), "faux appareil", VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
}

VKAPI_ATTR VkResult VKAPI_CALL fake_extensions(VkPhysicalDevice, const char*, std::uint32_t* count,
                                               VkExtensionProperties* out) {
  *count = 1;
  if (out != nullptr) {
    out[0] = {};
    std::strncpy(static_cast<char*>(out[0].extensionName), VK_EXT_PCI_BUS_INFO_EXTENSION_NAME,
                 VK_MAX_EXTENSION_NAME_SIZE - 1);
  }
  return VK_SUCCESS;
}

// Remplit seulement les structures que la chaîne contient : une structure absente reste à zéro, comme sur un
// pilote qui ne la connaît pas.
VKAPI_ATTR void VKAPI_CALL fake_properties2(VkPhysicalDevice, VkPhysicalDeviceProperties2* out) {
  for (auto* link = static_cast<VkBaseOutStructure*>(out->pNext); link != nullptr; link = link->pNext) {
    if (link->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT) {
      auto* pci = reinterpret_cast<VkPhysicalDevicePCIBusInfoPropertiesEXT*>(link);
      pci->pciDomain = 0;
      pci->pciBus = 0x2b;
      pci->pciDevice = 0;
      pci->pciFunction = 0;
    }
  }
}

InstanceFunctions fake_functions() {
  InstanceFunctions fn{};
  fn.vkGetPhysicalDeviceProperties = fake_properties;
  fn.vkEnumerateDeviceExtensionProperties = fake_extensions;
  fn.vkGetPhysicalDeviceProperties2 = fake_properties2;
  return fn;
}

}  // namespace

TEST_CASE("Capacités : l'adresse PCI se lit aussi sur une API 1.2", "[vk_compute][caps]") {
  // B49-03 : sans l'API 1.3, la structure PCI n'était pas chaînée et l'appariement par adresse échouait.
  const InstanceFunctions fn = fake_functions();
  for (const std::uint32_t api : {VK_API_VERSION_1_2, VK_API_VERSION_1_3}) {
    g_fake_api = api;
    const DeviceCaps caps = probe_device(fn, VK_NULL_HANDLE, api);
    INFO("API " << VK_API_VERSION_MINOR(api));
    REQUIRE(caps.pci.has_value());
    CHECK(caps.pci->bus == 0x2b);
  }
}
