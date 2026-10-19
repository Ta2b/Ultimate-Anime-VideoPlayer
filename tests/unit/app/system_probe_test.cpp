// Sonde du vrai système : chargeur Vulkan et pilotes installés. Sur la RTX de référence
// (ajp-gpu), en CI sur lavapipe (with-lavapipe) ; SKIP avec la raison sans Vulkan.
#include <algorithm>
#include <format>

#include <catch2/catch_test_macros.hpp>

#include "app/data_paths.h"
#include "app/library_loader.h"
#include "app/system_report.h"
#include "app/vulkan_probe.h"
#include "support.h"

using namespace ajp;
using namespace ajp::app;

namespace {

#if defined(_WIN32)
constexpr const char* kSystemLibrary = "kernel32.dll";
constexpr const char* kSystemSymbol = "GetTickCount64";
#elif defined(__APPLE__)
constexpr const char* kSystemLibrary = "libSystem.B.dylib";
constexpr const char* kSystemSymbol = "malloc";
#else
constexpr const char* kSystemLibrary = "libc.so.6";
constexpr const char* kSystemSymbol = "malloc";
#endif

}  // namespace

TEST_CASE("Chargeur du système : bibliothèque trouvée par son nom, symboles, refus nommé", "[app][probe]") {
  const auto loader = system_library_loader();
  const auto library = loader->open(kSystemLibrary);
  REQUIRE(library);
  CHECK((*library)->raw_symbol(kSystemSymbol) != nullptr);
  CHECK((*library)->raw_symbol("ajp_symbole_absent") == nullptr);

  const auto missing = loader->open("libajp_absente_pour_le_test.so.9");
  REQUIRE_FALSE(missing);
  CHECK(missing.error().code == RefusalCode::LibraryUnavailable);
  CHECK_FALSE(missing.error().detail.empty());
}

TEST_CASE("Sonde système réelle : les GPU de ce système par son chargeur Vulkan", "[app][vulkan]") {
  const auto loader = system_library_loader();
  const SystemReport report = detect_system(*loader, host_os());
  if (report.vulkan_refusal) {
    SKIP(std::format("Vulkan indisponible : {} : {}", name(report.vulkan_refusal->code),
                     report.vulkan_refusal->detail));
  }
  if (report.devices.empty()) SKIP("aucun appareil Vulkan sur ce système");
  for (const std::string& line : describe(report)) UNSCOPED_INFO(line);

  for (const VulkanDevice& device : report.devices) {
    INFO(device.name);
    CHECK_FALSE(device.name.empty());
    CHECK(device.vendor_id != 0);
    CHECK(device.api_version >= VK_API_VERSION_1_1);
    CHECK(device.uuid_known);
    CHECK_FALSE(driver_version_text(device, host_os()).empty());
    if (device.type != DeviceType::Cpu) CHECK(device.vram_bytes > 0);
  }
  REQUIRE(render_device(report.devices).has_value());
  CHECK(to_system_info(report).gpus.size() == report.devices.size());
}

TEST_CASE("Sonde système réelle : une RTX recommande TensorRT et RTX HDR", "[app][nvidia]") {
  const auto loader = system_library_loader();
  const SystemReport report = detect_system(*loader, host_os());
  const auto rtx = std::ranges::find_if(report.devices, [](const VulkanDevice& device) {
    return device.vendor_id == pci::kNvidia && device.name.find("RTX") != std::string::npos;
  });
  if (rtx == report.devices.end()) SKIP("aucune RTX vue par Vulkan sur ce système");
  for (const std::string& line : describe(report)) UNSCOPED_INFO(line);

  CHECK(rtx->type == DeviceType::Discrete);
  CHECK(rtx->cooperative_matrix);
  CHECK(rtx->driver_name.find("NVIDIA") != std::string::npos);
  REQUIRE(report.runtimes.cuda.has_value());  // le pilote NVIDIA installe toujours libcuda
  CHECK(report.runtimes.cuda->version >= 11000);

  if (static_cast<std::size_t>(rtx - report.devices.begin()) == render_device(report.devices)) {
    const auto offers = core::recommend_engines(to_system_info(report));
    const auto tensorrt = std::ranges::find(offers, core::EngineId::TensorRt, &core::EngineOffer::id);
    REQUIRE(tensorrt != offers.end());
    CHECK(tensorrt->preselected);
  }
}
