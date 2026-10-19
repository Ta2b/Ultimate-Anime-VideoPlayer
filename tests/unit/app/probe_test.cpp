// Détection du système par des sondes factices : chargeur Vulkan qui décrit des GPU
// inventés, pilote CUDA factice. La sonde réelle est dans system_probe_test.cpp.
#include <algorithm>

#include <catch2/catch_test_macros.hpp>

#include "app/runtime_probe.h"
#include "app/system_report.h"
#include "app/vulkan_probe.h"
#include "support.h"

using namespace ajp;
using namespace ajp::app;
using namespace ajp::app::test;

namespace {

SystemReport detect(std::vector<FakeDevice> devices, core::Os os = core::Os::Linux) {
  FakeLoader loader;
  FakeVulkan fake{.devices = std::move(devices)};
  const FakeVulkanScope scope(loader, fake);
  return detect_system(loader, os);
}

core::SystemInfo info_of(std::vector<FakeDevice> devices, core::Os os = core::Os::Linux) {
  return to_system_info(detect(std::move(devices), os));
}

}  // namespace

TEST_CASE("Sonde Vulkan : chaque appareil, ses identifiants PCI, son pilote et son UUID", "[app][probe]") {
  FakeLoader loader;
  FakeVulkan fake{.devices = {rtx_4070_ti(), uhd_770()}};
  const FakeVulkanScope scope(loader, fake);

  const auto devices = probe_vulkan(loader, core::Os::Linux);
  REQUIRE(devices);
  REQUIRE(devices->size() == 2);
  const VulkanDevice& rtx = devices->front();
  CHECK(rtx.name == "NVIDIA GeForce RTX 4070 Ti");
  CHECK(rtx.vendor_id == pci::kNvidia);
  CHECK(rtx.device_id == 0x2782);
  CHECK(rtx.type == DeviceType::Discrete);
  CHECK(rtx.driver_name == "NVIDIA");
  CHECK(driver_version_text(rtx, core::Os::Linux) == "615.78.08");
  CHECK(rtx.uuid_known);
  CHECK(uuid_text(rtx.uuid) == "82838485-8687-8889-8a8b-8c8d8e8f9091");
  CHECK(rtx.vram_bytes == 12ULL << 30);
  CHECK(rtx.cooperative_matrix);

  // Vulkan 1.1 sans VK_KHR_driver_properties : pas de nom de pilote, version au format Vulkan.
  const VulkanDevice& uhd = devices->back();
  CHECK(uhd.type == DeviceType::Integrated);
  CHECK(uhd.driver_name.empty());
  CHECK(driver_version_text(uhd, core::Os::Linux) == "24.1.7");
  CHECK_FALSE(uhd.cooperative_matrix);

  // Une instance créée le temps de la sonde, puis détruite.
  CHECK(fake.created == 1);
  CHECK(fake.destroyed == 1);
}

TEST_CASE("Sonde Vulkan : refus nommés sans chargeur ou sans instance", "[app][probe]") {
  SECTION("aucun chargeur") {
    const FakeLoader empty;
    const auto devices = probe_vulkan(empty, core::Os::Linux);
    REQUIRE_FALSE(devices);
    CHECK(devices.error().code == RefusalCode::LibraryUnavailable);
    CHECK(devices.error().detail.find("libvulkan.so.1") != std::string::npos);
  }
  SECTION("aucun pilote : vkCreateInstance refuse") {
    FakeLoader loader;
    FakeVulkan fake{.create_result = VK_ERROR_INCOMPATIBLE_DRIVER};
    const FakeVulkanScope scope(loader, fake);
    const auto devices = probe_vulkan(loader, core::Os::Linux);
    REQUIRE_FALSE(devices);
    CHECK(devices.error().code == RefusalCode::VulkanCallFailed);
    CHECK(devices.error().detail.find("vkCreateInstance") != std::string::npos);
    CHECK(devices.error().detail.find("VK_ERROR_INCOMPATIBLE_DRIVER") != std::string::npos);
  }
  SECTION("chargeur Vulkan 1.0") {
    FakeLoader loader;
    FakeVulkan fake{.vulkan_1_0 = true};
    const FakeVulkanScope scope(loader, fake);
    const auto devices = probe_vulkan(loader, core::Os::Linux);
    REQUIRE_FALSE(devices);
    CHECK(devices.error().code == RefusalCode::VulkanCallFailed);
  }
  SECTION("aucun appareil : liste vide, sans refus") {
    FakeLoader loader;
    FakeVulkan fake;
    const FakeVulkanScope scope(loader, fake);
    const auto devices = probe_vulkan(loader, core::Os::Linux);
    REQUIRE(devices);
    CHECK(devices->empty());
  }
}

TEST_CASE("Sonde Vulkan : MoltenVK s'énumère avec l'extension de portabilité", "[app][probe]") {
  FakeLoader loader;
  FakeVulkan fake{.devices = {rtx_4070_ti()}, .instance_extensions = {"VK_KHR_portability_enumeration"}};
  const FakeVulkanScope scope(loader, fake);
  CHECK(probe_vulkan(loader, core::Os::MacOs)->size() == 1);
  CHECK(vulkan_loader_names(core::Os::MacOs).front() == "libvulkan.1.dylib");
  CHECK(vulkan_loader_names(core::Os::Windows).front() == "vulkan-1.dll");
}

TEST_CASE("Sonde des runtimes : version du pilote CUDA, ROCm et OpenVINO par leurs bibliothèques",
          "[app][probe]") {
  SECTION("CUDA 13.4 : TensorRT 11 possible") {
    FakeLoader loader;
    add_fake_cuda(loader, FakeCuda::Cuda13_4);
    loader.add("libopenvino.so", {});
    const RuntimeReport report = probe_runtimes(loader, core::Os::Linux);
    REQUIRE(report.cuda);
    CHECK(report.cuda->library == "libcuda.so.1");
    CHECK(report.cuda->major() == 13);
    CHECK(report.cuda->minor() == 4);
    CHECK(cuda_fits_tensorrt(report) == true);
    CHECK(report.openvino == "libopenvino.so");
    CHECK_FALSE(report.hip);
    CHECK_FALSE(report.migraphx);
  }
  SECTION("CUDA 12.8 : pilote trop ancien pour TensorRT 11") {
    FakeLoader loader;
    add_fake_cuda(loader, FakeCuda::Cuda12_8);
    const RuntimeReport report = probe_runtimes(loader, core::Os::Linux);
    CHECK(cuda_fits_tensorrt(report) == false);
  }
  SECTION("pilote cassé ou sans symbole : absent, avec une note") {
    for (const FakeCuda cuda : {FakeCuda::Broken, FakeCuda::NoSymbol}) {
      FakeLoader loader;
      add_fake_cuda(loader, cuda);
      const RuntimeReport report = probe_runtimes(loader, core::Os::Linux);
      CHECK_FALSE(report.cuda);
      CHECK(cuda_fits_tensorrt(report) == std::nullopt);
      REQUIRE(report.notes.size() == 1);
      CHECK(report.notes.front().starts_with("libcuda.so.1"));
    }
  }
  SECTION("ROCm et MIGraphX sous Windows") {
    FakeLoader loader;
    loader.add("amdhip64_6.dll", {});
    loader.add("migraphx_c.dll", {});
    const RuntimeReport report = probe_runtimes(loader, core::Os::Windows);
    CHECK(report.hip == "amdhip64_6.dll");
    CHECK(report.migraphx == "migraphx_c.dll");
    CHECK_FALSE(report.cuda);
  }
}

TEST_CASE("GPU de rendu : un dédié avant un intégré, le rendu logiciel en dernier", "[app][probe]") {
  const SystemReport report = detect({llvmpipe(), uhd_770(), gtx_1660(), rtx_4070_ti()});
  REQUIRE(render_device(report.devices) == 3);  // dédiés à égalité de type : le plus de mémoire
  const core::SystemInfo info = to_system_info(report);
  REQUIRE(info.gpus.size() == 4);
  CHECK(info.gpus.front().name == "NVIDIA GeForce RTX 4070 Ti");
  CHECK(info.gpus[1].name == "llvmpipe (LLVM 20.1.2, 256 bits)");  // puis l'ordre de l'énumération
  CHECK_FALSE(render_device({}).has_value());
}

TEST_CASE("Règles de la présélection : unités matricielles par constructeur", "[app][probe]") {
  const auto first = [](std::vector<FakeDevice> devices) { return info_of(std::move(devices)).gpus.front(); };
  CHECK(first({rtx_4070_ti()}).tensor_cores);
  CHECK_FALSE(first({gtx_1660()}).tensor_cores);
  CHECK(first({radeon_7900()}).rocm_supported);
  CHECK_FALSE(first({radeon_6800()}).rocm_supported);
  CHECK(first({arc_a770()}).openvino_gpu);
  CHECK_FALSE(first({uhd_770()}).openvino_gpu);
  CHECK(first({llvmpipe()}).vendor == core::GpuVendor::Other);

  // De la détection à la recommandation : TensorRT coché d'office sur la RTX seulement.
  const auto preselected = [](const core::SystemInfo& info, core::EngineId id) {
    const auto offers = core::recommend_engines(info);
    const auto found = std::ranges::find(offers, id, &core::EngineOffer::id);
    return found != offers.end() && found->preselected;
  };
  CHECK(preselected(info_of({rtx_4070_ti()}), core::EngineId::TensorRt));
  CHECK_FALSE(preselected(info_of({gtx_1660()}), core::EngineId::TensorRt));
  CHECK(preselected(info_of({radeon_7900()}), core::EngineId::MiGraphX));
  CHECK(preselected(info_of({arc_a770()}), core::EngineId::OpenVino));
}

TEST_CASE("Description du système, une ligne par GPU puis par runtime", "[app][probe]") {
  FakeLoader loader;
  FakeVulkan fake{.devices = {uhd_770(), rtx_4070_ti()}};
  add_fake_cuda(loader, FakeCuda::Cuda12_8);
  const FakeVulkanScope scope(loader, fake);
  const std::vector<std::string> lines = describe(detect_system(loader, core::Os::Linux));
  REQUIRE(lines.size() >= 6);
  CHECK(lines[0] == "Système : linux");
  CHECK(lines[1].starts_with("GPU 0 : Intel(R) UHD Graphics 770"));
  CHECK(lines[2].starts_with("GPU 1 (rendu) : NVIDIA GeForce RTX 4070 Ti (dédié, 12,0 Gio), PCI 10de:2782"));
  CHECK(lines[2].find("pilote NVIDIA 615.78.08") != std::string::npos);
  CHECK(lines[3] == "Pilote CUDA : 12.8 (libcuda.so.1), trop ancien pour TensorRT 11");

  const SystemReport without = detect_system(FakeLoader{}, core::Os::Linux);
  CHECK(without.vulkan_refusal.has_value());
  CHECK(describe(without)[1].starts_with("Vulkan : aucun GPU lu (LibraryUnavailable"));
}
