#include "app/play/engine_switch.h"

#include <array>
#include <filesystem>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/settings.h"

using namespace ajp;
using namespace ajp::app::play;

namespace {

constexpr std::size_t usage(core::Usage u) {
  return static_cast<std::size_t>(u);
}

}  // namespace

TEST_CASE("Le moteur d'une bibliothèque se lit à son nom", "[app][engine_switch]") {
  CHECK(engine_of_library("base/ajp_backend_tensorrt.so") == core::EngineId::TensorRt);
  CHECK(engine_of_library("C:/ajp/ajp_backend_vulkan.dll") == core::EngineId::Vulkan);
  CHECK(engine_of_library("build/libajp_backend_migraphx.so") == core::EngineId::MiGraphX);
  CHECK(engine_of_library("AJP.app/Contents/PlugIns/ajp_backend_coreml.so") == core::EngineId::Metal);
  CHECK_FALSE(engine_of_library("base/libvulkan.so.1").has_value());
}

TEST_CASE("L'onglet propose les moteurs présents, Vulkan d'abord, sans doublon", "[app][engine_switch]") {
  const std::array<std::filesystem::path, 4> candidates{
      "data/backends/tensorrt/10/ajp_backend_tensorrt.so", "base/ajp_backend_tensorrt.so",
      "base/ajp_backend_openvino.so", "base/ajp_backend_vulkan.so"};
  const std::vector<core::EngineId> expected{core::EngineId::Vulkan, core::EngineId::TensorRt,
                                             core::EngineId::OpenVino};
  CHECK(switchable_engines(candidates) == expected);
}

TEST_CASE("Changer de moteur le choisit, l'installe et le préfère pour SPAN et RIFE",
          "[app][engine_switch]") {
  core::Settings settings;
  settings.first_run_done = true;
  const core::Settings tensorrt = using_engine(settings, core::EngineId::TensorRt);
  CHECK(tensorrt.preferred[usage(core::Usage::Upscale)] == core::EngineId::TensorRt);
  CHECK(tensorrt.preferred[usage(core::Usage::Interpolate)] == core::EngineId::TensorRt);
  CHECK(tensorrt.preferred[usage(core::Usage::TrueHdr)] == settings.preferred[usage(core::Usage::TrueHdr)]);
  CHECK(core::validate_settings(tensorrt).has_value());

  // Revenir à Vulkan ne retire pas TensorRT : il reste proposé à l'onglet.
  const core::Settings back = using_engine(tensorrt, core::EngineId::Vulkan);
  CHECK(back.preferred[usage(core::Usage::Upscale)] == core::EngineId::Vulkan);
  CHECK(back.chosen_engines.size() == 2);
  CHECK(core::validate_settings(back).has_value());
}

TEST_CASE("La bibliothèque du moteur préféré s'essaie en premier", "[app][engine_switch]") {
  const std::vector<std::filesystem::path> candidates{
      "base/ajp_backend_tensorrt.so", "base/ajp_backend_migraphx.so", "base/ajp_backend_vulkan.so"};
  core::Settings settings = using_engine(core::Settings{}, core::EngineId::Vulkan);
  const auto vulkan = preferred_first(candidates, settings);
  REQUIRE(vulkan.size() == 3);
  CHECK(vulkan[0].filename() == "ajp_backend_vulkan.so");
  CHECK(vulkan[1].filename() == "ajp_backend_tensorrt.so");
  CHECK(vulkan[2].filename() == "ajp_backend_migraphx.so");

  settings = using_engine(settings, core::EngineId::TensorRt);
  CHECK(preferred_first(candidates, settings) == candidates);
}
