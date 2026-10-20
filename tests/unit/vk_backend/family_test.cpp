// Familles de GPU par identifiants PCI, liste d'autorisation des unités matricielles, et leur
// sélection par l'abaissement : la variante coopmat n'est proposée que là où elle est permise
// (jamais sur lavapipe, la CI), et vient en tête là où elle l'est (RTX).
#include "gpu/vulkan/backend/family.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "core/gpu_device.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/device_profile.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "net_rig.h"

using namespace ajp;
using namespace ajp::gpu;
using namespace ajp::gpu::vk;
using namespace ajp::gpu::vk::test;

TEST_CASE("Familles : identifiants PCI connus", "[vk_backend][family]") {
  CHECK(detect_family(pci::kNvidia, 0x2782, false) == GpuFamily::NvidiaAda);        // RTX 4070 Ti
  CHECK(detect_family(pci::kNvidia, 0x2684, false) == GpuFamily::NvidiaAda);        // RTX 4090
  CHECK(detect_family(pci::kNvidia, 0x2206, false) == GpuFamily::NvidiaAmpere);     // RTX 3080
  CHECK(detect_family(pci::kNvidia, 0x1E87, false) == GpuFamily::NvidiaTuring);     // RTX 2080
  CHECK(detect_family(pci::kNvidia, 0x2184, false) == GpuFamily::NvidiaTuring);     // GTX 1660
  CHECK(detect_family(pci::kNvidia, 0x2B85, false) == GpuFamily::NvidiaBlackwell);  // RTX 5090
  CHECK(detect_family(pci::kNvidia, 0x1B80, false) == GpuFamily::Unknown);          // Pascal : sans unités
  CHECK(detect_family(pci::kAmd, 0x744C, false) == GpuFamily::AmdRdna3);            // RX 7900 XTX
  CHECK(detect_family(pci::kAmd, 0x7550, false) == GpuFamily::AmdRdna4);            // RX 9070
  CHECK(detect_family(pci::kAmd, 0x73BF, false) == GpuFamily::AmdBeforeRdna3);      // RX 6800
  CHECK(detect_family(pci::kIntel, 0x56A0, false) == GpuFamily::IntelAlchemist);    // Arc A770
  CHECK(detect_family(pci::kIntel, 0xE20B, false) == GpuFamily::IntelXe2);          // Arc B580
  CHECK(detect_family(pci::kNvidia, 0x2782, true) == GpuFamily::Software);
  CHECK(detect_family(0x10005, 0, false) == GpuFamily::Unknown);
}

TEST_CASE("Familles : liste d'autorisation des unités matricielles", "[vk_backend][family]") {
  CHECK(driver_kind("NVIDIA") == DriverKind::NvidiaProprietary);
  CHECK(driver_kind("radv") == DriverKind::Radv);
  CHECK(driver_kind("llvmpipe") == DriverKind::Other);
  CHECK(matrix_units_allowed(GpuFamily::NvidiaAda, DriverKind::NvidiaProprietary));
  CHECK(matrix_units_allowed(GpuFamily::NvidiaTuring, DriverKind::NvidiaProprietary));
  CHECK_FALSE(matrix_units_allowed(GpuFamily::NvidiaAda, DriverKind::Other));  // nouveau/NVK : à qualifier
  CHECK_FALSE(matrix_units_allowed(GpuFamily::AmdRdna3, DriverKind::Radv));
  CHECK_FALSE(matrix_units_allowed(GpuFamily::Software, DriverKind::Other));
  CHECK(net::preset_tables().empty());
}

TEST_CASE("Familles : variante matricielle choisie ou écartée selon l'appareil", "[vk_backend][family]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  const DeviceProfile profile = profile_device(*rig.context);
  INFO(rig.context->caps().device_name << " : famille " << name(profile.family));
  ir::test::ModelBuilder m;
  m.input("x", {1, 16, 12, 12}, ir::onnx::kFloat16);
  m.initializer(ir::test::tensor_f16("w", {16, 16, 3, 3}, std::vector<float>(16 * 16 * 9, 0.01F)));
  m.node("Relu", {"x"}, {"r"});
  m.node("Conv", {"r", "w"}, {"c"}, {ir::test::attr_ints("pads", {1, 1, 1, 1})});
  m.node("Relu", {"c"}, {"y"}).output("y", ir::onnx::kFloat16);
  ir::Graph graph = require_value(ir::load_onnx(m.bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  const auto conv = std::ranges::find(graph.nodes, ir::OpKind::Conv, &ir::Node::op) - graph.nodes.begin();
  const auto options = require_value(net::candidates(graph, static_cast<std::size_t>(conv), profile.target));
  const bool any_matrix =
      std::ranges::any_of(options, [](const net::KernelChoice& c) { return c.variant == "coopmat"; });
  if (profile.family == GpuFamily::Software) CHECK_FALSE(profile.target.matrix_units);
  CHECK(any_matrix == profile.target.matrix_units);
  if (profile.target.matrix_units) CHECK(options.front().variant == "coopmat");
  if (profile.family == GpuFamily::NvidiaAda && profile.driver == DriverKind::NvidiaProprietary) {
    CHECK(profile.matrix_hardware);
    CHECK(profile.target.matrix_units);
  }
}

TEST_CASE("Autotuner : mesure, cache par appareil et relecture", "[vk_backend][autotune]") {
  NetRig rig = make_net_rig();
  const DeviceProfile profile = profile_device(*rig.context);
  ir::test::ModelBuilder m;
  m.input("x", {1, 8, 16, 16});
  m.weight("w", {16, 8, 3, 3}, std::vector<float>(16 * 8 * 9, 0.02F));
  m.node("Conv", {"x", "w"}, {"y"}, {ir::test::attr_ints("pads", {1, 1, 1, 1})}).output("y");
  ir::Graph graph = require_value(ir::load_onnx(m.bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  const auto options = require_value(net::candidates(graph, 0, profile.target));
  REQUIRE(options.size() > 1);
  const std::filesystem::path file =
      std::filesystem::temp_directory_path() /
      std::format("ajp-autotune-{}.tsv", std::chrono::steady_clock::now().time_since_epoch().count());
  const net::TunerOptions quick{.warmup = 1, .passes = 3, .repeats = 2};
  net::KernelChoice best;
  {
    net::Autotuner tuner(*rig.context, *rig.memory, *rig.pipelines, profile, file, quick);
    const net::Measured one = require_value(tuner.measure(graph, 0, options.front()));
    CHECK(one.median_us > 0.0);
    best = require_value(tuner.choose(graph, 0, options, "test"));
    CHECK(tuner.measured() == 1);
    CHECK(std::ranges::find(options, best) != options.end());
    (void)require_value(tuner.choose(graph, 0, options, "test"));
    CHECK(tuner.cached() == 1);
    require_ok(tuner.save());
  }
  net::Autotuner reread(*rig.context, *rig.memory, *rig.pipelines, profile, file, quick);
  CHECK(require_value(reread.choose(graph, 0, options, "test")) == best);
  CHECK(reread.measured() == 0);
  // Une autre résolution ou un autre modèle se mesure à part.
  (void)require_value(reread.choose(graph, 0, options, "autre"));
  CHECK(reread.measured() == 1);
  std::filesystem::remove(file);
}
