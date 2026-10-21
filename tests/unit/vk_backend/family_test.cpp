// Familles de GPU par identifiants PCI, liste d'autorisation des unités matricielles, et leur
// sélection par l'abaissement : la variante coopmat n'est proposée que là où elle est permise
// (jamais sur lavapipe, la CI), et vient en tête là où elle l'est (RTX).
#include "gpu/vulkan/backend/family.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "core/gpu_device.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/device_profile.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "kernels/ref/half.h"
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
  CHECK(detect_family(pci::kIntel, 0x7D55, false) == GpuFamily::IntelXeLpg);  // Arc du Core Ultra 7 155H
  CHECK(detect_family(pci::kIntel, 0xA780, false) == GpuFamily::IntelXeLp);   // UHD 770
  CHECK(detect_family(pci::kIntel, 0xE20B, false) == GpuFamily::IntelXe2);    // Arc B580
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
  CHECK(matrix_units_allowed(GpuFamily::AmdRdna3, DriverKind::Radv));
  CHECK(matrix_units_allowed(GpuFamily::AmdRdna4, DriverKind::AmdProprietary));
  CHECK_FALSE(matrix_units_allowed(GpuFamily::AmdBeforeRdna3, DriverKind::AmdProprietary));
  CHECK_FALSE(matrix_units_allowed(GpuFamily::AmdRdna4, DriverKind::Other));
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
  // Une autre version du même pilote : rien de remesuré, la sortie du choix revérifiée une fois.
  DeviceProfile updated = profile;
  updated.driver_build += "+1";
  updated.key = updated.stable_key + ":" + updated.driver_build;
  {
    net::Autotuner after(*rig.context, *rig.memory, *rig.pipelines, updated, file, quick);
    CHECK(require_value(after.choose(graph, 0, options, "test")) == best);
    CHECK(after.measured() == 0);
    CHECK(after.reverified() == 1);
    CHECK(after.cached() == 1);
    require_ok(after.save());
  }
  net::Autotuner settled(*rig.context, *rig.memory, *rig.pipelines, updated, file, quick);
  CHECK(require_value(settled.choose(graph, 0, options, "test")) == best);
  CHECK(settled.reverified() == 0);
  CHECK(settled.cached() == 1);
  // Un autre pilote (ou une autre carte) se mesure à part.
  DeviceProfile other = profile;
  other.stable_key += "-autre";
  net::Autotuner elsewhere(*rig.context, *rig.memory, *rig.pipelines, other, file, quick);
  (void)require_value(elsewhere.choose(graph, 0, options, "test"));
  CHECK(elsewhere.measured() == 1);
  std::filesystem::remove(file);

  // Le sondage : tout est mesuré en entier sous un facteur immense, seul le plus rapide sous 1.
  net::Autotuner wide(*rig.context, *rig.memory, *rig.pipelines, profile, {},
                      {.warmup = 1, .passes = 2, .repeats = 1, .keep_within = 1e9});
  (void)require_value(wide.choose(graph, 0, options, "test"));
  CHECK(wide.pruned() == 0);
  net::Autotuner narrow(*rig.context, *rig.memory, *rig.pipelines, profile, {},
                        {.warmup = 1, .passes = 2, .repeats = 1, .keep_within = 1.0});
  CHECK(std::ranges::find(options, require_value(narrow.choose(graph, 0, options, "test"))) != options.end());
  CHECK(narrow.pruned() >= 1);
  CHECK(narrow.pruned() + narrow.rejected() < options.size());
}

TEST_CASE("Autotuner : motif d'entrée pseudo-aléatoire et périodique", "[vk_backend][autotune]") {
  for (const auto& [type, period] : {std::pair{ir::DataType::F16, net::kPatternPeriodF16},
                                     std::pair{ir::DataType::F32, net::kPatternPeriodF32}}) {
    std::set<float> distinct;
    double mean = 0.0;
    for (std::uint64_t i = 0; i < period; ++i) {
      const float v = net::tuning_input(type, i);
      REQUIRE(v >= -1.0F);
      REQUIRE(v <= 1.0F);
      CHECK(net::tuning_input(type, i + period) == v);
      distinct.insert(v);
      mean += v;
    }
    mean /= period;
    CHECK(std::abs(mean) < 0.02);
    CHECK(distinct.size() > period / 4);
    // Un voisin, une ligne de 1920, un plan de 1920 × 1080 ne retombent pas sur la même valeur.
    std::size_t same = 0;
    for (std::uint64_t i = 0; i < 4096; ++i) {
      const float v = net::tuning_input(type, i);
      same += static_cast<std::size_t>(v == net::tuning_input(type, i + 1)) +
              static_cast<std::size_t>(v == net::tuning_input(type, i + 1920)) +
              static_cast<std::size_t>(v == net::tuning_input(type, i + (1920 * 1080)));
    }
    CHECK(same < 64);
  }
  const float h = net::tuning_input(ir::DataType::F16, 7);
  CHECK(kernels::ref::half_to_float(kernels::ref::float_to_half(h)) == h);
  CHECK(net::tuning_input(ir::DataType::I64, 3) == 0.0F);
}

// Chaque candidat d'une Conv 3 × 3 sur les entrées de tuning_input, contre le même calcul sur
// l'hôte : le motif couvre bien chaque élément, période après période (les entrées dépassent la
// période du type), et un noyau qui se tromperait de voisin y serait visible.
TEST_CASE("Autotuner : chaque candidat contre l'hôte, entrées pseudo-aléatoires", "[vk_backend][autotune]") {
  NetRig rig = make_net_rig();
  const bool f16 = GENERATE(false, true);
  if (f16) require_f16(rig);
  const DeviceProfile profile = profile_device(*rig.context);
  constexpr std::int64_t kIn = 3;
  constexpr std::int64_t kOut = 4;
  constexpr std::int64_t kH = 72;
  constexpr std::int64_t kW = 88;  // 3 × 72 × 88 = 19 008 : au-delà des deux périodes
  std::vector<float> weights(kOut * kIn * 9);
  for (std::size_t k = 0; k < weights.size(); ++k)
    weights[k] = static_cast<float>(static_cast<int>((k * 7) % 11) - 5) * 0.0625F;
  ir::test::ModelBuilder m;
  m.input("x", {1, kIn, kH, kW}, f16 ? ir::onnx::kFloat16 : ir::onnx::kFloat);
  if (f16) {
    m.initializer(ir::test::tensor_f16("w", {kOut, kIn, 3, 3}, weights));
  } else {
    m.weight("w", {kOut, kIn, 3, 3}, weights);
  }
  m.node("Conv", {"x", "w"}, {"y"}, {ir::test::attr_ints("pads", {1, 1, 1, 1})})
      .output("y", f16 ? ir::onnx::kFloat16 : ir::onnx::kFloat);
  ir::Graph graph = require_value(ir::load_onnx(m.bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  REQUIRE(graph.nodes.size() == 1);
  const ir::DataType type = f16 ? ir::DataType::F16 : ir::DataType::F32;
  std::vector<float> expected(static_cast<std::size_t>(kOut * kH * kW), 0.0F);
  float peak = 0.0F;
  for (std::int64_t o = 0; o < kOut; ++o) {
    for (std::int64_t y = 0; y < kH; ++y) {
      for (std::int64_t x = 0; x < kW; ++x) {
        double sum = 0.0;
        for (std::int64_t c = 0; c < kIn; ++c) {
          for (std::int64_t dy = -1; dy <= 1; ++dy) {
            for (std::int64_t dx = -1; dx <= 1; ++dx) {
              if (y + dy < 0 || y + dy >= kH || x + dx < 0 || x + dx >= kW) continue;
              const auto index = static_cast<std::uint64_t>((((c * kH) + y + dy) * kW) + x + dx);
              sum += static_cast<double>(
                         weights[static_cast<std::size_t>((((o * kIn) + c) * 9) + ((dy + 1) * 3) + dx + 1)]) *
                     net::tuning_input(type, index);
            }
          }
        }
        expected[static_cast<std::size_t>((((o * kH) + y) * kW) + x)] = static_cast<float>(sum);
        peak = std::max(peak, std::abs(static_cast<float>(sum)));
      }
    }
  }
  // Une sortie qui varie d'un pixel à l'autre : un motif constant la rendrait presque uniforme.
  CHECK(std::set<float>(expected.begin(), expected.end()).size() > expected.size() / 2);
  const auto options = require_value(net::candidates(graph, 0, profile.target));
  REQUIRE_FALSE(options.empty());
  net::Autotuner tuner(*rig.context, *rig.memory, *rig.pipelines, profile, {},
                       {.warmup = 0, .passes = 1, .repeats = 1});
  const float tolerance = f16 ? 4e-3F : 1e-5F;
  for (const net::KernelChoice& choice : options) {
    INFO(choice.variant);
    const net::Measured measured = require_value(tuner.measure(graph, 0, choice));
    REQUIRE(measured.output.size() == expected.size());
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const float slack = (tolerance * peak) + (tolerance * std::abs(expected[i]));
      wrong += static_cast<std::size_t>(!(std::abs(measured.output[i] - expected[i]) <= slack));
    }
    CHECK(wrong == 0);
    CHECK(measured.median_us > 0.0);
  }
}
