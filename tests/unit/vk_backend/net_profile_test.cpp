// Profil des dispatchs (net_profile.h) : un petit graphe pour le mécanisme, et le banc caché du
// vrai RIFE v4.25, qui ne tourne que sur demande (« ajp_vk_backend_tests "[.rife_profil]" »).
#include "gpu/vulkan/backend/net_profile.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/ops.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_engine.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/compute/buffer.h"
#include "net_rig.h"
#include "test_models.h"

namespace ajp::gpu::vk::test {
namespace {

// Le modèle du banc, où l'installation locale le range.
constexpr const char* kRifeModel = "/home/ta2b/ajp/models/rife/rife_v4.25.onnx";
constexpr std::uint32_t kWarmup = 5;
constexpr std::uint32_t kTimedRuns = 20;

std::string megabytes(std::uint64_t bytes) {
  return std::format("{:.2f}", static_cast<double>(bytes) / (1024.0 * 1024.0));
}

// Un tampon d'appareil rempli de float16 à 0,5 : des valeurs sans signification, mais finies.
Buffer filled_input(NetRig& rig, std::uint64_t bytes) {
  std::vector<std::byte> data(bytes);
  for (std::size_t k = 0; k + 1 < data.size(); k += 2) {
    const std::uint16_t half = 0x3800;
    std::memcpy(data.data() + k, &half, 2);
  }
  Buffer buffer = require_value(Buffer::create(*rig.context, *rig.memory, bytes, MemoryUse::Device));
  require_ok(upload(*rig.context, *rig.memory, buffer, 0, data));
  return buffer;
}

}  // namespace

TEST_CASE("Profil des dispatchs : un graphe mesuré par shader", "[vk_backend][net_profile]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  ir::Graph graph = require_value(ir::load_onnx(span_like(true).bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  const std::vector<net::KernelChoice> choices = require_value(net::default_choices(graph, rig.target));
  net::Program program = require_value(net::lower(graph, choices, rig.target));
  const std::size_t dispatches = program.dispatches.size();

  std::uint32_t counted = 0;
  for (const net::ShaderCount& row : net::static_count(graph, program)) counted += row.dispatches;
  CHECK(counted == dispatches);

  std::string logged;
  auto engine = require_value(net::NetEngine::create(
      *rig.context, *rig.memory, *rig.pipelines, graph, std::move(program),
      {.profile_dispatches = true, .log = [&](std::string_view text) { logged += text; }}));
  if (!engine->profiling()) {
    CHECK(logged.find("horodatage") != std::string::npos);  // dit pourquoi rien n'est mesuré
    SKIP("pas d'horodatage sur " << rig.context->caps().device_name);
  }
  Buffer input = filled_input(rig, engine->input_bytes(0));
  Buffer output =
      require_value(Buffer::create(*rig.context, *rig.memory, engine->output_bytes(0), MemoryUse::Device));
  const BufferView in_view = input.view();
  const BufferView out_view = output.view();
  Recording* recording = require_value(engine->recording({&in_view, 1}, {&out_view, 1}));
  for (int k = 0; k < 3; ++k) {
    require_ok(rig.context->submit(*recording));
    require_ok(rig.context->wait_on_host(recording->last_run()));
    require_ok(engine->collect_profile(*recording));
  }
  const net::ProfileReport report = engine->profile_report();
  CHECK(report.runs == 3);
  std::uint32_t profiled = 0;
  double share = 0.0;
  for (const net::ShaderTime& row : report.shaders) {
    profiled += row.dispatches;
    share += row.share;
  }
  CHECK(profiled == dispatches);
  CHECK(report.total_ms > 0.0);
  CHECK(report.span_ms >= report.total_ms * 0.99);
  CHECK(share == Catch::Approx(1.0).margin(0.05));
  CHECK(std::ranges::is_sorted(report.shaders, std::greater<>{}, &net::ShaderTime::ms));
  engine.reset();  // journalise le tableau
  CHECK(logged.find("profil des dispatchs") != std::string::npos);
}

TEST_CASE("Profil des dispatchs : désactivé, rien n'est mesuré", "[vk_backend][net_profile]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  ir::Graph graph = require_value(ir::load_onnx(span_like(true).bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  const std::vector<net::KernelChoice> choices = require_value(net::default_choices(graph, rig.target));
  net::Program program = require_value(net::lower(graph, choices, rig.target));
  auto engine = require_value(
      net::NetEngine::create(*rig.context, *rig.memory, *rig.pipelines, graph, std::move(program)));
  CHECK_FALSE(engine->profiling());
  Buffer input = filled_input(rig, engine->input_bytes(0));
  Buffer output =
      require_value(Buffer::create(*rig.context, *rig.memory, engine->output_bytes(0), MemoryUse::Device));
  const BufferView in_view = input.view();
  const BufferView out_view = output.view();
  Recording* recording = require_value(engine->recording({&in_view, 1}, {&out_view, 1}));
  require_ok(rig.context->submit(*recording));
  require_ok(rig.context->wait_on_host(recording->last_run()));
  require_ok(engine->collect_profile(*recording));
  CHECK(engine->profile_report().runs == 0);
}

TEST_CASE("Profil des dispatchs : RIFE v4.25 en 1x11x768x1280", "[.rife_profil]") {
  if (!std::filesystem::exists(kRifeModel)) SKIP("modèle absent : " << kRifeModel);
  NetRig rig = make_net_rig();
  require_f16(rig);

  const net::BuildRequest request{.onnx = kRifeModel,
                                  .model_id = "rife_v4.25",
                                  .model_sha256 = "",
                                  .input_shape = {1, 11, 768, 1280},
                                  .fp16 = true};
  ir::Graph graph = require_value(net::prepare_graph(request, rig.target));
  const std::vector<net::KernelChoice> choices = require_value(net::default_choices(graph, rig.target));
  net::Program program = require_value(net::lower(graph, choices, rig.target));

  std::cout << std::format("\n== RIFE v4.25, entrée {}, {} : {} nœuds, {} dispatchs ==\n",
                           ir::format_dims(request.input_shape), rig.context->caps().device_name,
                           graph.nodes.size(), program.dispatches.size());
  std::cout << "\n-- décompte statique (net::lower) --\n";
  std::cout << std::format("{:<40} {:>9} {:>12} {:>12}\n", "shader", "dispatchs", "lu (Mo)", "écrit (Mo)");
  std::uint64_t read = 0;
  std::uint64_t written = 0;
  for (const net::ShaderCount& row : net::static_count(graph, program)) {
    std::cout << std::format("{:<40} {:>9} {:>12} {:>12}\n", row.shader, row.dispatches,
                             megabytes(row.read_bytes), megabytes(row.written_bytes));
    read += row.read_bytes;
    written += row.written_bytes;
  }
  std::cout << std::format("{:<40} {:>9} {:>12} {:>12}\n", "total", program.dispatches.size(),
                           megabytes(read), megabytes(written));

  std::cout << "\n-- copies restantes (net_copy_f16) : nœud, opérateur, entrées -> sorties --\n";
  for (const net::Dispatch& d : program.dispatches) {
    if (d.shader != "net_copy_f16") continue;
    const ir::Node& node = graph.nodes[d.node];
    std::string in;
    for (const ir::TensorId t : node.inputs) {
      if (t == ir::kNoTensor || graph.tensors[t].kind != ir::TensorKind::Activation) continue;
      in += std::format("{}{} ", ir::format_dims(graph.tensors[t].dims),
                        graph.tensors[t].layout == ir::Layout::Nchw ? "" : "*");
    }
    std::string out;
    for (const ir::TensorId t : node.outputs)
      out += std::format("{}{} ", ir::format_dims(graph.tensors[t].dims),
                         graph.tensors[t].layout == ir::Layout::Nchw ? "" : "*");
    std::cout << std::format("{:>4} {:<10} {} -> {}\n", d.node, ir::name(node.op), in, out);
  }
  auto engine = require_value(net::NetEngine::create(
      *rig.context, *rig.memory, *rig.pipelines, std::move(graph), std::move(program),
      {.profile_dispatches = true, .log = [](std::string_view text) { std::cout << text << '\n'; }}));
  if (!engine->profiling()) SKIP("pas d'horodatage sur " << rig.context->caps().device_name);

  Buffer input = filled_input(rig, engine->input_bytes(0));
  std::vector<Buffer> outputs;
  std::vector<BufferView> out_views;
  for (std::size_t k = 0; k < engine->output_names().size(); ++k) {
    outputs.push_back(
        require_value(Buffer::create(*rig.context, *rig.memory, engine->output_bytes(k), MemoryUse::Device)));
    out_views.push_back(outputs.back().view());
  }
  const BufferView in_view = input.view();
  Recording* recording = require_value(engine->recording({&in_view, 1}, out_views));
  const auto run = [&] {
    require_ok(rig.context->submit(*recording));
    require_ok(rig.context->wait_on_host(recording->last_run()));
    require_ok(engine->collect_profile(*recording));
  };
  for (std::uint32_t k = 0; k < kWarmup; ++k) run();
  engine->clear_profile();
  for (std::uint32_t k = 0; k < kTimedRuns; ++k) run();

  const net::ProfileReport report = engine->profile_report();
  REQUIRE(report.runs == kTimedRuns);
  CHECK(report.total_ms > 0.0);
  std::cout << "\n-- temps GPU par shader (médiane de " << kTimedRuns << " exécutions) --\n"
            << net::format_profile(report) << '\n';
}

}  // namespace ajp::gpu::vk::test
