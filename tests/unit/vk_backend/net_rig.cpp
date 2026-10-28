#include "net_rig.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/device_profile.h"
#include "gpu/vulkan/backend/net_engine.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/compute/buffer.h"
#include "kernels/ref/half.h"
#include "vulkan_validation.h"

namespace ajp::gpu::vk::test {
namespace {

std::vector<std::byte> to_bytes(const ir::Value& value, ir::DataType type) {
  std::vector<std::byte> out(value.f.size() * (type == ir::DataType::F16 ? 2 : 4));
  for (std::size_t k = 0; k < value.f.size(); ++k) {
    if (type == ir::DataType::F16) {
      const std::uint16_t bits = kernels::ref::float_to_half(value.f[k]);
      std::memcpy(out.data() + (2 * k), &bits, 2);
    } else {
      std::memcpy(out.data() + (4 * k), &value.f[k], 4);
    }
  }
  return out;
}

ir::Value from_bytes(std::span<const std::byte> bytes, const ir::Tensor& tensor) {
  ir::Value v{.type = tensor.type, .dims = tensor.dims};
  const std::size_t count = ir::element_count(tensor.dims);
  v.f.resize(count);
  for (std::size_t k = 0; k < count; ++k) {
    if (tensor.type == ir::DataType::F16) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, bytes.data() + (2 * k), 2);
      v.f[k] = kernels::ref::half_to_float(bits);
    } else {
      std::memcpy(&v.f[k], bytes.data() + (4 * k), 4);
    }
  }
  return v;
}

}  // namespace

std::string describe(const Refusal& refusal) {
  return std::format("{} : {}", name(refusal.code), refusal.detail);
}

NetRig make_net_rig() {
  auto context = ajp::test::with_validation([](bool validation, ajp::test::ValidationSink sink) {
    return VkContext::create(
        {.validation = validation, .validation_sink = std::move(sink), .cooperative_matrix = true});
  });
  if (!context) {
    const RefusalCode code = context.error().code;
    if (code == RefusalCode::LibraryUnavailable || code == RefusalCode::DeviceUnavailable) {
      SKIP("aucun Vulkan : " << describe(context.error()));
    }
    FAIL(describe(context.error()));
  }
  NetRig rig;
  rig.context = std::move(*context);
  rig.memory = require_value(MemoryAllocator::create(*rig.context));
  rig.pipelines = require_value(PipelineCache::create(*rig.context));
  rig.target = profile_device(*rig.context).target;
  return rig;
}

void require_f16(const NetRig& rig) {
  if (!rig.target.f16_storage) SKIP("pas de stockage 16 bits sur " << rig.context->caps().device_name);
}

std::vector<ir::Value> run_on_device(NetRig& rig, const ir::Graph& graph,
                                     std::span<const ir::NamedValue> inputs,
                                     std::span<const net::KernelChoice> choices) {
  std::vector<net::KernelChoice> chosen(choices.begin(), choices.end());
  if (chosen.empty()) chosen = require_value(net::default_choices(graph, rig.target));
  net::Program program = require_value(net::lower(graph, chosen, rig.target));
  auto engine = require_value(
      net::NetEngine::create(*rig.context, *rig.memory, *rig.pipelines, graph, std::move(program)));

  std::vector<Buffer> in_buffers;
  std::vector<BufferView> in_views;
  for (std::size_t k = 0; k < graph.inputs.size(); ++k) {
    const ir::Tensor& t = graph.tensors[graph.inputs[k]];
    const auto found = std::ranges::find(inputs, t.name, &ir::NamedValue::name);
    REQUIRE(found != inputs.end());
    const std::vector<std::byte> bytes = to_bytes(found->value, t.type);
    Buffer buffer = require_value(Buffer::create(*rig.context, *rig.memory, bytes.size(), MemoryUse::Device));
    require_ok(upload(*rig.context, *rig.memory, buffer, 0, bytes));
    in_views.push_back(buffer.view());
    in_buffers.push_back(std::move(buffer));
  }
  std::vector<Buffer> out_buffers;
  std::vector<BufferView> out_views;
  for (std::size_t k = 0; k < graph.outputs.size(); ++k) {
    Buffer buffer =
        require_value(Buffer::create(*rig.context, *rig.memory, engine->output_bytes(k), MemoryUse::Device));
    out_views.push_back(buffer.view());
    out_buffers.push_back(std::move(buffer));
  }
  Recording* recording = require_value(engine->recording(in_views, out_views));
  require_ok(rig.context->submit(*recording));
  require_ok(rig.context->wait_on_host(recording->last_run()));

  std::vector<ir::Value> out;
  for (std::size_t k = 0; k < graph.outputs.size(); ++k) {
    std::vector<std::byte> bytes(engine->output_bytes(k));
    require_ok(download(*rig.context, *rig.memory, out_buffers[k], 0, bytes));
    out.push_back(from_bytes(bytes, graph.tensors[graph.outputs[k]]));
  }
  return out;
}

std::vector<ir::Value> run_reference(const ir::Graph& graph, std::span<const ir::NamedValue> inputs) {
  std::vector<ir::NamedValue> outputs = require_value(ir::run(graph, inputs));
  std::vector<ir::Value> out;
  for (ir::NamedValue& o : outputs) out.push_back(std::move(o.value));
  return out;
}

void require_close(const ir::Value& got, const ir::Value& want, Tolerance tolerance,
                   const std::string& what) {
  REQUIRE(got.f.size() == want.f.size());
  std::size_t worst = 0;
  float worst_excess = -1.0F;
  std::size_t failures = 0;
  for (std::size_t k = 0; k < got.f.size(); ++k) {
    const float bound = tolerance.atol + (tolerance.rtol * std::fabs(want.f[k]));
    const float error = std::fabs(got.f[k] - want.f[k]);
    const bool bad = !(error <= bound) && !(std::isnan(got.f[k]) && std::isnan(want.f[k]));
    if (bad) ++failures;
    if (error - bound > worst_excess || (bad && failures == 1)) {
      worst_excess = error - bound;
      worst = k;
    }
  }
  INFO(std::format("{} : {} écarts sur {}, pire à {} : {} au lieu de {}", what, failures, got.f.size(), worst,
                   got.f.empty() ? 0.0F : got.f[worst], want.f.empty() ? 0.0F : want.f[worst]));
  CHECK(failures == 0);
}

void check_graph(NetRig& rig, const ir::Graph& graph, std::span<const ir::NamedValue> inputs,
                 Tolerance tolerance, std::span<const net::KernelChoice> choices) {
  const std::vector<ir::Value> want = run_reference(graph, inputs);
  const std::vector<ir::Value> got = run_on_device(rig, graph, inputs, choices);
  REQUIRE(got.size() == want.size());
  for (std::size_t k = 0; k < got.size(); ++k) {
    require_close(got[k], want[k], tolerance, graph.tensors[graph.outputs[k]].name);
  }
}

ir::Value test_values(ir::DataType type, const ir::Dims& dims, std::size_t seed, float lo, float hi) {
  ir::Value v{.type = type, .dims = dims};
  const std::size_t count = ir::element_count(dims);
  v.f.resize(count);
  // Partie fractionnaire de (k + graine) × φ : une suite déterministe et bien répartie.
  constexpr double kPhi = 0.6180339887498949;
  for (std::size_t k = 0; k < count; ++k) {
    const double u = std::fmod(static_cast<double>(k + (seed * 7919)) * kPhi, 1.0);
    float x = lo + static_cast<float>(u) * (hi - lo);
    if (type == ir::DataType::F16) x = kernels::ref::round_to_half(x);
    v.f[k] = x;
  }
  return v;
}

}  // namespace ajp::gpu::vk::test
