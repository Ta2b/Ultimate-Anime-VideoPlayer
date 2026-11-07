#include "gpu/vulkan/backend/net_gate.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <ranges>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_autotune.h"
#include "gpu/vulkan/backend/net_engine.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "gpu/vulkan/compute/buffer.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::vk::net {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Le motif d'entrée dans le type du tenseur, ramené à [0, 1] comme une image.
std::vector<std::byte> input_bytes(const ir::Tensor& tensor) {
  const std::uint64_t bytes = ir::storage_bytes(tensor);
  std::vector<std::byte> out(bytes, std::byte{0});
  if (tensor.type == ir::DataType::F16) {
    for (std::uint64_t i = 0; i < bytes / 2; ++i) {
      const std::uint16_t h = kernels::ref::float_to_half(0.5F * (tuning_input(tensor.type, i) + 1.0F));
      std::memcpy(out.data() + (2 * i), &h, 2);
    }
  } else if (tensor.type == ir::DataType::F32) {
    for (std::uint64_t i = 0; i < bytes / 4; ++i) {
      const float v = 0.5F * (tuning_input(tensor.type, i) + 1.0F);
      std::memcpy(out.data() + (4 * i), &v, 4);
    }
  }
  return out;
}

double value_at(const std::byte* data, ir::DataType type, std::uint64_t i) {
  if (type == ir::DataType::F16) {
    std::uint16_t h = 0;
    std::memcpy(&h, data + (2 * i), 2);
    return kernels::ref::half_to_float(h);
  }
  float v = 0.0F;
  std::memcpy(&v, data + (4 * i), 4);
  return v;
}

struct Outputs {
  std::vector<Buffer> buffers;
};

// Le réseau aux choix donnés, une fois, sur les entrées : ses sorties, lisibles par l'hôte.
std::expected<Outputs, Refusal> run_once(VkContext& context, MemoryAllocator& memory,
                                         PipelineCache& pipelines, const ir::Graph& graph,
                                         std::span<const KernelChoice> choices, const Target& target,
                                         std::span<const Buffer> inputs) {
  auto program = lower(graph, choices, target);
  if (!program) return std::unexpected(program.error());
  auto engine = NetEngine::create(context, memory, pipelines, graph, std::move(*program));
  if (!engine) return std::unexpected(engine.error());
  Outputs out;
  std::vector<BufferView> in_views;
  std::vector<BufferView> out_views;
  for (const Buffer& b : inputs) in_views.push_back(b.view());
  for (std::size_t k = 0; k < graph.outputs.size(); ++k) {
    auto buffer = Buffer::create(context, memory, std::max<std::uint64_t>((*engine)->output_bytes(k), 4),
                                 MemoryUse::Readback);
    if (!buffer) return std::unexpected(buffer.error());
    out_views.push_back(buffer->view());
    out.buffers.push_back(std::move(*buffer));
  }
  auto recording = (*engine)->recording(in_views, out_views);
  if (!recording) return std::unexpected(recording.error());
  if (auto ok = context.submit(**recording); !ok) return std::unexpected(ok.error());
  if (auto ok = context.wait_on_host((*recording)->last_run()); !ok) return std::unexpected(ok.error());
  return out;
}

}  // namespace

std::expected<std::vector<KernelChoice>, Refusal> reference_choices(const ir::Graph& graph,
                                                                    const Target& target) {
  std::vector<KernelChoice> out;
  out.reserve(graph.nodes.size());
  for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
    auto options = candidates(graph, k, target);
    if (!options) return std::unexpected(options.error());
    const auto direct = std::ranges::find(*options, std::string("direct"), &KernelChoice::variant);
    out.push_back(direct != options->end() ? *direct : options->front());
  }
  return out;
}

std::expected<double, Refusal> network_drift(VkContext& context, MemoryAllocator& memory,
                                             PipelineCache& pipelines, const ir::Graph& graph,
                                             std::span<const KernelChoice> tuned,
                                             std::span<const KernelChoice> reference, const Target& target) {
  std::vector<Buffer> inputs;
  for (const ir::TensorId id : graph.inputs) {
    const std::vector<std::byte> bytes = input_bytes(graph.tensors[id]);
    auto buffer = Buffer::create(context, memory, std::max<std::size_t>(bytes.size(), 4), MemoryUse::Device);
    if (!buffer) return std::unexpected(buffer.error());
    if (auto ok = upload(context, memory, *buffer, 0, bytes); !ok) return std::unexpected(ok.error());
    inputs.push_back(std::move(*buffer));
  }
  auto a = run_once(context, memory, pipelines, graph, tuned, target, inputs);
  if (!a) return std::unexpected(a.error());
  auto b = run_once(context, memory, pipelines, graph, reference, target, inputs);
  if (!b) return std::unexpected(b.error());
  double diff = 0.0;
  double energy = 0.0;
  for (std::size_t k = 0; k < graph.outputs.size(); ++k) {
    const ir::Tensor& t = graph.tensors[graph.outputs[k]];
    if (t.type != ir::DataType::F16 && t.type != ir::DataType::F32) continue;
    const std::uint64_t count = ir::storage_bytes(t) / (t.type == ir::DataType::F16 ? 2 : 4);
    const std::byte* pa = a->buffers[k].mapped();
    const std::byte* pb = b->buffers[k].mapped();
    if (pa == nullptr || pb == nullptr) return refuse(RefusalCode::InputMismatch, "sortie non lisible");
    for (std::uint64_t i = 0; i < count; ++i) {
      const double va = value_at(pa, t.type, i);
      const double vb = value_at(pb, t.type, i);
      if (!std::isfinite(va)) return std::numeric_limits<double>::infinity();
      diff += (va - vb) * (va - vb);
      energy += vb * vb;
    }
  }
  if (energy <= 0.0) return diff > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
  return std::sqrt(diff / energy);
}

std::expected<Gated, Refusal> gate_network(VkContext& context, MemoryAllocator& memory,
                                           PipelineCache& pipelines, const ir::Graph& graph,
                                           std::vector<KernelChoice> tuned, const Target& target) {
  auto reference = reference_choices(graph, target);
  if (!reference) return std::unexpected(reference.error());
  Gated gated{.choices = std::move(tuned)};
  if (gated.choices == *reference) return gated;
  auto drift = network_drift(context, memory, pipelines, graph, gated.choices, *reference, target);
  if (!drift) return std::unexpected(drift.error());
  gated.drift = *drift;
  const bool low = std::ranges::any_of(gated.choices, low_precision);
  if (*drift <= (low ? kNetworkDriftLow : kNetworkDrift)) return gated;
  // La précision réduite d'abord : coopmat_acc16 reprend la même tuile en sommes float (toujours proposée à
  // côté), les autres le noyau de référence de leur couche.
  if (low) {
    std::vector<KernelChoice> wider = gated.choices;
    for (std::size_t k = 0; k < wider.size(); ++k) {
      if (wider[k].variant == "coopmat_acc16") {
        wider[k].variant = "coopmat";
      } else if (low_precision(wider[k])) {
        wider[k] = (*reference)[k];
      }
    }
    // Un réglage que les sommes en float ne prennent pas (refus de l'abaissement) : la référence partout.
    auto again = network_drift(context, memory, pipelines, graph, wider, *reference, target);
    if (again && *again <= kNetworkDrift) {
      gated.choices = std::move(wider);
      gated.fallback = std::format(
          "précision pleine à la place de la précision réduite (écart {:.4f} puis {:.4f})", *drift, *again);
      return gated;
    }
  }
  gated.choices = std::move(*reference);
  gated.fallback = std::format("noyaux de référence partout (écart {:.4f})", *drift);
  return gated;
}

}  // namespace ajp::gpu::vk::net
