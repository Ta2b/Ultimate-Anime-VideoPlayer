#include "gpu/ir/describe.h"

#include <algorithm>
#include <format>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"

namespace ajp::gpu::ir {
namespace {

std::string_view activation_name(Activation act) {
  switch (act) {
    case Activation::None:
      return "";
    case Activation::Relu:
      return "Relu";
    case Activation::LeakyRelu:
      return "LeakyRelu";
    case Activation::PRelu:
      return "PRelu";
    case Activation::Sigmoid:
      return "Sigmoid";
    case Activation::Tanh:
      return "Tanh";
    case Activation::Clip:
      return "Clip";
  }
  return "?";
}

std::string tensor_line(const Tensor& tensor) {
  // Type et forme d'une activation n'existent qu'après infer_shapes.
  if (tensor.kind == TensorKind::Activation && !tensor.known)
    return std::format("{} (à inférer)", tensor.name);
  return std::format("{} {} {}{}", tensor.name, name(tensor.type), format_dims(tensor.dims),
                     tensor.layout == Layout::Nhwc16 ? " NHWC16" : "");
}

// Paramètres saillants d'un nœud : noyau et épilogue d'une Conv, bloc d'un réarrangement…
std::string node_params(const Graph& graph, const Node& node) {
  if (const auto* conv = std::get_if<ConvAttrs>(&node.attrs)) {
    const Dims& w = graph.tensors[node.inputs[kConvW]].dims;  // [M, C/g, kH, kW] après inférence
    std::string out = w.size() == 4 ? std::format("k{}x{} ", w[2], w[3]) : std::string{};
    out += std::format("s{}x{} g{}", conv->strides[0], conv->strides[1], conv->group);
    if (node.inputs.size() > kConvB && node.inputs[kConvB] != kNoTensor) out += " +biais";
    if (conv->epilogue.residual) out += " +résiduel";
    if (conv->epilogue.act != Activation::None)
      out += std::format(" +{}", activation_name(conv->epilogue.act));
    return out;
  }
  if (const auto* block = std::get_if<BlockAttrs>(&node.attrs)) {
    return std::format("{} b{}", block->mode == BlockMode::Crd ? "CRD" : "DCR", block->blocksize);
  }
  if (const auto* grid = std::get_if<GridSampleAttrs>(&node.attrs)) {
    return std::format("{} {}{}{}", grid->mode == GridMode::Bilinear ? "bilinéaire" : "proche",
                       grid->padding == GridPadding::Border ? "border" : "zeros",
                       grid->align_corners ? " align_corners" : "", grid->grid_nchw ? " grille NCHW" : "");
  }
  if (const auto* resize = std::get_if<ResizeAttrs>(&node.attrs)) {
    return resize->mode == ResizeMode::Linear ? "linéaire" : "proche";
  }
  return "";
}

}  // namespace

std::string format_bytes(std::uint64_t bytes) {
  constexpr double kMib = 1024.0 * 1024.0;
  if (bytes < std::uint64_t{1024} * 1024)
    return std::format("{:.1f} Kio", static_cast<double>(bytes) / 1024.0);
  return std::format("{:.1f} Mio", static_cast<double>(bytes) / kMib);
}

std::string describe_ops(const Graph& graph) {
  std::map<std::string_view, int> counts;
  for (const Node& node : graph.nodes) ++counts[name(node.op)];
  std::vector<std::pair<std::string_view, int>> sorted(counts.begin(), counts.end());
  std::ranges::stable_sort(sorted, [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string out;
  for (const auto& [op, count] : sorted) out += std::format("{}{} {}", out.empty() ? "" : ", ", op, count);
  return out.empty() ? "aucun" : out;
}

std::string describe_interface(const Graph& graph) {
  std::string out;
  for (const TensorId id : graph.inputs) out += std::format("  entrée  {}\n", tensor_line(graph.tensors[id]));
  for (const TensorId id : graph.outputs)
    out += std::format("  sortie  {}\n", tensor_line(graph.tensors[id]));
  return out;
}

std::string describe_nodes(const Graph& graph) {
  std::string out;
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    const Node& node = graph.nodes[n];
    std::string outputs;
    for (const TensorId id : node.outputs) {
      outputs += std::format("{}{}", outputs.empty() ? "" : " ; ", tensor_line(graph.tensors[id]));
    }
    out += std::format("  {:>4}  {:<13} {:<22} -> {}\n", n, name(node.op), node_params(graph, node), outputs);
  }
  return out;
}

std::string describe_stats(const PassStats& stats) {
  return std::format(
      "  pliés {}, morts {}, sans effet {}, réarrangements {}, biais {}, résiduels {}, activations {}, "
      "tenseurs NHWC16 {}\n",
      stats.folded, stats.dead, stats.identities, stats.pixel_shuffles, stats.bias, stats.residuals,
      stats.activations, stats.nhwc);
}

std::string describe_memory(const MemoryPlan& plan) {
  return std::format(
      "  arène {} pour {} tampons (sans réutilisation : {})\n  constantes {}\n  entrées et sorties {}\n",
      format_bytes(plan.arena_bytes), plan.allocations.size(), format_bytes(plan.unshared_bytes),
      format_bytes(plan.constant_bytes), format_bytes(plan.io_bytes));
}

}  // namespace ajp::gpu::ir
