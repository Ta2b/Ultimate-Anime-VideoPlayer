#include "gpu/ir/onnx_fp8.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ajp::gpu::ir::onnx {
namespace {

bool default_domain(const Node& node) {
  return node.domain.empty() || node.domain == "ai.onnx";
}

const Tensor* find_initializer(const Graph& graph, const std::string& name) {
  const auto it = std::ranges::find(graph.initializers, name, &Tensor::name);
  return it == graph.initializers.end() ? nullptr : &*it;
}

// Zéro FP8 E4M3 scalaire valant 0 : le seul que la fusion sait absorber.
bool fp8_zero(const Graph& graph, const Node& node) {
  if (node.inputs.size() != 3) return false;
  const Tensor* zero = find_initializer(graph, node.inputs[2]);
  return zero != nullptr && zero->data_type == kFloat8E4M3Fn && zero->dims.empty() &&
         zero->floats.size() == 1 && zero->floats[0] == 0.0F;
}

// Échelle scalaire, flottante et strictement positive.
const Tensor* scalar_scale(const Graph& graph, const Node& node) {
  const Tensor* scale = find_initializer(graph, node.inputs[1]);
  if (scale == nullptr || !scale->dims.empty() || scale->floats.size() != 1 || !(scale->floats[0] > 0.0F)) {
    return nullptr;
  }
  return scale->data_type == kFloat || scale->data_type == kFloat16 ? scale : nullptr;
}

bool has_axis_or_odd_attrs(const Node& node) {
  return std::ranges::any_of(node.attributes, [](const Attribute& a) {
    return (a.name == "axis" || a.name == "block_size") && a.i != 0;
  });
}

}  // namespace

std::size_t fuse_fp8_quant_pairs(Graph& graph) {
  std::unordered_map<std::string, std::vector<std::size_t>> readers;
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    for (const std::string& name : graph.nodes[n].inputs) readers[name].push_back(n);
  }
  const std::unordered_set<std::string> graph_outputs = [&] {
    std::unordered_set<std::string> names;
    for (const ValueInfo& info : graph.outputs) names.insert(info.name);
    return names;
  }();

  std::vector<bool> removed(graph.nodes.size(), false);
  std::size_t fused = 0;
  for (std::size_t q = 0; q < graph.nodes.size(); ++q) {
    Node& quant = graph.nodes[q];
    if (!default_domain(quant) || quant.op_type != "QuantizeLinear" || quant.outputs.size() != 1) continue;
    if (quant.inputs.size() != 3 || !fp8_zero(graph, quant) || has_axis_or_odd_attrs(quant)) continue;
    const auto users = readers.find(quant.outputs[0]);
    if (users == readers.end() || users->second.size() != 1 || graph_outputs.contains(quant.outputs[0]))
      continue;
    const std::size_t d = users->second[0];
    Node& dequant = graph.nodes[d];
    if (!default_domain(dequant) || dequant.op_type != "DequantizeLinear" || dequant.outputs.size() != 1 ||
        dequant.inputs.size() != 3 || dequant.inputs[0] != quant.outputs[0] || !fp8_zero(graph, dequant) ||
        has_axis_or_odd_attrs(dequant)) {
      continue;
    }
    const Tensor* qs = scalar_scale(graph, quant);
    const Tensor* ds = scalar_scale(graph, dequant);
    if (qs == nullptr || ds == nullptr || qs->floats[0] != ds->floats[0]) continue;

    Node fake{.name = dequant.name.empty() ? quant.name : dequant.name,
              .op_type = "FakeQuantE4M3",
              .domain = "",
              .inputs = {quant.inputs[0], quant.inputs[1]},
              .outputs = {dequant.outputs[0]}};
    quant = std::move(fake);
    removed[d] = true;
    ++fused;
  }
  if (fused == 0) return 0;

  std::size_t kept = 0;
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    if (removed[n]) continue;
    if (kept != n) graph.nodes[kept] = std::move(graph.nodes[n]);
    ++kept;
  }
  graph.nodes.resize(kept);

  // Les zéros FP8 que plus aucun nœud ne lit.
  std::unordered_set<std::string> still_read;
  for (const Node& node : graph.nodes) still_read.insert(node.inputs.begin(), node.inputs.end());
  std::erase_if(graph.initializers, [&](const Tensor& t) {
    return t.data_type == kFloat8E4M3Fn && !still_read.contains(t.name);
  });
  return fused;
}

}  // namespace ajp::gpu::ir::onnx
