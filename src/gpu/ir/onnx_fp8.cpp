#include "gpu/ir/onnx_fp8.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gpu/ir/fp8.h"

namespace ajp::gpu::ir::onnx {
namespace {

bool default_domain(const Node& node) {
  return node.domain.empty() || node.domain == "ai.onnx";
}

const Tensor* find_initializer(const Graph& graph, const std::string& name) {
  const auto it = std::ranges::find(graph.initializers, name, &Tensor::name);
  return it == graph.initializers.end() ? nullptr : &*it;
}

// Zéros FP8 E4M3 tous nuls (scalaire ou un par canal) : les seuls que la fusion sait absorber.
bool fp8_zero(const Graph& graph, const Node& node) {
  if (node.inputs.size() != 3) return false;
  const Tensor* zero = find_initializer(graph, node.inputs[2]);
  return zero != nullptr && zero->data_type == kFloat8E4M3Fn &&
         std::ranges::all_of(zero->floats, [](float v) { return v == 0.0F; });
}

// Échelles flottantes strictement positives, scalaire ou une par canal.
const Tensor* positive_scale(const Graph& graph, const Node& node) {
  const Tensor* scale = find_initializer(graph, node.inputs[1]);
  if (scale == nullptr || scale->dims.size() > 1 || scale->floats.empty() ||
      !std::ranges::all_of(scale->floats, [](float v) { return v > 0.0F; })) {
    return nullptr;
  }
  return scale->data_type == kFloat || scale->data_type == kFloat16 ? scale : nullptr;
}

// Axe de quantification (1 par défaut, ONNX) ; nullopt si block_size ou un axe hors rang.
std::optional<std::int64_t> quant_axis(const Node& node) {
  std::int64_t axis = 1;
  for (const Attribute& a : node.attributes) {
    if (a.name == "block_size" && a.i != 0) return std::nullopt;
    if (a.name == "axis") axis = a.i;
  }
  return axis;
}

// Poids constants quantifiés par canal : x_k arrondi à la grille E4M3 de l'échelle de son canal.
std::optional<Tensor> fold_per_channel(const Tensor& weights, const Tensor& scale, std::int64_t axis,
                                       std::string name) {
  const auto rank = static_cast<std::int64_t>(weights.dims.size());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank || weights.floats.empty() ||
      weights.dims[static_cast<std::size_t>(axis)] != static_cast<std::int64_t>(scale.floats.size()) ||
      (weights.data_type != kFloat && weights.data_type != kFloat16)) {
    return std::nullopt;
  }
  std::size_t inner = 1;
  for (std::int64_t d = axis + 1; d < rank; ++d)
    inner *= static_cast<std::size_t>(weights.dims[static_cast<std::size_t>(d)]);
  Tensor out = weights;
  out.name = std::move(name);
  for (std::size_t k = 0; k < out.floats.size(); ++k) {
    const float s = scale.floats[(k / inner) % scale.floats.size()];
    out.floats[k] = round_to_e4m3(out.floats[k] / s) * s;
  }
  return out;
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
  std::vector<Tensor> folded;
  std::size_t fused = 0;
  for (std::size_t q = 0; q < graph.nodes.size(); ++q) {
    Node& quant = graph.nodes[q];
    if (!default_domain(quant) || quant.op_type != "QuantizeLinear" || quant.outputs.size() != 1) continue;
    if (quant.inputs.size() != 3 || !fp8_zero(graph, quant)) continue;
    const auto users = readers.find(quant.outputs[0]);
    if (users == readers.end() || users->second.size() != 1 || graph_outputs.contains(quant.outputs[0]))
      continue;
    const std::size_t d = users->second[0];
    Node& dequant = graph.nodes[d];
    if (!default_domain(dequant) || dequant.op_type != "DequantizeLinear" || dequant.outputs.size() != 1 ||
        dequant.inputs.size() != 3 || dequant.inputs[0] != quant.outputs[0] || !fp8_zero(graph, dequant)) {
      continue;
    }
    const Tensor* qs = positive_scale(graph, quant);
    const Tensor* ds = positive_scale(graph, dequant);
    const auto q_axis = quant_axis(quant);
    const auto d_axis = quant_axis(dequant);
    if (qs == nullptr || ds == nullptr || !q_axis || !d_axis || qs->floats != ds->floats) continue;

    if (qs->dims.empty()) {
      Node fake{.name = dequant.name.empty() ? quant.name : dequant.name,
                .op_type = "FakeQuantE4M3",
                .domain = "",
                .inputs = {quant.inputs[0], quant.inputs[1]},
                .outputs = {dequant.outputs[0]}};
      quant = std::move(fake);
      removed[d] = true;
    } else {
      // Par canal : seuls des poids constants se plient, en un initialiseur arrondi.
      const Tensor* weights = find_initializer(graph, quant.inputs[0]);
      if (weights == nullptr || *q_axis != *d_axis) continue;
      auto rounded = fold_per_channel(*weights, *qs, *q_axis, dequant.outputs[0]);
      if (!rounded) continue;
      folded.push_back(std::move(*rounded));
      removed[q] = true;
      removed[d] = true;
    }
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
  for (Tensor& t : folded) graph.initializers.push_back(std::move(t));

  // Les zéros FP8 que plus aucun nœud ne lit.
  std::unordered_set<std::string> still_read;
  for (const Node& node : graph.nodes) still_read.insert(node.inputs.begin(), node.inputs.end());
  std::erase_if(graph.initializers, [&](const Tensor& t) {
    return t.data_type == kFloat8E4M3Fn && !still_read.contains(t.name);
  });
  return fused;
}

}  // namespace ajp::gpu::ir::onnx
