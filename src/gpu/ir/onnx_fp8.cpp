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
#include "gpu/ir/int8.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::ir::onnx {
namespace {

const Tensor* find_initializer(const Graph& graph, const std::string& name) {
  const auto it = std::ranges::find(graph.initializers, name, &Tensor::name);
  return it == graph.initializers.end() ? nullptr : &*it;
}

// La grille de zéros tous nuls (scalaire ou un par canal), FP8 E4M3 ou INT8 : les seuls que la fusion sait
// absorber. Sans zéro, l'ONNX quantifie en UINT8 : rien.
enum class Grid : std::uint8_t { E4m3, Int8 };
std::optional<Grid> zero_grid(const Graph& graph, const Node& node) {
  if (node.inputs.size() != 3) return std::nullopt;
  const Tensor* zero = find_initializer(graph, node.inputs[2]);
  if (zero == nullptr || !std::ranges::all_of(zero->floats, [](float v) { return v == 0.0F; }))
    return std::nullopt;
  if (zero->data_type == kFloat8E4M3Fn) return Grid::E4m3;
  if (zero->data_type == kInt8) return Grid::Int8;
  return std::nullopt;
}

float on_grid(Grid grid, float v, float s) {
  return grid == Grid::Int8 ? round_to_int8(v / s) * s : round_to_e4m3(v / s) * s;
}

// q × échelle comme le calcule DequantizeLinear : dans le type de l'échelle (FLOAT ou FLOAT16).
float dequantized(float q, float scale, std::int32_t type) {
  const float v = q * scale;
  return type == kFloat16 ? kernels::ref::round_to_half(v) : v;
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

// Poids constants quantifiés par canal : x_k arrondi à la grille de l'échelle de son canal. dequantize : x_k
// est déjà le q d'un initialiseur INT8 (DequantizeLinear seul), rendu q × échelle.
std::optional<Tensor> fold_per_channel(const Tensor& weights, const Tensor& scale, std::int64_t axis,
                                       std::string name, Grid grid, bool dequantize = false) {
  const auto rank = static_cast<std::int64_t>(weights.dims.size());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank || weights.floats.empty() ||
      weights.dims[static_cast<std::size_t>(axis)] != static_cast<std::int64_t>(scale.floats.size()) ||
      (dequantize ? weights.data_type != kInt8
                  : (weights.data_type != kFloat && weights.data_type != kFloat16))) {
    return std::nullopt;
  }
  std::size_t inner = 1;
  for (std::int64_t d = axis + 1; d < rank; ++d)
    inner *= static_cast<std::size_t>(weights.dims[static_cast<std::size_t>(d)]);
  Tensor out = weights;
  out.name = std::move(name);
  if (dequantize) out.data_type = scale.data_type;
  for (std::size_t k = 0; k < out.floats.size(); ++k) {
    const float s = scale.floats[(k / inner) % scale.floats.size()];
    out.floats[k] =
        dequantize ? dequantized(out.floats[k], s, scale.data_type) : on_grid(grid, out.floats[k], s);
  }
  return out;
}

// Le DequantizeLinear de même grille qui lit seul la sortie du QuantizeLinear. Sans zéro (deux entrées), le
// sien vaut 0 dans le type de son entrée, celui du QuantizeLinear.
bool pairs_with(const Graph& graph, const Node& quant, const Node& dequant, Grid grid) {
  return is_default_domain(dequant.domain) && dequant.op_type == "DequantizeLinear" &&
         dequant.outputs.size() == 1 && (dequant.inputs.size() == 2 || dequant.inputs.size() == 3) &&
         dequant.inputs[0] == quant.outputs[0] &&
         (dequant.inputs.size() == 2 || zero_grid(graph, dequant) == grid);
}

// Un DequantizeLinear seul d'un initialiseur INT8 (poids rangés en entiers par l'outil de quantification) :
// l'initialiseur flottant q × échelle, dans le type de l'échelle, scalaire ou par canal ; nullopt sinon. Sans
// zéro (deux entrées), celui d'un INT8 vaut 0.
std::optional<Tensor> fold_dequantized(const Graph& graph, const Node& dequant) {
  if (!is_default_domain(dequant.domain) || dequant.op_type != "DequantizeLinear" ||
      dequant.outputs.size() != 1 || (dequant.inputs.size() != 2 && zero_grid(graph, dequant) != Grid::Int8))
    return std::nullopt;
  const Tensor* weights = find_initializer(graph, dequant.inputs[0]);
  const Tensor* scale = positive_scale(graph, dequant);
  const auto axis = quant_axis(dequant);
  if (weights == nullptr || weights->data_type != kInt8 || scale == nullptr || !axis) return std::nullopt;
  if (scale->dims.empty()) {
    Tensor out = *weights;
    out.name = dequant.outputs[0];
    out.data_type = scale->data_type;
    for (float& v : out.floats) v = dequantized(v, scale->floats[0], scale->data_type);
    return out;
  }
  return fold_per_channel(*weights, *scale, *axis, dequant.outputs[0], Grid::Int8, true);
}

// Les DequantizeLinear INT8 seuls, sur des poids entiers, pliés en initialiseurs (ajoutés à folded) ; leur
// nombre.
std::size_t fold_lone_dequantized(const Graph& graph, std::vector<bool>& removed,
                                  std::vector<Tensor>& folded) {
  std::size_t fused = 0;
  for (std::size_t d = 0; d < graph.nodes.size(); ++d) {
    if (removed[d]) continue;
    auto weights = fold_dequantized(graph, graph.nodes[d]);
    if (!weights) continue;
    folded.push_back(std::move(*weights));
    removed[d] = true;
    ++fused;
  }
  return fused;
}

// Les nœuds retirés, les initialiseurs pliés ajoutés ; puis les zéros FP8 et INT8, et les poids INT8, que
// plus aucun nœud ne lit.
void apply_fusion(Graph& graph, const std::vector<bool>& removed, std::vector<Tensor>& folded) {
  std::size_t kept = 0;
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    if (removed[n]) continue;
    if (kept != n) graph.nodes[kept] = std::move(graph.nodes[n]);
    ++kept;
  }
  graph.nodes.resize(kept);
  for (Tensor& t : folded) graph.initializers.push_back(std::move(t));
  std::unordered_set<std::string> still_read;
  for (const Node& node : graph.nodes) still_read.insert(node.inputs.begin(), node.inputs.end());
  std::erase_if(graph.initializers, [&](const Tensor& t) {
    return (t.data_type == kFloat8E4M3Fn || t.data_type == kInt8) && !still_read.contains(t.name);
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
  std::vector<Tensor> folded;
  std::size_t fused = 0;
  for (std::size_t q = 0; q < graph.nodes.size(); ++q) {
    Node& quant = graph.nodes[q];
    if (!is_default_domain(quant.domain) || quant.op_type != "QuantizeLinear" || quant.outputs.size() != 1)
      continue;
    const std::optional<Grid> grid = zero_grid(graph, quant);
    if (!grid) continue;
    const auto users = readers.find(quant.outputs[0]);
    if (users == readers.end() || users->second.size() != 1 || graph_outputs.contains(quant.outputs[0]))
      continue;
    const std::size_t d = users->second[0];
    Node& dequant = graph.nodes[d];
    if (!pairs_with(graph, quant, dequant, *grid)) continue;
    const Tensor* qs = positive_scale(graph, quant);
    const Tensor* ds = positive_scale(graph, dequant);
    const auto q_axis = quant_axis(quant);
    const auto d_axis = quant_axis(dequant);
    if (qs == nullptr || ds == nullptr || !q_axis || !d_axis || qs->floats != ds->floats) continue;

    if (qs->dims.empty()) {
      Node fake{.name = dequant.name.empty() ? quant.name : dequant.name,
                .op_type = *grid == Grid::Int8 ? "FakeQuantInt8" : "FakeQuantE4M3",
                .domain = "",
                .inputs = {quant.inputs[0], quant.inputs[1]},
                .outputs = {dequant.outputs[0]}};
      quant = std::move(fake);
      removed[d] = true;
    } else {
      // Par canal : seuls des poids constants se plient, en un initialiseur arrondi.
      const Tensor* weights = find_initializer(graph, quant.inputs[0]);
      if (weights == nullptr || *q_axis != *d_axis) continue;
      auto rounded = fold_per_channel(*weights, *qs, *q_axis, dequant.outputs[0], *grid);
      if (!rounded) continue;
      folded.push_back(std::move(*rounded));
      removed[q] = true;
      removed[d] = true;
    }
    ++fused;
  }
  fused += fold_lone_dequantized(graph, removed, folded);
  if (fused == 0) return 0;
  apply_fusion(graph, removed, folded);
  return fused;
}

}  // namespace ajp::gpu::ir::onnx
