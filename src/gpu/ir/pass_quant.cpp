// Fusion des FakeQuantE4M3 (paires Q/DQ FP8) dans les convolutions voisines : une Conv qui lit
// ou écrit des valeurs arrondies à la grille E4M3 le fait elle-même (ConvAttrs::in_quant et
// out_quant), sans noyau ni tenseur intermédiaire pour l'arrondi.
#include <algorithm>
#include <optional>

#include "gpu/ir/graph.h"
#include "gpu/ir/passes.h"

namespace ajp::gpu::ir {
namespace {

// Échelle constante, scalaire et positive du FakeQuantE4M3.
std::optional<float> constant_scale(const Graph& graph, const Node& node) {
  if (node.inputs.size() != 2 || node.inputs[1] == kNoTensor) return std::nullopt;
  const Tensor& scale = graph.tensors[node.inputs[1]];
  if (scale.kind != TensorKind::Constant || scale.value.f.size() != 1 || !(scale.value.f[0] > 0.0F))
    return std::nullopt;
  return scale.value.f[0];
}

bool plain_epilogue(const Epilogue& e) {
  return !e.residual && e.act == Activation::None && !e.gate && e.space == 0;
}

bool try_quant(Graph& graph, std::size_t n, GraphIndex& index) {
  const Node& quant = graph.nodes[n];
  if (quant.op != OpKind::FakeQuantE4M3) return false;
  const auto scale = constant_scale(graph, quant);
  if (!scale) return false;
  const TensorId x = quant.inputs[0];
  const TensorId y = quant.outputs[0];

  // La Conv qui lit y seule : elle arrondit son entrée.
  if (const auto reader = index.sole_reader(y)) {
    Node& conv = graph.nodes[*reader];
    auto* attrs = std::get_if<ConvAttrs>(&conv.attrs);
    if (conv.op == OpKind::Conv && attrs != nullptr && attrs->in_quant == 0.0F && conv.inputs[kConvX] == y) {
      conv.inputs[kConvX] = x;
      attrs->in_quant = *scale;
      graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(n));
      return true;
    }
  }
  // Sinon la Conv qui produit x, dont ce nœud est le seul lecteur : elle arrondit sa sortie.
  const std::size_t producer = index.producer(x);
  if (producer == kNoNode || index.sole_reader(x) != n) return false;
  Node& conv = graph.nodes[producer];
  auto* attrs = std::get_if<ConvAttrs>(&conv.attrs);
  if (conv.op != OpKind::Conv || attrs == nullptr || attrs->out_quant != 0.0F ||
      !plain_epilogue(attrs->epilogue) || conv.outputs[0] != x) {
    return false;
  }
  conv.outputs[0] = y;
  attrs->out_quant = *scale;
  graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(n));
  return true;
}

}  // namespace

int fuse_fp8_quant(Graph& graph) {
  int fused = 0;
  GraphIndex index(graph);
  for (std::size_t n = 0; n < graph.nodes.size();) {
    if (try_quant(graph, n, index)) {
      index.invalidate();
      ++fused;
    } else {
      ++n;
    }
  }
  return fused;
}

}  // namespace ajp::gpu::ir
