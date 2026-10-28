// Cast entre float et float16 que des noyaux typés font d'eux-mêmes (passes.h, fold_float_casts).
#include <algorithm>
#include <cstddef>

#include "gpu/ir/graph.h"
#include "gpu/ir/passes.h"

namespace ajp::gpu::ir {
namespace {

bool float_type(DataType type) {
  return type == DataType::F32 || type == DataType::F16;
}

// Un nœud dont le noyau lit et écrit chaque tenseur dans son type et calcule en FP32. Les
// convolutions en sont exclues : leurs variantes dépendent des types (calcul en float16 et
// matrices coopératives quand entrée et sortie le sont), changer un type changerait le calcul.
bool typed(const Node& node) {
  switch (node.op) {
    case OpKind::Conv:
    case OpKind::ConvTranspose:
    case OpKind::Shape:
    case OpKind::Gather:
      return false;
    default:
      return true;
  }
}

bool read_anywhere(const Graph& graph, TensorId id) {
  return std::ranges::any_of(graph.nodes, [&](const Node& n) { return std::ranges::contains(n.inputs, id); });
}

// Le Cast n de in vers out disparaît-il ? Les lecteurs sont recâblés au passage.
bool fold(Graph& graph, std::size_t n) {
  const TensorId in = graph.nodes[n].inputs[0];
  const TensorId out = graph.nodes[n].outputs[0];
  const DataType from = graph.tensors[in].type;
  const DataType to = graph.tensors[out].type;
  if (from == to) {
    // Un float16 rangé est déjà arrondi : la copie ne change rien, même vers float16.
    replace_uses(graph, out, in);
  } else if (from == DataType::F16) {
    // Vers float, la conversion est exacte : les lecteurs typés lisent la source.
    for (Node& reader : graph.nodes) {
      if (typed(reader)) std::ranges::replace(reader.inputs, out, in);
    }
  }
  if (!is_graph_output(graph, out) && !read_anywhere(graph, out)) return true;
  if (from == DataType::F16 && from != to) return false;
  // Vers float16 : le producteur écrit lui-même le résultat, arrondi à l'écriture comme le Cast.
  const std::size_t producer = producers(graph)[in];
  if (producer == kNoNode || !typed(graph.nodes[producer]) || is_graph_output(graph, in)) return false;
  const auto readers = consumers(graph)[in];
  if (readers.size() != 1 || readers[0] != n || std::ranges::count(graph.nodes[n].inputs, in) != 1)
    return false;
  std::ranges::replace(graph.nodes[producer].outputs, in, out);
  return true;
}

}  // namespace

int fold_float_casts(Graph& graph) {
  int removed = 0;
  for (std::size_t n = 0; n < graph.nodes.size();) {
    const Node& node = graph.nodes[n];
    const bool candidate = node.op == OpKind::Cast && !node.inputs.empty() && node.inputs[0] != kNoTensor &&
                           float_type(graph.tensors[node.inputs[0]].type) &&
                           float_type(graph.tensors[node.outputs[0]].type);
    if (!candidate || !fold(graph, n)) {
      ++n;
      continue;
    }
    graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(n));
    ++removed;
  }
  if (removed > 0) compact(graph);
  return removed;
}

}  // namespace ajp::gpu::ir
