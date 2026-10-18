// Pliage des constantes, élimination du code mort et des nœuds sans effet.
#include <algorithm>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"
#include "gpu/ir/passes.h"

namespace ajp::gpu::ir {
namespace {

// Au-delà, plier coûterait plus de mémoire que de calculer à l'exécution.
constexpr std::size_t kMaxFoldedElements = std::size_t{1} << 22;

bool constant(const Graph& graph, TensorId id) {
  return id == kNoTensor || graph.tensors[id].kind == TensorKind::Constant;
}

const Value* constant_value(const Graph& graph, const Node& node, std::size_t index) {
  if (index >= node.inputs.size() || node.inputs[index] == kNoTensor) return nullptr;
  const Tensor& tensor = graph.tensors[node.inputs[index]];
  return tensor.kind == TensorKind::Constant ? &tensor.value : nullptr;
}

bool has_input(const Node& node, std::size_t index) {
  return index < node.inputs.size() && node.inputs[index] != kNoTensor;
}

// Entrée facultative absente, ou constante dont tous les éléments vérifient pred.
template <class Pred>
bool absent_or_all(const Graph& graph, const Node& node, std::size_t index, Pred pred) {
  if (!has_input(node, index)) return true;
  const Value* value = constant_value(graph, node, index);
  if (value == nullptr) return false;
  return value->type == DataType::I64 ? std::ranges::all_of(value->i, pred)
                                      : std::ranges::all_of(value->f, pred);
}

bool identity_perm(const Node& node) {
  const auto* attrs = std::get_if<TransposeAttrs>(&node.attrs);
  if (attrs == nullptr) return false;
  for (std::size_t k = 0; k < attrs->perm.size(); ++k) {
    if (attrs->perm[k] != static_cast<std::int64_t>(k)) return false;
  }
  return !attrs->perm.empty();
}

// Le nœud recopie-t-il sa première entrée sur sa sortie, sans rien changer ?
bool is_noop(const Graph& graph, const Node& node) {
  if (node.inputs.empty() || node.outputs.size() != 1 || node.inputs[0] == kNoTensor) return false;
  const Tensor& in = graph.tensors[node.inputs[0]];
  const Tensor& out = graph.tensors[node.outputs[0]];
  if (!in.known || !out.known || in.type != out.type) return false;
  const bool same_dims = in.dims == out.dims;
  switch (node.op) {
    case OpKind::Identity:
    case OpKind::Cast:  // types égaux vérifiés plus haut
      return true;
    case OpKind::Reshape:
    case OpKind::Unsqueeze:
    case OpKind::Squeeze:
      return same_dims;
    case OpKind::Transpose:
      return identity_perm(node) || in.dims.size() <= 1;
    case OpKind::Concat:
      return node.inputs.size() == 1;
    case OpKind::Split:
      return same_dims;
    case OpKind::Slice:  // même forme et pas de 1 : tout est gardé, dans l'ordre
      return same_dims && absent_or_all(graph, node, 4, [](auto v) { return v == 1; });
    case OpKind::Pad:
      return same_dims && has_input(node, 1) && absent_or_all(graph, node, 1, [](auto v) { return v == 0; });
    case OpKind::Resize:  // sizes égales, ou échelles exactement 1
      return same_dims && absent_or_all(graph, node, 2, [](auto v) { return v == 1; });
    default:
      return false;
  }
}

}  // namespace

std::expected<int, Refusal> fold_constants(Graph& graph) {
  int folded = 0;
  std::vector<Node> kept;
  for (Node& node : graph.nodes) {
    const bool shape_of_known = node.op == OpKind::Shape && graph.tensors[node.inputs[0]].known;
    const bool inputs_constant =
        std::ranges::all_of(node.inputs, [&](TensorId id) { return constant(graph, id); });
    const bool small = std::ranges::all_of(node.outputs, [&](TensorId id) {
      const Tensor& t = graph.tensors[id];
      return t.known && element_count(t.dims) <= kMaxFoldedElements;
    });
    if (!(shape_of_known || inputs_constant) || !small) {
      kept.push_back(std::move(node));
      continue;
    }
    // Shape ne lit que la forme : une valeur sans données suffit.
    Value probe;
    std::vector<const Value*> args;
    if (node.op == OpKind::Shape) {
      const Tensor& in = graph.tensors[node.inputs[0]];
      probe = Value{.type = in.type, .dims = in.dims};
      args.push_back(&probe);
    } else {
      for (const TensorId id : node.inputs)
        args.push_back(id == kNoTensor ? nullptr : &graph.tensors[id].value);
    }
    auto results = evaluate(node, args);
    if (!results) return std::unexpected(results.error());
    for (std::size_t k = 0; k < node.outputs.size(); ++k) {
      Tensor& out = graph.tensors[node.outputs[k]];
      out.kind = TensorKind::Constant;
      out.type = (*results)[k].type;
      out.dims = (*results)[k].dims;
      out.known = true;
      out.value = std::move((*results)[k]);
    }
    ++folded;
  }
  graph.nodes = std::move(kept);
  return folded;
}

int eliminate_dead_code(Graph& graph) {
  std::vector<bool> live(graph.tensors.size(), false);
  for (const TensorId id : graph.outputs) live[id] = true;
  std::vector<bool> keep(graph.nodes.size(), false);
  for (std::size_t n = graph.nodes.size(); n-- > 0;) {
    const Node& node = graph.nodes[n];
    if (std::ranges::none_of(node.outputs, [&](TensorId id) { return live[id]; })) continue;
    keep[n] = true;
    for (const TensorId id : node.inputs) {
      if (id != kNoTensor) live[id] = true;
    }
  }
  std::vector<Node> kept;
  for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
    if (keep[n]) kept.push_back(std::move(graph.nodes[n]));
  }
  const auto removed = static_cast<int>(graph.nodes.size() - kept.size());
  graph.nodes = std::move(kept);
  compact(graph);
  return removed;
}

int eliminate_identities(Graph& graph) {
  int removed = 0;
  for (std::size_t n = 0; n < graph.nodes.size();) {
    const Node& node = graph.nodes[n];
    if (!is_noop(graph, node)) {
      ++n;
      continue;
    }
    const TensorId in = node.inputs[0];
    const TensorId out = node.outputs[0];
    if (!is_graph_output(graph, out)) {
      replace_uses(graph, out, in);
    } else {
      // La sortie du graphe garde son nom : son producteur l'écrit directement.
      const std::size_t producer = producers(graph)[in];
      if (producer == kNoNode || is_graph_output(graph, in)) {
        ++n;
        continue;
      }
      std::ranges::replace(graph.nodes[producer].outputs, in, out);
      replace_uses(graph, in, out);
    }
    graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(n));
    ++removed;
  }
  return removed;
}

}  // namespace ajp::gpu::ir
