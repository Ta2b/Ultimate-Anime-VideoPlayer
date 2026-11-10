// La grille des GridSample en FP32 (passes.h, promote_grid_chains).
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gpu/ir/graph.h"
#include "gpu/ir/passes.h"

namespace ajp::gpu::ir {
namespace {

// Entre la somme de la grille et GridSample : des nœuds qui déplacent ou convertissent, sans calcul.
bool carries(OpKind op) {
  switch (op) {
    case OpKind::Cast:
    case OpKind::Transpose:
    case OpKind::Concat:
    case OpKind::Reshape:
    case OpKind::Unsqueeze:
    case OpKind::Squeeze:
    case OpKind::Identity:
      return true;
    default:
      return false;
  }
}

class Chains {
 public:
  explicit Chains(Graph& graph)
      : graph_(graph), producers_(producers(graph)), seen_(graph.tensors.size(), false) {}

  // De la grille d'un GridSample vers l'amont : les nœuds qui la portent, jusqu'à sa somme.
  void carried(TensorId id) {
    if (!enter(id)) return;
    Node& node = graph_.nodes[producers_[id]];
    if (carries(node.op)) {
      promote(id);
      const std::size_t count = node.op == OpKind::Concat ? node.inputs.size() : 1;
      for (std::size_t k = 0; k < count; ++k) carried(node.inputs[k]);
    } else if (node.op == OpKind::Add || node.op == OpKind::Sub) {
      // La somme base + flot × pas : c'est elle qui ne tient pas en float16.
      promote(id);
      for (const TensorId in : node.inputs) term(in);
    }
  }

  int promoted() const { return promoted_; }

  // Un nœud ne lit que des flottants d'un seul type : devant chaque entrée restée float16 d'un nœud passé
  // en float (le flot devant son produit, une base constante), un Cast vers float, exact.
  void cast_inputs() {
    std::vector<Node> nodes;
    nodes.reserve(graph_.nodes.size());
    for (Node& node : graph_.nodes) {
      const bool lifted = node.op != OpKind::Cast && !node.outputs.empty() && touched(node.outputs[0]);
      for (TensorId& in : node.inputs) {
        if (!lifted || in == kNoTensor || graph_.tensors[in].type != DataType::F16) continue;
        std::string name =
            graph_.tensors[in].name + "/grille32";  // avant add_tensor, qui déplace les tenseurs
        const TensorId cast = add_tensor(graph_, Tensor{.name = name,
                                                        .kind = TensorKind::Activation,
                                                        .type = DataType::F32,
                                                        .dims = graph_.tensors[in].dims,
                                                        .known = graph_.tensors[in].known});
        nodes.push_back(Node{.name = std::move(name),
                             .op = OpKind::Cast,
                             .attrs = CastAttrs{.to = DataType::F32},
                             .inputs = {in},
                             .outputs = {cast}});
        in = cast;
      }
      nodes.push_back(std::move(node));
    }
    graph_.nodes = std::move(nodes);
  }

 private:
  bool enter(TensorId id) {
    if (id == kNoTensor || seen_[id] || producers_[id] == kNoNode) return false;
    seen_[id] = true;
    return graph_.tensors[id].type != DataType::I64;
  }

  // Un terme de la somme : la base, derrière ses Cast, ou le produit flot × pas, dont les facteurs gardent
  // leur type (le flot est une donnée float16, seul un Cast qui arrondirait le pas est levé).
  void term(TensorId id) {
    if (!enter(id)) return;
    const Node& node = graph_.nodes[producers_[id]];
    if (node.op == OpKind::Cast) {
      promote(id);
      term(node.inputs[0]);
    } else if (node.op == OpKind::Mul) {
      promote(id);
      for (const TensorId in : node.inputs) {
        if (in != kNoTensor && producers_[in] != kNoNode && graph_.nodes[producers_[in]].op == OpKind::Cast)
          promote(in);
      }
    }
  }

  // Le tenseur passe en float ; le Cast qui l'écrivait en float16 ne l'arrondit plus.
  void promote(TensorId id) {
    Tensor& tensor = graph_.tensors[id];
    if (tensor.type != DataType::F16 || tensor.kind != TensorKind::Activation) return;
    tensor.type = DataType::F32;
    touched_.push_back(id);
    Node& node = graph_.nodes[producers_[id]];
    if (node.op == OpKind::Cast) std::get<CastAttrs>(node.attrs).to = DataType::F32;
    ++promoted_;
  }

  bool touched(TensorId id) const { return std::ranges::find(touched_, id) != touched_.end(); }

  Graph& graph_;
  std::vector<TensorId> touched_;
  std::vector<std::size_t> producers_;
  std::vector<bool> seen_;
  int promoted_ = 0;
};

}  // namespace

int promote_grid_chains(Graph& graph) {
  Chains chains(graph);
  for (const Node& node : graph.nodes) {
    if (node.op == OpKind::GridSample && node.inputs.size() > 1) chains.carried(node.inputs[1]);
  }
  if (chains.promoted() > 0) chains.cast_inputs();
  return chains.promoted();
}

}  // namespace ajp::gpu::ir
