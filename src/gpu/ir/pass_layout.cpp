// Disposition NHWC à canaux alignés sur 16, et enchaînement des passes.
#include <algorithm>
#include <format>

#include "gpu/ir/graph.h"
#include "gpu/ir/log.h"
#include "gpu/ir/passes.h"

namespace ajp::gpu::ir {
namespace {

// Activation interne de rang 4 et flottante : seule candidate à Nhwc16.
bool internal_image(const Graph& graph, TensorId id) {
  if (id == kNoTensor) return false;
  const Tensor& t = graph.tensors[id];
  return t.kind == TensorKind::Activation && t.known && t.dims.size() == 4 && t.type != DataType::I64 &&
         !is_graph_output(graph, id);
}

// Le nœud sait-il lire ou écrire ce tenseur en NHWC ? Les opérations qui
// réinterprètent les axes (Reshape, Transpose, Gather…) ne le savent pas.
bool accepts_nhwc(const Graph& graph, const Node& node, TensorId id) {
  const auto output_dims = [&] { return graph.tensors[node.outputs[0]].dims; };
  switch (node.op) {
    case OpKind::Conv:
    case OpKind::ConvTranspose:
    case OpKind::Relu:
    case OpKind::LeakyRelu:
    case OpKind::Sigmoid:
    case OpKind::Tanh:
    case OpKind::Clip:
    case OpKind::Cast:
    case OpKind::Identity:
    case OpKind::Concat:
    case OpKind::Split:
    case OpKind::Slice:
    case OpKind::Pad:
    case OpKind::Resize:
    case OpKind::DepthToSpace:
    case OpKind::SpaceToDepth:
    case OpKind::AveragePool:
      return true;
    case OpKind::Add:
    case OpKind::Sub:
    case OpKind::Mul:
    case OpKind::Div:
    case OpKind::PRelu:
      return graph.tensors[id].dims == output_dims();  // un opérande diffusé reste en Nchw
    case OpKind::GridSample:
      return id != node.inputs[1];  // la grille [N, H, W, 2] n'a pas de canaux
    default:
      return false;
  }
}

// Activations internes d'images d'un nœud, couplées : une seule disposition pour toutes.
// Une convolution relit son entrée par son propre noyau : elle peut changer de
// disposition, seuls sa sortie et son résiduel (lu élément par élément) sont couplés.
std::vector<TensorId> coupled(const Graph& graph, const Node& node) {
  std::vector<TensorId> candidates;
  if (node.op == OpKind::Conv) {
    candidates.push_back(node.outputs[0]);
    if (node.inputs.size() > kConvResidual) candidates.push_back(node.inputs[kConvResidual]);
  } else if (node.op != OpKind::ConvTranspose) {
    candidates = node.inputs;
    candidates.insert(candidates.end(), node.outputs.begin(), node.outputs.end());
  }
  std::vector<TensorId> ids;
  for (const TensorId id : candidates) {
    if (internal_image(graph, id) && accepts_nhwc(graph, node, id)) ids.push_back(id);
  }
  return ids;
}

}  // namespace

int assign_layout(Graph& graph) {
  std::vector<bool> nhwc(graph.tensors.size(), false);
  for (TensorId id = 0; id < graph.tensors.size(); ++id) nhwc[id] = internal_image(graph, id);
  // Un tenseur qu'un de ses nœuds ne sait pas ranger en NHWC reste en Nchw.
  for (const Node& node : graph.nodes) {
    for (const auto* list : {&node.inputs, &node.outputs}) {
      for (const TensorId id : *list) {
        if (id != kNoTensor && !accepts_nhwc(graph, node, id)) nhwc[id] = false;
      }
    }
  }
  // Point fixe : si une activation couplée d'un nœud reste en Nchw, toutes le restent.
  for (bool changed = true; changed;) {
    changed = false;
    for (const Node& node : graph.nodes) {
      const std::vector<TensorId> ids = coupled(graph, node);
      const bool any_nchw = std::ranges::any_of(ids, [&](TensorId id) { return !nhwc[id]; });
      for (const TensorId id : ids) {
        if (any_nchw && nhwc[id]) {
          nhwc[id] = false;
          changed = true;
        }
      }
    }
  }
  int count = 0;
  for (TensorId id = 0; id < graph.tensors.size(); ++id) {
    graph.tensors[id].layout = nhwc[id] ? Layout::Nhwc16 : Layout::Nchw;
    count += nhwc[id] ? 1 : 0;
  }
  return count;
}

std::expected<PassStats, Refusal> optimize(Graph& graph) {
  for (const Tensor& tensor : graph.tensors) {
    if (!tensor.known) {
      return std::unexpected(Refusal{RefusalCode::ShapeInvalid,
                                     std::format("{} : forme inconnue, infer_shapes d'abord", tensor.name)});
    }
  }
  PassStats stats;
  auto folded = fold_constants(graph);
  if (!folded) return std::unexpected(folded.error());
  stats.folded = *folded;
  stats.dead = eliminate_dead_code(graph);
  stats.identities = eliminate_identities(graph);
  stats.pixel_shuffles = fuse_pixel_shuffles(graph);
  stats.bias = fuse_conv_bias(graph);
  stats.residuals = fuse_residuals(graph);
  stats.activations = fuse_activations(graph);
  stats.dead += eliminate_dead_code(graph);
  stats.nhwc = assign_layout(graph);
  if (auto ok = validate(graph); !ok) return std::unexpected(ok.error());
  make_logger()->debug(
      "passes : {} pliés, {} morts, {} sans effet, {} réarrangements, {} biais, {} résiduels, "
      "{} activations, {} tenseurs NHWC16 ; {} nœuds restants",
      stats.folded, stats.dead, stats.identities, stats.pixel_shuffles, stats.bias, stats.residuals,
      stats.activations, stats.nhwc, graph.nodes.size());
  return stats;
}

}  // namespace ajp::gpu::ir
