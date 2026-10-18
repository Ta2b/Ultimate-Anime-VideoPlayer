// Reconnaissance des réarrangements de pixels écrits en Reshape, Transpose, Reshape.
//
// PyTorch exporte pixel_unshuffle ainsi (SPAN « unshuffle » d'AnimeJaNai), et
// parfois pixel_shuffle. Ramenés à SpaceToDepth et DepthToSpace, ils restent
// des opérations de rang 4 que la disposition NHWC sait ranger.
#include <optional>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/passes.h"

namespace ajp::gpu::ir {
namespace {

struct Rearrangement {
  OpKind op;
  BlockAttrs attrs;
};

// x : entrée de rang 4, t : forme de rang 6 après le premier Reshape, y : sortie de rang 4.
std::optional<Rearrangement> match(const Dims& x, const Dims& t, const Dims& perm, const Dims& y) {
  const std::int64_t n = x[0];
  const std::int64_t c = x[1];
  const std::int64_t h = x[2];
  const std::int64_t w = x[3];
  // SpaceToDepth : t = [N, C, H/b, b, W/b, b].
  const std::int64_t b = t[3];
  if (b > 1 && t[5] == b && h % b == 0 && w % b == 0 && t == Dims{n, c, h / b, b, w / b, b} &&
      y == Dims{n, c * b * b, h / b, w / b}) {
    if (perm == Dims{0, 1, 3, 5, 2, 4})
      return Rearrangement{OpKind::SpaceToDepth, {.blocksize = b, .mode = BlockMode::Crd}};
    if (perm == Dims{0, 3, 5, 1, 2, 4})
      return Rearrangement{OpKind::SpaceToDepth, {.blocksize = b, .mode = BlockMode::Dcr}};
  }
  // DepthToSpace CRD : t = [N, C/b², b, b, H, W] ; DCR : t = [N, b, b, C/b², H, W].
  for (const BlockMode mode : {BlockMode::Crd, BlockMode::Dcr}) {
    const std::int64_t s = mode == BlockMode::Crd ? t[2] : t[1];
    if (s < 2 || c % (s * s) != 0 || y != Dims{n, c / (s * s), h * s, w * s}) continue;
    const Dims crd{n, c / (s * s), s, s, h, w};
    const Dims dcr{n, s, s, c / (s * s), h, w};
    if (mode == BlockMode::Crd && t == crd && perm == Dims{0, 1, 4, 2, 5, 3}) {
      return Rearrangement{OpKind::DepthToSpace, {.blocksize = s, .mode = mode}};
    }
    if (mode == BlockMode::Dcr && t == dcr && perm == Dims{0, 3, 4, 1, 5, 2}) {
      return Rearrangement{OpKind::DepthToSpace, {.blocksize = s, .mode = mode}};
    }
  }
  return std::nullopt;
}

// Le nœud qui lit seul la sortie unique de node, s'il est de type op.
std::optional<std::size_t> next_of(const Graph& graph, const std::vector<std::vector<std::size_t>>& readers,
                                   const Node& node, OpKind op) {
  const TensorId out = node.outputs[0];
  if (is_graph_output(graph, out) || readers[out].size() != 1) return std::nullopt;
  const std::size_t next = readers[out][0];
  if (graph.nodes[next].op != op || graph.nodes[next].inputs[0] != out) return std::nullopt;
  return next;
}

bool try_rearrangement(Graph& graph, std::size_t first) {
  const Node& reshape = graph.nodes[first];
  if (reshape.op != OpKind::Reshape) return false;
  const auto readers = consumers(graph);
  const auto second = next_of(graph, readers, reshape, OpKind::Transpose);
  if (!second) return false;
  const auto third = next_of(graph, readers, graph.nodes[*second], OpKind::Reshape);
  if (!third) return false;

  const Tensor& x = graph.tensors[reshape.inputs[0]];
  const Tensor& t = graph.tensors[reshape.outputs[0]];
  const Tensor& y = graph.tensors[graph.nodes[*third].outputs[0]];
  const auto* transpose = std::get_if<TransposeAttrs>(&graph.nodes[*second].attrs);
  if (transpose == nullptr || x.dims.size() != 4 || t.dims.size() != 6 || y.dims.size() != 4) return false;
  const auto found = match(x.dims, t.dims, transpose->perm, y.dims);
  if (!found) return false;

  // Le nouveau nœud prend la place du dernier Reshape ; les deux autres disparaissent.
  Node& last = graph.nodes[*third];
  last.op = found->op;
  last.attrs = found->attrs;
  last.inputs = {reshape.inputs[0]};
  graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(*second));
  graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(first));
  return true;
}

}  // namespace

int fuse_pixel_shuffles(Graph& graph) {
  int fused = 0;
  for (std::size_t n = 0; n < graph.nodes.size();) {
    if (try_rearrangement(graph, n)) {
      ++fused;
    } else {
      ++n;
    }
  }
  return fused;
}

}  // namespace ajp::gpu::ir
