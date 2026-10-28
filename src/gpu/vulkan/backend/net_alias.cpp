#include "gpu/vulkan/backend/net_alias.h"

#include <algorithm>
#include <variant>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_params.h"

namespace ajp::gpu::vk::net {
namespace {

// Rangé sans bourrage, dans l'ordre logique : une tranche contiguë y est un intervalle d'octets.
bool plain(const ir::Tensor& t) {
  return t.known && t.layout == ir::Layout::Nchw && t.type != ir::DataType::I64;
}

std::optional<std::size_t> normalized_axis(std::int64_t axis, std::size_t rank) {
  const std::int64_t r = static_cast<std::int64_t>(rank);
  if (axis < -r || axis >= r) return std::nullopt;
  return static_cast<std::size_t>(axis < 0 ? axis + r : axis);
}

class Planner {
 public:
  Planner(const ir::Graph& graph, std::uint64_t alignment)
      : graph_(graph),
        alignment_(std::max<std::uint64_t>(alignment, 4)),
        aliases_(graph.tensors.size()),
        last_(graph.tensors.size(), 0),
        io_(graph.tensors.size(), false) {
    for (std::size_t n = 0; n < graph.nodes.size(); ++n) {
      for (const ir::TensorId id : graph.nodes[n].inputs) {
        if (id != ir::kNoTensor) last_[id] = std::max(last_[id], n);
      }
    }
    for (const ir::TensorId id : graph.inputs) io_[id] = true;
    for (const ir::TensorId id : graph.outputs) io_[id] = true;
  }

  std::vector<ir::Alias> run() {
    for (const ir::Node& node : graph_.nodes) {
      switch (node.op) {
        case ir::OpKind::Concat:
          concat(node);
          break;
        case ir::OpKind::Split:
          split(node);
          break;
        case ir::OpKind::Slice:
          slice(node);
          break;
        case ir::OpKind::Reshape:
        case ir::OpKind::Squeeze:
        case ir::OpKind::Unsqueeze:
        case ir::OpKind::Identity:
          view(node.outputs[0], node.inputs[0], 0, true);
          break;
        default:
          break;
      }
    }
    return std::move(aliases_);
  }

 private:
  ir::TensorId root(ir::TensorId id) const {
    while (aliases_[id].parent != ir::kNoTensor) id = aliases_[id].parent;
    return id;
  }

  // Un tenseur qu'on peut ranger ailleurs : une activation interne, sans parent, rangée à plat.
  bool free_child(ir::TensorId id) const {
    const ir::Tensor& t = graph_.tensors[id];
    return t.kind == ir::TensorKind::Activation && !io_[id] && aliases_[id].parent == ir::kNoTensor &&
           plain(t);
  }

  // Range child dans parent si les deux le permettent. always : la famille ne s'allonge pas, ou
  // la tranche couvre l'essentiel du parent ; sinon un enfant qui vit plus longtemps que la
  // famille garderait tout le parent réservé pour lui seul.
  void link(ir::TensorId child, ir::TensorId parent, std::uint64_t offset, bool always) {
    const ir::Tensor& c = graph_.tensors[child];
    const ir::Tensor& p = graph_.tensors[parent];
    if (!free_child(child) || p.kind == ir::TensorKind::Constant || !plain(p) || c.type != p.type ||
        offset % alignment_ != 0 || offset + ir::storage_bytes(c) > ir::storage_bytes(p)) {
      return;
    }
    const ir::TensorId top = root(parent);
    if (top == child) return;  // déjà de la famille : un cycle
    const bool longer = last_[child] > last_[top];
    if (!always && longer && 2 * ir::storage_bytes(c) < ir::storage_bytes(p)) return;
    aliases_[child] = {.parent = parent, .offset = offset};
    last_[top] = std::max(last_[top], last_[child]);
  }

  // Une sortie lue dans son entrée : rien à écrire.
  void view(ir::TensorId out, ir::TensorId in, std::uint64_t offset, bool always) {
    if (in == ir::kNoTensor || out == ir::kNoTensor) return;
    link(out, in, offset, always);
  }

  void concat(const ir::Node& node) {
    const ir::TensorId out = node.outputs[0];
    const ir::Tensor& y = graph_.tensors[out];
    const auto* attrs = std::get_if<ir::AxisAttrs>(&node.attrs);
    if (attrs == nullptr || !plain(y)) return;
    const auto axis = normalized_axis(attrs->axis, y.dims.size());
    if (!axis) return;
    std::int64_t start = 0;
    for (const ir::TensorId in : node.inputs) {
      const auto offset = slab_offset(y, *axis, start);
      start += graph_.tensors[in].dims[*axis];
      if (!offset) return;
      // Le producteur écrit en place : une entrée de graphe ou une vue n'a pas de producteur.
      if (std::ranges::count(node.inputs, in) == 1) link(in, out, *offset, false);
    }
  }

  void split(const ir::Node& node) {
    const ir::TensorId in = node.inputs[0];
    const ir::Tensor& x = graph_.tensors[in];
    const auto* attrs = std::get_if<ir::SplitAttrs>(&node.attrs);
    if (attrs == nullptr || !plain(x)) return;
    const auto axis = normalized_axis(attrs->axis, x.dims.size());
    if (!axis) return;
    std::int64_t start = 0;
    for (const ir::TensorId out : node.outputs) {
      const auto offset = slab_offset(x, *axis, start);
      start += graph_.tensors[out].dims[*axis];
      if (offset) view(out, in, *offset, true);
    }
  }

  void slice(const ir::Node& node) {
    const ir::TensorId in = node.inputs[0];
    const ir::Tensor& x = graph_.tensors[in];
    auto plan = slice_plan(graph_, node);
    if (!plan || !plain(x) || plan->size() != x.dims.size()) return;
    // Le premier axe découpé ; ceux qui suivent sont entiers, tous au pas de 1.
    std::optional<std::size_t> cut;
    for (std::size_t k = 0; k < plan->size(); ++k) {
      const SliceAxis& a = (*plan)[k];
      const bool whole = a.start == 0 && a.length == x.dims[k];
      if (a.step != 1 && a.length > 1) return;
      if (!whole && cut) return;
      if (!whole) cut = k;
    }
    if (!cut) return;
    if (const auto offset = slab_offset(x, *cut, (*plan)[*cut].start))
      view(node.outputs[0], in, *offset, false);
  }

  const ir::Graph& graph_;
  std::uint64_t alignment_;
  std::vector<ir::Alias> aliases_;
  std::vector<std::size_t> last_;  // dernière lecture, de la famille pour une racine
  std::vector<bool> io_;
};

}  // namespace

std::optional<std::uint64_t> slab_offset(const ir::Tensor& whole, std::size_t axis, std::int64_t start) {
  if (!plain(whole) || axis >= whole.dims.size() || start < 0) return std::nullopt;
  for (std::size_t k = 0; k < axis; ++k) {
    if (whole.dims[k] != 1) return std::nullopt;
  }
  std::uint64_t inner = ir::element_size(whole.type);
  for (std::size_t k = axis + 1; k < whole.dims.size(); ++k)
    inner *= static_cast<std::uint64_t>(whole.dims[k]);
  return static_cast<std::uint64_t>(start) * inner;
}

std::vector<ir::Alias> plan_aliases(const ir::Graph& graph, std::uint64_t alignment) {
  return Planner(graph, alignment).run();
}

}  // namespace ajp::gpu::vk::net
