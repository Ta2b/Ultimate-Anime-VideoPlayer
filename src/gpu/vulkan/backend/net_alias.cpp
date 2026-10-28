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
  Planner(const ir::Graph& graph, std::uint64_t alignment, bool concat_in_place, bool split_views)
      : graph_(graph),
        alignment_(std::max<std::uint64_t>(alignment, 4)),
        concat_in_place_(concat_in_place),
        split_views_(split_views),
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
    producer_.assign(graph.tensors.size(), nullptr);
    readers_.assign(graph.tensors.size(), 0);
    for (const ir::Node& node : graph.nodes) {
      for (const ir::TensorId id : node.outputs) {
        if (id != ir::kNoTensor) producer_[id] = &node;
      }
      for (const ir::TensorId id : node.inputs) {
        if (id != ir::kNoTensor) ++readers_[id];
      }
    }
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

  // Les noyaux qui écrivent leur sortie par son descripteur (indice t_index, au pas de pixel cp) : un
  // enfant de Concat y reçoit sa tranche de canaux sans que le noyau le sache.
  static bool slab_writer(ir::OpKind op) {
    switch (op) {
      case ir::OpKind::Conv:
      case ir::OpKind::Relu:
      case ir::OpKind::LeakyRelu:
      case ir::OpKind::Sigmoid:
      case ir::OpKind::Tanh:
      case ir::OpKind::Clip:
      case ir::OpKind::Reciprocal:
      case ir::OpKind::Add:
      case ir::OpKind::Sub:
      case ir::OpKind::Mul:
      case ir::OpKind::Div:
      case ir::OpKind::PRelu:
      case ir::OpKind::Resize:
      case ir::OpKind::GridSample:
        return true;
      default:
        return false;
    }
  }

  // Sorties d'un Split sur les canaux Nhwc16 : des vues de l'entrée, lues au pas de pixel de celle-ci à
  // un décalage de canaux multiple de 16 (les canaux de bourrage d'une sortie, lus dans la tranche
  // suivante, ne comptent pas : les poids et les boucles s'arrêtent aux canaux logiques).
  void split_channels(const ir::Node& node, std::size_t axis) {
    const ir::TensorId in = node.inputs[0];
    const ir::Tensor& x = graph_.tensors[in];
    if (axis != 1 || x.dims.size() != 4 || x.layout != ir::Layout::Nhwc16 || !x.known ||
        aliases_[in].parent != ir::kNoTensor || x.kind == ir::TensorKind::Constant) {
      return;
    }
    std::int64_t start = 0;
    for (const ir::TensorId out : node.outputs) {
      const ir::Tensor& y = graph_.tensors[out];
      const std::int64_t offset = start;
      start += y.dims[1];
      const std::uint64_t bytes = static_cast<std::uint64_t>(offset) * ir::element_size(x.type);
      if (offset % 16 != 0 || !y.known || y.kind != ir::TensorKind::Activation || io_[out] ||
          aliases_[out].parent != ir::kNoTensor || y.layout != ir::Layout::Nhwc16 || y.type != x.type ||
          bytes % alignment_ != 0 || y.dims.size() != 4 || y.dims[0] != x.dims[0] || y.dims[2] != x.dims[2] ||
          y.dims[3] != x.dims[3] || root(in) == out) {
        continue;
      }
      aliases_[out] = {.parent = in, .offset = bytes, .channel_slab = true};
      last_[root(in)] = std::max(last_[root(in)], last_[out]);
    }
  }

  // Un Slice des canaux Nhwc16 au pas de 1 (les autres axes entiers) : une vue de l'entrée, comme une
  // sortie de Split.
  void slice_channels(const ir::Node& node, const std::vector<SliceAxis>& plan) {
    const ir::TensorId in = node.inputs[0];
    const ir::TensorId out = node.outputs[0];
    const ir::Tensor& x = graph_.tensors[in];
    const ir::Tensor& y = graph_.tensors[out];
    if (x.dims.size() != 4 || plan.size() != 4 || !x.known || !y.known || x.layout != ir::Layout::Nhwc16 ||
        y.layout != ir::Layout::Nhwc16 || x.kind == ir::TensorKind::Constant ||
        aliases_[in].parent != ir::kNoTensor || y.kind != ir::TensorKind::Activation || io_[out] ||
        aliases_[out].parent != ir::kNoTensor || y.type != x.type || y.dims.size() != 4) {
      return;
    }
    for (std::size_t k = 0; k < plan.size(); ++k) {
      const bool whole = plan[k].start == 0 && plan[k].length == x.dims[k] && plan[k].step == 1;
      if (k != 1 && !whole) return;
    }
    const SliceAxis& channels = plan[1];
    const std::uint64_t bytes = static_cast<std::uint64_t>(channels.start) * ir::element_size(x.type);
    if (channels.step != 1 || channels.start < 0 || channels.start % 16 != 0 || bytes % alignment_ != 0 ||
        y.dims[1] != channels.length || channels.start + channels.length > x.dims[1] || root(in) == out) {
      return;
    }
    aliases_[out] = {.parent = in, .offset = bytes, .channel_slab = true};
    last_[root(in)] = std::max(last_[root(in)], last_[out]);
  }

  // Entrées d'un Concat sur les canaux Nhwc16 : la Conv écrit dans la sortie du Concat, au pas
  // de pixel de celle-ci, à son décalage de canaux. L'enfant n'est lu que par le Concat, dont les
  // décalages sont des multiples de 16 canaux (tuiles entières, vues alignées) ; les canaux de
  // bourrage d'un enfant, non multiple de 16, tomberaient dans l'entrée suivante : seul le dernier
  // enfant peut en avoir.
  void concat_channels(const ir::Node& node, std::size_t axis) {
    const ir::TensorId out = node.outputs[0];
    const ir::Tensor& y = graph_.tensors[out];
    if (axis != 1 || y.dims.size() != 4 || y.layout != ir::Layout::Nhwc16 || !y.known) return;
    std::int64_t start = 0;
    for (std::size_t k = 0; k < node.inputs.size(); ++k) {
      const ir::TensorId in = node.inputs[k];
      const ir::Tensor& x = graph_.tensors[in];
      const std::int64_t channels = x.dims[1];
      const std::int64_t offset = start;
      start += channels;
      if (offset % 16 != 0 || (channels % 16 != 0 && k + 1 != node.inputs.size())) continue;
      const std::uint64_t bytes = static_cast<std::uint64_t>(offset) * ir::element_size(y.type);
      const ir::Node* maker = producer_[in];
      if (maker == nullptr || !slab_writer(maker->op) || readers_[in] != 1 ||
          std::ranges::count(node.inputs, in) != 1 || x.kind != ir::TensorKind::Activation || io_[in] ||
          aliases_[in].parent != ir::kNoTensor || x.layout != ir::Layout::Nhwc16 || x.type != y.type ||
          !x.known || bytes % alignment_ != 0 || x.dims[0] != y.dims[0] || x.dims[2] != y.dims[2] ||
          x.dims[3] != y.dims[3] || root(out) == in) {
        continue;
      }
      aliases_[in] = {.parent = out, .offset = bytes, .channel_slab = true};
      last_[root(out)] = std::max(last_[root(out)], last_[in]);
    }
  }

  void concat(const ir::Node& node) {
    const ir::TensorId out = node.outputs[0];
    const ir::Tensor& y = graph_.tensors[out];
    const auto* attrs = std::get_if<ir::AxisAttrs>(&node.attrs);
    if (attrs != nullptr && concat_in_place_) {
      if (const auto axis = normalized_axis(attrs->axis, y.dims.size())) concat_channels(node, *axis);
    }
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
    if (attrs != nullptr && split_views_) {
      if (const auto axis = normalized_axis(attrs->axis, x.dims.size())) split_channels(node, *axis);
    }
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
    if (plan && split_views_) slice_channels(node, *plan);
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
  bool concat_in_place_;
  bool split_views_;
  std::vector<ir::Alias> aliases_;
  std::vector<const ir::Node*> producer_;  // le nœud qui écrit chaque tenseur
  std::vector<std::size_t> readers_;       // nombre de lectures par tenseur
  std::vector<std::size_t> last_;          // dernière lecture, de la famille pour une racine
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

std::vector<ir::Alias> plan_aliases(const ir::Graph& graph, std::uint64_t alignment, bool concat_in_place,
                                    bool split_views) {
  return Planner(graph, alignment, concat_in_place, split_views).run();
}

}  // namespace ajp::gpu::vk::net
