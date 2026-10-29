#include "gpu/vulkan/backend/net_hoist.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "gpu/ir/graph.h"

namespace ajp::gpu::vk::net {
namespace {

using ir::Graph;
using ir::Node;
using ir::OpKind;
using ir::TensorId;

constexpr std::size_t kSizes = 3;  // entrée « sizes » d'un Resize

// Les noyaux qui écrivent leur sortie dans sa tranche d'un Concat (net_alias.cpp, slab_writer) :
// une entrée qu'ils produisent seuls n'a pas besoin d'être copiée.
bool writes_in_place(OpKind op) {
  switch (op) {
    case OpKind::Conv:
    case OpKind::Relu:
    case OpKind::LeakyRelu:
    case OpKind::Sigmoid:
    case OpKind::Tanh:
    case OpKind::Clip:
    case OpKind::Reciprocal:
    case OpKind::Add:
    case OpKind::Sub:
    case OpKind::Mul:
    case OpKind::Div:
    case OpKind::PRelu:
    case OpKind::Resize:
    case OpKind::GridSample:
      return true;
    default:
      return false;
  }
}

class Hoister {
 public:
  explicit Hoister(Graph& graph) : g_(graph) {}

  int run() {
    int done = 0;
    while (step()) ++done;
    return done;
  }

 private:
  const ir::Tensor& tensor(TensorId id) const { return g_.tensors[id]; }
  const ir::Dims& dims(TensorId id) const { return g_.tensors[id].dims; }

  void refresh() {
    readers_ = ir::consumers(g_);
    maker_ = ir::producers(g_);
  }

  // Le lecteur unique de id, hors sortie du graphe.
  std::optional<std::size_t> sole_reader(TensorId id) const {
    if (ir::is_graph_output(g_, id) || readers_[id].size() != 1) return std::nullopt;
    const Node& reader = g_.nodes[readers_[id][0]];
    if (std::ranges::count(reader.inputs, id) != 1) return std::nullopt;
    return readers_[id][0];
  }

  // Tenseur de rang 4 connu, en Nchw, calculé ou donné par le graphe.
  bool plain4(TensorId id) const {
    const ir::Tensor& t = tensor(id);
    return t.known && t.kind != ir::TensorKind::Constant && t.layout == ir::Layout::Nchw &&
           t.dims.size() == 4 && t.type != ir::DataType::I64;
  }

  static std::optional<std::size_t> channel_axis(const Node& node, std::size_t rank) {
    std::int64_t axis = 0;
    if (const auto* a = std::get_if<ir::AxisAttrs>(&node.attrs)) {
      axis = a->axis;
    } else if (const auto* s = std::get_if<ir::SplitAttrs>(&node.attrs)) {
      axis = s->axis;
    } else {
      return std::nullopt;
    }
    const auto r = static_cast<std::int64_t>(rank);
    if (axis < -r || axis >= r) return std::nullopt;
    return static_cast<std::size_t>(axis < 0 ? axis + r : axis);
  }

  // Resize de rang 4 qui réduit l'image sans toucher aux lots ni aux canaux, entrée et sortie en Nchw.
  bool downscale(const Node& node) const {
    if (node.op != OpKind::Resize || node.inputs.empty() || node.outputs.size() != 1) return false;
    const TensorId in = node.inputs[0];
    const TensorId out = node.outputs[0];
    if (in == ir::kNoTensor || !plain4(in) || !plain4(out)) return false;
    const ir::Dims& x = dims(in);
    const ir::Dims& y = dims(out);
    if (x[0] != y[0] || x[1] != y[1] || y[2] > x[2] || y[3] > x[3] || y[2] * y[3] >= x[2] * x[3])
      return false;
    if (tensor(in).type != tensor(out).type) return false;
    const auto* attrs = std::get_if<ir::ResizeAttrs>(&node.attrs);
    if (attrs == nullptr) return false;
    // Axes explicites : H et W seuls ; sinon, des échelles à 1 sur le lot et les canaux.
    if (!attrs->axes.empty()) {
      return std::ranges::all_of(attrs->axes,
                                 [](std::int64_t a) { return a == 2 || a == 3 || a == -2 || a == -1; });
    }
    if (node.inputs.size() > 2 && node.inputs[2] != ir::kNoTensor &&
        !tensor(node.inputs[2]).value.f.empty()) {
      const std::vector<float>& scales = tensor(node.inputs[2]).value.f;
      return scales.size() == 4 && scales[0] == 1.0F && scales[1] == 1.0F;
    }
    return true;  // « sizes » : lot et canaux égaux à ceux de l'entrée, vérifié ci-dessus
  }

  // Même réduction, sur les mêmes axes, avec les mêmes paramètres.
  bool same_resize(const Node& a, const Node& b) const {
    const auto* pa = std::get_if<ir::ResizeAttrs>(&a.attrs);
    const auto* pb = std::get_if<ir::ResizeAttrs>(&b.attrs);
    if (pa == nullptr || pb == nullptr || pa->mode != pb->mode || pa->coord != pb->coord ||
        pa->nearest != pb->nearest || pa->axes != pb->axes) {
      return false;
    }
    const ir::Dims& xa = dims(a.inputs[0]);
    const ir::Dims& xb = dims(b.inputs[0]);
    const ir::Dims& ya = dims(a.outputs[0]);
    const ir::Dims& yb = dims(b.outputs[0]);
    if (xa[2] != xb[2] || xa[3] != xb[3] || ya[2] != yb[2] || ya[3] != yb[3]) return false;
    const auto scales = [&](const Node& n) -> const std::vector<float>* {
      if (n.inputs.size() > 2 && n.inputs[2] != ir::kNoTensor && !tensor(n.inputs[2]).value.f.empty())
        return &tensor(n.inputs[2]).value.f;
      return nullptr;
    };
    const auto* sa = scales(a);
    const auto* sb = scales(b);
    return (sa == nullptr) == (sb == nullptr) && (sa == nullptr || *sa == *sb);
  }

  TensorId new_activation(const ir::Tensor& like, const std::string& name, ir::Dims shape) {
    ir::Tensor t;
    t.name = name;
    t.kind = ir::TensorKind::Activation;
    t.type = like.type;
    t.dims = std::move(shape);
    t.known = true;
    t.layout = ir::Layout::Nchw;
    return ir::add_tensor(g_, std::move(t));
  }

  TensorId new_ints(const std::string& name, ir::Dims values) {
    ir::Tensor t;
    t.name = name;
    t.kind = ir::TensorKind::Constant;
    t.type = ir::DataType::I64;
    t.dims = {static_cast<std::int64_t>(values.size())};
    t.known = true;
    t.value = ir::make_value(ir::DataType::I64, t.dims);
    t.value.i = std::move(values);
    return ir::add_tensor(g_, std::move(t));
  }

  // Le Resize like appliqué à in, vers out (dont les dimensions sont posées) : mêmes paramètres,
  // et les tailles demandées, si le Resize en donne, suivent la sortie.
  Node resized(const Node& like, TensorId in, TensorId out) {
    Node n = like;
    n.name = std::format("{}_remonte{}", like.name, serial_++);
    n.inputs[0] = in;
    n.outputs = {out};
    if (n.inputs.size() > kSizes && n.inputs[kSizes] != ir::kNoTensor &&
        !tensor(n.inputs[kSizes]).value.i.empty()) {
      const auto* attrs = std::get_if<ir::ResizeAttrs>(&like.attrs);
      ir::Dims sizes;
      if (attrs != nullptr && !attrs->axes.empty()) {
        for (const std::int64_t a : attrs->axes) {
          const std::int64_t axis = a < 0 ? a + 4 : a;
          sizes.push_back(dims(out)[static_cast<std::size_t>(axis)]);
        }
      } else {
        sizes = dims(out);
      }
      n.inputs[kSizes] = new_ints(std::format("{}_tailles{}", like.name, serial_++), std::move(sizes));
    }
    return n;
  }

  // Remplace les nœuds erased par add, rangés à la place du nœud at (indice d'avant la réécriture).
  void splice(const std::vector<std::size_t>& erased, std::size_t at, std::vector<Node> add) {
    std::vector<Node> nodes;
    nodes.reserve(g_.nodes.size() + add.size());
    for (std::size_t k = 0; k < g_.nodes.size(); ++k) {
      if (k == at) {
        for (Node& n : add) nodes.push_back(std::move(n));
      }
      if (std::ranges::count(erased, k) == 0) nodes.push_back(std::move(g_.nodes[k]));
    }
    g_.nodes = std::move(nodes);
  }

  bool step() {
    refresh();
    for (std::size_t k = 0; k < g_.nodes.size(); ++k) {
      if (g_.nodes[k].op == OpKind::Resize && downscale(g_.nodes[k])) {
        if (through_concat(k) || through_reshape(k)) return true;
      } else if (g_.nodes[k].op == OpKind::Split && through_split(k)) {
        return true;
      }
    }
    return false;
  }

  // Resize(Concat(...)) → Concat(Resize(...)), si une entrée au moins devrait être copiée.
  bool through_concat(std::size_t r) {
    const Node resize = g_.nodes[r];
    const TensorId x = resize.inputs[0];
    if (maker_[x] == ir::kNoNode || sole_reader(x) != r) return false;
    const std::size_t c = maker_[x];
    const Node cat = g_.nodes[c];
    if (cat.op != OpKind::Concat || channel_axis(cat, 4) != std::optional<std::size_t>{1} ||
        cat.outputs.size() != 1) {
      return false;
    }
    bool copied = false;
    for (const TensorId in : cat.inputs) {
      if (in == ir::kNoTensor || !plain4(in) || tensor(in).type != tensor(x).type) return false;
      const bool in_place = maker_[in] != ir::kNoNode && writes_in_place(g_.nodes[maker_[in]].op) &&
                            sole_reader(in) == c && tensor(in).kind == ir::TensorKind::Activation;
      copied = copied || !in_place;
    }
    if (!copied) return false;
    std::vector<Node> add;
    Node merged = cat;
    merged.name = std::format("{}_remonte{}", cat.name, serial_++);
    merged.inputs.clear();
    merged.outputs = {resize.outputs[0]};
    for (const TensorId in : cat.inputs) {
      ir::Dims shape = dims(resize.outputs[0]);
      shape[1] = dims(in)[1];
      const TensorId small =
          new_activation(tensor(in), std::format("{}_reduit{}", tensor(in).name, serial_++), shape);
      add.push_back(resized(resize, in, small));
      merged.inputs.push_back(small);
    }
    add.push_back(std::move(merged));
    splice({c, r}, c, std::move(add));
    return true;
  }

  // Resize(Reshape(y)) → Reshape(Resize(y)), y sortie d'un Split de lot > 1 (que through_split
  // remonte ensuite) et le Reshape ne regroupe que le lot et les canaux.
  bool through_reshape(std::size_t r) {
    const Node resize = g_.nodes[r];
    const TensorId x = resize.inputs[0];
    if (maker_[x] == ir::kNoNode || sole_reader(x) != r) return false;
    const std::size_t q = maker_[x];
    const Node shape_node = g_.nodes[q];
    if (shape_node.op != OpKind::Reshape || shape_node.inputs.empty()) return false;
    const TensorId y = shape_node.inputs[0];
    if (!plain4(y) || dims(y)[2] != dims(x)[2] || dims(y)[3] != dims(x)[3] || maker_[y] == ir::kNoNode ||
        tensor(y).type != tensor(x).type) {
      return false;
    }
    const Node& split = g_.nodes[maker_[y]];
    if (split.op != OpKind::Split || channel_axis(split, 4) != std::optional<std::size_t>{1} ||
        dims(split.inputs[0])[0] <= 1) {
      return false;
    }
    ir::Dims small_y = dims(y);
    small_y[2] = dims(resize.outputs[0])[2];
    small_y[3] = dims(resize.outputs[0])[3];
    const TensorId reduced =
        new_activation(tensor(y), std::format("{}_reduit{}", tensor(y).name, serial_++), small_y);
    Node reshape = shape_node;
    reshape.name = std::format("{}_remonte{}", shape_node.name, serial_++);
    reshape.inputs = {
        reduced, new_ints(std::format("{}_forme{}", shape_node.name, serial_++), dims(resize.outputs[0]))};
    reshape.outputs = {resize.outputs[0]};
    std::vector<Node> add;
    add.push_back(resized(resize, y, reduced));
    add.push_back(std::move(reshape));
    splice({q, r}, q, std::move(add));
    return true;
  }

  // Resize de chaque sortie d'un Split de lot > 1 → un Resize de son entrée, puis le Split.
  bool through_split(std::size_t s) {
    const Node split = g_.nodes[s];
    if (channel_axis(split, 4) != std::optional<std::size_t>{1} || split.inputs.empty()) return false;
    const TensorId z = split.inputs[0];
    if (!plain4(z) || dims(z)[0] <= 1) return false;
    std::vector<std::size_t> resizes;
    for (const TensorId o : split.outputs) {
      const auto reader = sole_reader(o);
      if (!reader || g_.nodes[*reader].op != OpKind::Resize || g_.nodes[*reader].inputs[0] != o ||
          !downscale(g_.nodes[*reader]) || !plain4(o)) {
        return false;
      }
      if (!resizes.empty() && !same_resize(g_.nodes[resizes[0]], g_.nodes[*reader])) return false;
      resizes.push_back(*reader);
    }
    if (resizes.empty()) return false;
    const Node first = g_.nodes[resizes[0]];
    ir::Dims small = dims(z);
    small[2] = dims(first.outputs[0])[2];
    small[3] = dims(first.outputs[0])[3];
    const TensorId reduced =
        new_activation(tensor(z), std::format("{}_reduit{}", tensor(z).name, serial_++), small);
    Node divided = split;
    divided.name = std::format("{}_remonte{}", split.name, serial_++);
    divided.inputs[0] = reduced;
    divided.outputs.clear();
    for (const std::size_t reader : resizes) divided.outputs.push_back(g_.nodes[reader].outputs[0]);
    std::vector<Node> add;
    add.push_back(resized(first, z, reduced));
    add.push_back(std::move(divided));
    std::vector<std::size_t> erased = resizes;
    erased.push_back(s);
    splice(erased, s, std::move(add));
    return true;
  }

  Graph& g_;
  std::vector<std::vector<std::size_t>> readers_;
  std::vector<std::size_t> maker_;
  int serial_ = 0;
};

}  // namespace

int hoist_downscales(ir::Graph& graph) {
  return Hoister(graph).run();
}

}  // namespace ajp::gpu::vk::net
