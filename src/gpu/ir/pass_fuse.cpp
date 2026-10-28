// Fusions : réarrangements de pixels reconnus, Conv + biais, Conv + résiduel,
// Conv + activation, porte de SPAN, Conv + DepthToSpace. Une fusion n'a lieu que si la sortie intermédiaire
// n'a qu'un lecteur et n'est pas une sortie du graphe.
#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/passes.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::ir {
namespace {

// Le lecteur unique de id, s'il le lit une seule fois et que id reste interne.
std::optional<std::size_t> sole_reader(const Graph& graph, TensorId id) {
  if (is_graph_output(graph, id)) return std::nullopt;
  const auto readers = consumers(graph)[id];
  if (readers.size() != 1) return std::nullopt;
  if (std::ranges::count(graph.nodes[readers[0]].inputs, id) != 1) return std::nullopt;
  return readers[0];
}

const Value* constant_input(const Graph& graph, const Node& node, std::size_t index) {
  if (index >= node.inputs.size() || node.inputs[index] == kNoTensor) return nullptr;
  const Tensor& tensor = graph.tensors[node.inputs[index]];
  return tensor.kind == TensorKind::Constant ? &tensor.value : nullptr;
}

// Valeurs par canal d'une constante qui se diffuse sur [N, channels, H, W] sans
// en changer la forme : scalaire, [C, 1, 1] ou [1, C, 1, 1].
std::optional<std::vector<float>> per_channel(const Value& value, std::int64_t channels) {
  if (value.type == DataType::I64 || value.dims.size() > 4) return std::nullopt;
  const std::size_t count = element_count(value.dims);
  Dims padded(4 - value.dims.size(), 1);
  padded.insert(padded.end(), value.dims.begin(), value.dims.end());
  if (padded[0] != 1 || padded[2] != 1 || padded[3] != 1) return std::nullopt;
  if (count == 1) return std::vector<float>(static_cast<std::size_t>(channels), value.f[0]);
  if (padded[1] != channels) return std::nullopt;
  return value.f;
}

TensorId add_constant(Graph& graph, const std::string& base, Value value) {
  std::string name = base;
  while (find_tensor(graph, name)) name += '_';
  Tensor tensor{.name = std::move(name),
                .kind = TensorKind::Constant,
                .type = value.type,
                .dims = value.dims,
                .known = true,
                .value = std::move(value)};
  return add_tensor(graph, std::move(tensor));
}

// Conv sans activation fusionnée ; plain exige aussi qu'elle soit sans résiduel.
// L'épilogue applique biais, résiduel puis activation : une activation se fond
// encore après un résiduel, jamais l'inverse.
ConvAttrs* open_conv(Node& node, bool plain) {
  auto* attrs = std::get_if<ConvAttrs>(&node.attrs);
  if (node.op != OpKind::Conv || attrs == nullptr) return nullptr;
  const Epilogue& e = attrs->epilogue;
  return (plain && e.residual) || e.act != Activation::None || e.gate ? nullptr : attrs;
}

ConvAttrs* plain_conv(Node& node) {
  return open_conv(node, true);
}

// Biais et poids après une opération par canal y op c, avec y la sortie de la Conv.
bool fold_into_conv(Graph& graph, Node& conv, OpKind op, const std::vector<float>& c) {
  const Tensor& weights = graph.tensors[conv.inputs[kConvW]];
  const std::int64_t channels = weights.dims[0];
  if (channels == 0) return false;
  const std::size_t per_out = element_count(weights.dims) / static_cast<std::size_t>(channels);
  const bool has_bias = conv.inputs.size() > kConvB && conv.inputs[kConvB] != kNoTensor;
  const Value* bias = constant_input(graph, conv, kConvB);
  if (has_bias && bias == nullptr) return false;  // biais calculé à l'exécution : rien à fondre
  if (op == OpKind::Div && std::ranges::find(c, 0.0F) != c.end()) return false;
  Value w = weights.value;
  Value b = bias != nullptr ? *bias : make_value(w.type, Dims{channels});
  for (std::size_t m = 0; m < static_cast<std::size_t>(channels); ++m) {
    switch (op) {
      case OpKind::Add:
        b.f[m] += c[m];
        break;
      case OpKind::Sub:
        b.f[m] -= c[m];
        break;
      case OpKind::Mul:
      case OpKind::Div: {
        auto scale = [&](float v) { return op == OpKind::Mul ? v * c[m] : v / c[m]; };
        b.f[m] = scale(b.f[m]);
        for (std::size_t k = 0; k < per_out; ++k) w.f[(m * per_out) + k] = scale(w.f[(m * per_out) + k]);
        break;
      }
      default:
        return false;
    }
  }
  // Une Value F16 de l'IR est exactement représentable en binary16 : les poids et le biais fondus
  // sont arrondis une fois ici, et non à l'envoi au GPU.
  if (w.type == DataType::F16) {
    for (float& v : w.f) v = kernels::ref::round_to_half(v);
  }
  if (b.type == DataType::F16) {
    for (float& v : b.f) v = kernels::ref::round_to_half(v);
  }
  if (op == OpKind::Mul || op == OpKind::Div)
    conv.inputs[kConvW] = add_constant(graph, conv.name + "/poids", std::move(w));
  if (conv.inputs.size() <= kConvB) conv.inputs.resize(kConvB + 1, kNoTensor);
  conv.inputs[kConvB] = add_constant(graph, conv.name + "/biais", std::move(b));
  return true;
}

// Tente de fondre le lecteur de la Conv n ; renvoie vrai si le graphe a changé.
bool try_bias(Graph& graph, std::size_t n) {
  Node& conv = graph.nodes[n];
  if (plain_conv(conv) == nullptr || constant_input(graph, conv, kConvW) == nullptr) return false;
  const TensorId y = conv.outputs[0];
  const auto reader = sole_reader(graph, y);
  if (!reader) return false;
  const Node& next = graph.nodes[*reader];
  const bool commutes = next.op == OpKind::Add || next.op == OpKind::Mul;
  if (next.op != OpKind::Add && next.op != OpKind::Sub && next.op != OpKind::Mul && next.op != OpKind::Div)
    return false;
  const std::size_t y_index = next.inputs[0] == y ? 0 : 1;
  if (y_index == 1 && !commutes) return false;
  const Value* c = constant_input(graph, next, 1 - y_index);
  if (c == nullptr || graph.tensors[next.outputs[0]].dims != graph.tensors[y].dims) return false;
  const auto values = per_channel(*c, graph.tensors[y].dims[1]);
  if (!values || !fold_into_conv(graph, conv, next.op, *values)) return false;
  conv.outputs[0] = next.outputs[0];
  graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(*reader));
  return true;
}

bool try_residual(Graph& graph, std::size_t n) {
  ConvAttrs* attrs = plain_conv(graph.nodes[n]);
  if (attrs == nullptr) return false;
  const TensorId y = graph.nodes[n].outputs[0];
  const auto reader = sole_reader(graph, y);
  if (!reader || graph.nodes[*reader].op != OpKind::Add) return false;
  const Node& add = graph.nodes[*reader];
  const TensorId r = add.inputs[0] == y ? add.inputs[1] : add.inputs[0];
  const Tensor& yt = graph.tensors[y];
  if (graph.tensors[r].dims != yt.dims || graph.tensors[r].type != yt.type ||
      graph.tensors[add.outputs[0]].dims != yt.dims) {
    return false;
  }
  // La Conv prend la place de l'Add : le résiduel y est déjà calculé.
  attrs->epilogue.residual = true;
  Node conv = std::move(graph.nodes[n]);
  conv.inputs.resize(kConvResidual + 1, kNoTensor);
  conv.inputs[kConvResidual] = r;
  conv.outputs[0] = add.outputs[0];
  graph.nodes[*reader] = std::move(conv);
  graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(n));
  return true;
}

// Bornes constantes de Clip, ±infini si absentes.
std::optional<std::pair<float, float>> clip_bounds(const Graph& graph, const Node& clip) {
  std::pair<float, float> bounds{-std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
  for (std::size_t k = 1; k < 3; ++k) {
    if (k >= clip.inputs.size() || clip.inputs[k] == kNoTensor) continue;
    const Value* value = constant_input(graph, clip, k);
    if (value == nullptr || value->f.size() != 1) return std::nullopt;
    (k == 1 ? bounds.first : bounds.second) = value->f[0];
  }
  return bounds;
}

bool try_activation(Graph& graph, std::size_t n) {
  ConvAttrs* attrs = open_conv(graph.nodes[n], false);
  if (attrs == nullptr) return false;
  const TensorId y = graph.nodes[n].outputs[0];
  const auto reader = sole_reader(graph, y);
  if (!reader || graph.nodes[*reader].inputs[0] != y) return false;
  const Node& act = graph.nodes[*reader];
  Epilogue e;
  std::optional<TensorId> slope;
  switch (act.op) {
    case OpKind::Relu:
      e.act = Activation::Relu;
      break;
    case OpKind::LeakyRelu: {
      const auto* leaky = std::get_if<LeakyReluAttrs>(&act.attrs);
      if (leaky == nullptr) return false;
      e.act = Activation::LeakyRelu;
      e.alpha = leaky->alpha;
      break;
    }
    case OpKind::Sigmoid:
      e.act = Activation::Sigmoid;
      break;
    case OpKind::Tanh:
      e.act = Activation::Tanh;
      break;
    case OpKind::Clip: {
      const auto bounds = clip_bounds(graph, act);
      if (!bounds) return false;
      e.act = Activation::Clip;
      std::tie(e.lo, e.hi) = *bounds;
      break;
    }
    case OpKind::PRelu: {
      const Value* value = constant_input(graph, act, 1);
      const auto values = value != nullptr ? per_channel(*value, graph.tensors[y].dims[1]) : std::nullopt;
      if (!values) return false;
      e.act = Activation::PRelu;
      const auto channels = static_cast<std::int64_t>(values->size());
      slope = add_constant(graph, act.name + "/pente",
                           Value{.type = graph.tensors[y].type, .dims = {channels}, .f = *values});
      break;
    }
    default:
      return false;
  }
  Node& conv = graph.nodes[n];
  e.residual = attrs->epilogue.residual;
  attrs->epilogue = e;
  if (slope) {
    conv.inputs.resize(kConvSlope + 1, kNoTensor);
    conv.inputs[kConvSlope] = *slope;
  }
  conv.outputs[0] = graph.nodes[*reader].outputs[0];
  graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(*reader));
  return true;
}

// Porte : m = Mul(s, a), s = Add(x, r), a = Conv(x) de la forme de x, a et s lus par m seul.
bool try_gate(Graph& graph, std::size_t m) {
  const Node& mul = graph.nodes[m];
  if (mul.op != OpKind::Mul || mul.inputs.size() != 2 || mul.inputs[0] == mul.inputs[1]) return false;
  const auto producer = producers(graph);
  for (std::size_t side = 0; side < 2; ++side) {
    const TensorId a = mul.inputs[side];
    const TensorId s = mul.inputs[1 - side];
    const std::size_t conv_at = producer[a];
    const std::size_t add_at = producer[s];
    if (conv_at == kNoNode || add_at == kNoNode) continue;
    Node& conv = graph.nodes[conv_at];
    const Node& add = graph.nodes[add_at];
    auto* attrs = std::get_if<ConvAttrs>(&conv.attrs);
    if (conv.op != OpKind::Conv || attrs == nullptr || attrs->epilogue.residual || attrs->epilogue.gate ||
        add.op != OpKind::Add || add.inputs.size() != 2)
      continue;
    if (sole_reader(graph, a) != m || sole_reader(graph, s) != m) continue;
    const TensorId x = conv.inputs[kConvX];
    if (add.inputs[0] != x && add.inputs[1] != x) continue;
    const TensorId r = add.inputs[0] == x ? add.inputs[1] : add.inputs[0];
    const Tensor& xt = graph.tensors[x];
    const Tensor& out = graph.tensors[mul.outputs[0]];
    if (!xt.known || graph.tensors[a].dims != xt.dims || graph.tensors[r].dims != xt.dims ||
        out.dims != xt.dims || graph.tensors[r].type != xt.type || out.type != graph.tensors[a].type)
      continue;
    attrs->epilogue.gate = true;
    conv.inputs.resize(std::max(conv.inputs.size(), kConvResidual + 1), kNoTensor);
    conv.inputs[kConvResidual] = r;
    conv.outputs[0] = mul.outputs[0];
    // La Conv prend la place du Mul, après l'Add et elle-même : r y est calculé.
    graph.nodes[m] = std::move(conv);
    for (const std::size_t gone : {std::max(conv_at, add_at), std::min(conv_at, add_at)})
      graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(gone));
    return true;
  }
  return false;
}

// DepthToSpace d dont l'entrée vient d'une Conv qu'il lit seul : la Conv écrit sa sortie.
bool try_space(Graph& graph, std::size_t d) {
  const Node& space = graph.nodes[d];
  const auto* block = std::get_if<BlockAttrs>(&space.attrs);
  if (space.op != OpKind::DepthToSpace || block == nullptr || block->blocksize < 2 || space.inputs.empty())
    return false;
  const TensorId t = space.inputs[0];
  const std::size_t conv_at = producers(graph)[t];
  if (conv_at == kNoNode || sole_reader(graph, t) != d) return false;
  Node& conv = graph.nodes[conv_at];
  auto* attrs = std::get_if<ConvAttrs>(&conv.attrs);
  if (conv.op != OpKind::Conv || attrs == nullptr || attrs->epilogue.space != 0) return false;
  const Tensor& out = graph.tensors[space.outputs[0]];
  if (!graph.tensors[t].known || !out.known || out.type != graph.tensors[t].type) return false;
  attrs->epilogue.space = block->blocksize;
  attrs->epilogue.space_mode = block->mode;
  conv.outputs[0] = space.outputs[0];
  graph.nodes.erase(graph.nodes.begin() + static_cast<std::ptrdiff_t>(d));
  return true;
}

// Après une fusion, le nœud d'indice n a changé : on le réexamine.
template <class Fn>
int fuse_each(Graph& graph, Fn&& fuse) {
  int fused = 0;
  for (std::size_t n = 0; n < graph.nodes.size();) {
    if (fuse(graph, n)) {
      ++fused;
    } else {
      ++n;
    }
  }
  return fused;
}

}  // namespace

// Une Conv peut absorber plusieurs opérations de suite (Mul puis Add) : fuse_each la réexamine.
int fuse_conv_bias(Graph& graph) {
  return fuse_each(graph, try_bias);
}

int fuse_residuals(Graph& graph) {
  return fuse_each(graph, try_residual);
}

int fuse_activations(Graph& graph) {
  return fuse_each(graph, try_activation);
}

int fuse_spaces(Graph& graph) {
  return fuse_each(graph, try_space);
}

// Deux nœuds avant le Mul disparaissent à chaque fusion : on repart du début plutôt que d'en
// sauter un (fuse_each réexamine l'indice du nœud changé).
int fuse_gates(Graph& graph) {
  int fused = 0;
  for (std::size_t m = 0; m < graph.nodes.size();) {
    if (try_gate(graph, m)) {
      ++fused;
      m = 0;
    } else {
      ++m;
    }
  }
  return fused;
}

}  // namespace ajp::gpu::ir
