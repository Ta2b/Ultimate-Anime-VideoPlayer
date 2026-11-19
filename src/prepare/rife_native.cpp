#include "prepare/rife_native.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <map>
#include <string>
#include <string_view>
#include <utility>

#include "gpu/ir/onnx_model.h"

namespace kirari::prepare {
namespace {

namespace onnx = gpu::ir::onnx;

std::unexpected<Refusal> unknown(std::string detail) {
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing, "RIFE natif : morceau non reconnu : " + std::move(detail)});
}

const onnx::Attribute* find_attribute(const onnx::Node& node, std::string_view name) {
  const auto found = std::ranges::find(node.attributes, name, &onnx::Attribute::name);
  return found == node.attributes.end() ? nullptr : &*found;
}

// Les entiers d'un attribut INTS, ou fallback s'il manque.
std::vector<std::int64_t> ints_of(const onnx::Node& node, std::string_view name,
                                  std::vector<std::int64_t> fallback) {
  const onnx::Attribute* found = find_attribute(node, name);
  return found != nullptr ? found->ints : std::move(fallback);
}

bool all_equal(const std::vector<std::int64_t>& values, std::int64_t want, std::size_t count) {
  return values.size() == count && std::ranges::all_of(values, [want](std::int64_t v) { return v == want; });
}

// Où en est la dernière couche : convolution seule, mise à l'échelle, addition résiduelle, activation.
enum class Stage : std::uint8_t { Conv, Scaled, Added, Activated };

struct Walk {
  std::map<std::string, const onnx::Tensor*, std::less<>> constants;
  std::map<std::string, std::vector<const onnx::Node*>, std::less<>> readers;
  std::vector<RifeNativeLayer> layers;
  std::vector<std::string> layer_input;  // le tenseur lu par chaque couche (le résiduel l'ajoute)
  Stage stage = Stage::Activated;

  const onnx::Tensor* constant(const std::string& name) const {
    const auto found = constants.find(name);
    return found != constants.end() ? found->second : nullptr;
  }
};

// Valeurs par canal d'une constante : cout valeurs, ou une seule répétée (formes 1, C, 1×C×1×1, C×1×1).
std::vector<float> per_channel(const onnx::Tensor& tensor, int cout) {
  if (tensor.floats.size() == 1) return std::vector<float>(static_cast<std::size_t>(cout), tensor.floats[0]);
  if (tensor.floats.size() == static_cast<std::size_t>(cout)) return tensor.floats;
  return {};
}

// Les attributs d'une Conv 3 × 3 (ou 1 × 1, kernel 1) ou d'une ConvTranspose 4 × 4 que les noyaux font : un
// seul groupe, ou autant que de canaux (par canal, groups), bord 1 (0 pour 1 × 1), sans dilatation ni
// auto_pad, pas 1 ou 2 (2 pour la transposée, sans output_padding ; 1 pour 1 × 1). Le pas, ou 0.
int conv_stride(const onnx::Node& node, bool transposed, int kernel, std::int64_t groups) {
  const onnx::Attribute* group = find_attribute(node, "group");
  const onnx::Attribute* auto_pad = find_attribute(node, "auto_pad");
  const std::int64_t g = group != nullptr ? group->i : 1;
  if ((g != 1 && (g != groups || transposed || kernel != 3)) ||
      (auto_pad != nullptr && auto_pad->s != "NOTSET"))
    return 0;
  const std::vector<std::int64_t> strides = ints_of(node, "strides", {1, 1});
  if (!all_equal(ints_of(node, "pads", {0, 0, 0, 0}), kernel == 1 ? 0 : 1, 4) ||
      !all_equal(ints_of(node, "dilations", {1, 1}), 1, 2) || strides.size() != 2 || strides[0] != strides[1])
    return 0;
  const auto stride = static_cast<int>(strides[0]);
  if (transposed) return stride == 2 && all_equal(ints_of(node, "output_padding", {0, 0}), 0, 2) ? 2 : 0;
  if (kernel == 1) return stride == 1 ? 1 : 0;
  return stride == 1 || stride == 2 ? stride : 0;
}

// Les poids ONNX ([cout][cin][k][k], ou [cin][cout][k][k] pour la transposée) rangés [cout][k²][cin].
std::vector<float> reorder(const std::vector<float>& from, const RifeNativeLayer& layer) {
  const auto taps = static_cast<std::size_t>(layer.kernel) * static_cast<std::size_t>(layer.kernel);
  const auto cin = static_cast<std::size_t>(layer.cin);
  const auto cout = static_cast<std::size_t>(layer.cout);
  std::vector<float> out(from.size());
  for (std::size_t o = 0; o < cout; ++o)
    for (std::size_t c = 0; c < cin; ++c)
      for (std::size_t t = 0; t < taps; ++t) {
        const std::size_t at = layer.transposed ? (c * cout + o) * taps + t : (o * cin + c) * taps + t;
        out[(o * taps + t) * cin + c] = from[at];
      }
  return out;
}

// Une Conv 1 × 1 (rangée en 3 × 3, sa seule prise au centre) ou par canal (rangée en 3 × 3 diagonale), en
// couche de [cout][9][cin] ; sinon les poids tels quels.
std::vector<float> expand(const onnx::Tensor& weight, const RifeNativeLayer& layer, bool depthwise) {
  const auto cin = static_cast<std::size_t>(layer.cin);
  const auto cout = static_cast<std::size_t>(layer.cout);
  if (!depthwise && weight.dims[2] == layer.kernel) return reorder(weight.floats, layer);
  std::vector<float> out(cout * 9 * cin, 0.0F);
  for (std::size_t o = 0; o < cout; ++o) {
    if (depthwise) {
      for (std::size_t t = 0; t < 9; ++t) out[(o * 9 + t) * cin + o] = weight.floats[o * 9 + t];
    } else {
      for (std::size_t c = 0; c < cin; ++c) out[(o * 9 + 4) * cin + c] = weight.floats[o * cin + c];
    }
  }
  return out;
}

// Une Conv 1 × 1 qui suit une convolution sans rien entre elles (la convolution séparable : par canal puis
// 1 × 1) : repliée dans la couche d'avant, W'[o] = Σ_m P[o][m] W[m], b'[o] = Σ_m P[o][m] b[m] + c[o].
// Calcul exact en réels ; les noyaux voient une seule convolution 3 × 3.
std::expected<void, Refusal> fold_pointwise(Walk& walk, const onnx::Node& node, const onnx::Tensor& weight,
                                            const std::vector<float>& bias) {
  RifeNativeLayer& last = walk.layers.back();
  const auto mid = static_cast<std::size_t>(last.cout);
  const auto cout = static_cast<std::size_t>(weight.dims[0]);
  if (last.transposed || static_cast<std::size_t>(weight.dims[1]) != mid)
    return unknown(node.name + " : 1 × 1 qui ne suit pas une convolution 3 × 3");
  const auto row = static_cast<std::size_t>(9 * last.cin);
  std::vector<float> folded(cout * row);
  std::vector<float> folded_bias(cout);
  for (std::size_t o = 0; o < cout; ++o) {
    double b = bias[o];
    for (std::size_t m = 0; m < mid; ++m) b += static_cast<double>(weight.floats[o * mid + m]) * last.bias[m];
    folded_bias[o] = static_cast<float>(b);
    for (std::size_t i = 0; i < row; ++i) {
      double sum = 0.0;
      for (std::size_t m = 0; m < mid; ++m)
        sum += static_cast<double>(weight.floats[o * mid + m]) * last.weights[m * row + i];
      folded[o * row + i] = static_cast<float>(sum);
    }
  }
  last.cout = static_cast<int>(cout);
  last.weights = std::move(folded);
  last.bias = std::move(folded_bias);
  return {};  // stage reste Conv : l'échelle, le résiduel et l'activation suivent
}

// Le biais d'une convolution (cout valeurs), ou des zéros sans biais.
std::expected<std::vector<float>, Refusal> conv_bias(const Walk& walk, const onnx::Node& node,
                                                     std::size_t cout) {
  if (node.inputs.size() < 3 || node.inputs[2].empty()) return std::vector<float>(cout, 0.0F);
  const onnx::Tensor* b = walk.constant(node.inputs[2]);
  if (b == nullptr || b->floats.size() != cout) return unknown(node.name + " : biais");
  return b->floats;
}

// La forme des poids : 1 × 1 (pointwise), par canal ([C][1][3][3], autant de groupes que de canaux), ou
// pleine ; le pas et les canaux remplis dans layer. Rend la taille du noyau lu (1, 3 ou 4).
std::expected<int, Refusal> conv_shape(const onnx::Node& node, const onnx::Tensor* weight,
                                       RifeNativeLayer& layer, bool& depthwise) {
  if (weight == nullptr || weight->dims.size() != 4)
    return unknown(node.name + std::format(" : poids constants {0}×{0} attendus", layer.kernel));
  const std::vector<std::int64_t>& d = weight->dims;
  const bool pointwise = !layer.transposed && d[2] == 1 && d[3] == 1;
  const int kernel = pointwise ? 1 : layer.kernel;
  if (d[2] != kernel || d[3] != kernel)
    return unknown(node.name + std::format(" : poids constants {0}×{0} attendus", layer.kernel));
  // Par canal : une entrée par sortie, et autant de groupes que de sorties (sans group, une pleine à une
  // entrée).
  const onnx::Attribute* group = find_attribute(node, "group");
  depthwise =
      !layer.transposed && !pointwise && d[1] == 1 && d[0] > 1 && group != nullptr && group->i == d[0];
  layer.stride = conv_stride(node, layer.transposed, kernel, depthwise ? d[0] : 1);
  if (layer.stride == 0) return unknown(node.name + " : groupes, bords, dilatation ou pas");
  layer.cout = static_cast<int>(layer.transposed ? d[1] : d[0]);
  layer.cin = static_cast<int>(depthwise || layer.transposed ? d[0] : d[1]);
  const std::size_t per_out = depthwise ? 1U : static_cast<std::size_t>(layer.transposed ? d[0] : d[1]);
  if (layer.cin < 1 || layer.cout < 1 ||
      weight->floats.size() !=
          per_out * static_cast<std::size_t>(layer.cout) * static_cast<std::size_t>(kernel * kernel))
    return unknown(node.name + " : taille des poids");
  return kernel;
}

std::expected<void, Refusal> add_conv(Walk& walk, const onnx::Node& node, const std::string& input) {
  RifeNativeLayer layer;
  layer.transposed = node.op_type == "ConvTranspose";
  layer.kernel = layer.transposed ? 4 : 3;
  const onnx::Tensor* weight = node.inputs.size() >= 2 ? walk.constant(node.inputs[1]) : nullptr;
  bool depthwise = false;
  const auto kernel = conv_shape(node, weight, layer, depthwise);
  if (!kernel) return std::unexpected(std::move(kernel.error()));
  auto bias = conv_bias(walk, node, static_cast<std::size_t>(layer.cout));
  if (!bias) return std::unexpected(std::move(bias.error()));
  if (*kernel == 1 && walk.stage == Stage::Conv && !walk.layers.empty())
    return fold_pointwise(walk, node, *weight, *bias);
  layer.weights = expand(*weight, layer, depthwise);
  layer.bias = std::move(*bias);
  walk.layers.push_back(std::move(layer));
  walk.layer_input.push_back(input);
  walk.stage = Stage::Conv;
  return {};
}

// Une activation après la convolution (ou après son addition résiduelle).
std::expected<void, Refusal> add_activation(Walk& walk, const onnx::Node& node) {
  RifeNativeLayer& last = walk.layers.back();
  const bool after_add = walk.stage == Stage::Added;
  if (!(walk.stage == Stage::Conv || after_add)) return unknown(node.name + " : activation mal placée");
  if (node.op_type == "LeakyRelu") {
    const onnx::Attribute* alpha = find_attribute(node, "alpha");
    last.slope.assign(static_cast<std::size_t>(last.cout), alpha != nullptr ? alpha->f : 0.01F);
  } else if (node.op_type == "Relu") {
    last.slope.assign(static_cast<std::size_t>(last.cout), 0.0F);
  } else {  // PRelu
    const onnx::Tensor* slope = node.inputs.size() == 2 ? walk.constant(node.inputs[1]) : nullptr;
    last.slope = slope != nullptr ? per_channel(*slope, last.cout) : std::vector<float>{};
    if (last.slope.empty()) return unknown(node.name + " : pente de PRelu");
  }
  walk.stage = Stage::Activated;
  return {};
}

// Mul par une constante par canal (le bêta d'un ResConv), puis Add de l'entrée de la convolution.
std::expected<void, Refusal> add_residual_part(Walk& walk, const onnx::Node& node,
                                               const std::string& current) {
  RifeNativeLayer& last = walk.layers.back();
  if (node.inputs.size() != 2) return unknown(node.name + " : deux entrées attendues");
  const std::string& other = node.inputs[0] == current ? node.inputs[1] : node.inputs[0];
  if (node.op_type == "Mul") {
    const onnx::Tensor* factor = walk.constant(other);
    if (walk.stage != Stage::Conv || factor == nullptr) return unknown(node.name + " : Mul mal placé");
    last.scale = per_channel(*factor, last.cout);
    if (last.scale.empty()) return unknown(node.name + " : facteur du Mul");
    walk.stage = Stage::Scaled;
    return {};
  }
  if ((walk.stage != Stage::Conv && walk.stage != Stage::Scaled) || other != walk.layer_input.back() ||
      last.cin != last.cout)
    return unknown(node.name + " : Add qui n'est pas le résiduel de la convolution");
  if (last.scale.empty()) last.scale.assign(static_cast<std::size_t>(last.cout), 1.0F);
  walk.stage = Stage::Added;
  return {};
}

// Les constantes (initialiseurs et nœuds Constant) et les lecteurs de chaque tenseur calculé.
void index_graph(const onnx::Graph& graph, Walk& walk) {
  for (const onnx::Tensor& t : graph.initializers) walk.constants.emplace(t.name, &t);
  for (const onnx::Node& node : graph.nodes) {
    if (node.op_type == "Constant" && node.outputs.size() == 1) {
      const onnx::Attribute* value = find_attribute(node, "value");
      if (value != nullptr && value->tensors.size() == 1)
        walk.constants.emplace(node.outputs[0], &value->tensors[0]);
      continue;
    }
    for (const std::string& input : node.inputs) {
      if (!input.empty() && !walk.constants.contains(input)) walk.readers[input].push_back(&node);
    }
  }
}

// Le nœud qui suit current dans la chaîne : son seul lecteur, ou la convolution d'un ResConv dont l'addition
// est l'autre lecteur. Nul sinon.
const onnx::Node* next_node(const Walk& walk, const std::string& current) {
  const auto found = walk.readers.find(current);
  if (found == walk.readers.end()) return nullptr;
  const std::vector<const onnx::Node*>& next = found->second;
  if (next.size() == 1) return next[0];
  if (next.size() != 2) return nullptr;
  if (next[0]->op_type == "Conv" && next[1]->op_type == "Add") return next[0];
  if (next[1]->op_type == "Conv" && next[0]->op_type == "Add") return next[1];
  return nullptr;
}

std::expected<void, Refusal> step(Walk& walk, const onnx::Node& node, const std::string& current) {
  const std::string& op = node.op_type;
  if (op == "Conv" || op == "ConvTranspose") {
    if (walk.stage == Stage::Scaled || walk.stage == Stage::Added)
      return unknown(node.name + " : résiduel sans activation");
    return add_conv(walk, node, current);
  }
  if (op == "Cast" || op == "Identity") return {};
  if (walk.layers.empty()) return unknown(node.name + " : " + op + " avant la première convolution");
  if (op == "LeakyRelu" || op == "Relu" || op == "PRelu") return add_activation(walk, node);
  if (op == "Mul" || op == "Add") return add_residual_part(walk, node, current);
  return unknown(node.name + " : opérateur " + op);
}

}  // namespace

std::expected<std::vector<RifeNativeLayer>, Refusal> rife_native_layers(std::span<const std::byte> piece) {
  auto model = onnx::read_onnx(piece);
  if (!model) return std::unexpected(std::move(model.error()));
  const onnx::Graph& graph = model->graph;
  Walk walk;
  index_graph(graph, walk);
  std::vector<std::string> inputs;
  for (const onnx::ValueInfo& info : graph.inputs) {
    if (!walk.constants.contains(info.name)) inputs.push_back(info.name);
  }
  if (inputs.size() != 1 || graph.outputs.size() != 1) return unknown("une entrée et une sortie attendues");
  std::string current = inputs[0];
  for (std::size_t steps = 0; current != graph.outputs[0].name; ++steps) {
    const onnx::Node* node = next_node(walk, current);
    if (node == nullptr || node->outputs.size() != 1 || steps > graph.nodes.size())
      return unknown("chaîne coupée ou embranchement après " + current);
    if (!onnx::is_default_domain(node->domain)) return unknown(node->name + " : domaine");
    if (auto ok = step(walk, *node, current); !ok) return std::unexpected(std::move(ok.error()));
    current = node->outputs[0];
  }
  // Un Mul non suivi de son Add, ou un résiduel sans activation, à la fin du morceau.
  if (walk.stage == Stage::Scaled || walk.stage == Stage::Added)
    return unknown("résiduel inachevé à la fin du morceau");
  return std::move(walk.layers);
}

std::expected<void, Refusal> rife_native_check(std::span<const RifeNativeLayer> layers, int in_channels,
                                               int out_channels) {
  if (layers.size() < 2) return unknown("au moins une convolution et la transposée");
  int channels = in_channels;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const RifeNativeLayer& layer = layers[i];
    const bool last = i + 1 == layers.size();
    if (layer.cin != channels)
      return unknown(std::format("couche {} : {} canaux d'entrée au lieu de {}", i, layer.cin, channels));
    if (layer.transposed != last)
      return unknown(std::format("couche {} : la transposée n'est qu'à la fin", i));
    if (last && (layer.cout != out_channels || !layer.slope.empty() || !layer.scale.empty()))
      return unknown(std::format("transposée à {} canaux sans activation attendue", out_channels));
    if (!layer.scale.empty() && layer.cin != layer.cout)
      return unknown(std::format("couche {} : résiduel", i));
    channels = layer.cout;
  }
  return {};
}

std::expected<void, Refusal> conv_chain_check(std::span<const RifeNativeLayer> layers, int in_channels,
                                              int out_channels) {
  if (layers.empty()) return unknown("au moins une convolution");
  int channels = in_channels;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const RifeNativeLayer& layer = layers[i];
    if (layer.cin != channels)
      return unknown(std::format("couche {} : {} canaux d'entrée au lieu de {}", i, layer.cin, channels));
    if (layer.transposed) return unknown(std::format("couche {} : transposée", i));
    if (!layer.scale.empty() && layer.cin != layer.cout)
      return unknown(std::format("couche {} : résiduel", i));
    channels = layer.cout;
  }
  if (channels != out_channels)
    return unknown(std::format("{} canaux de sortie au lieu de {}", channels, out_channels));
  return {};
}

}  // namespace kirari::prepare
