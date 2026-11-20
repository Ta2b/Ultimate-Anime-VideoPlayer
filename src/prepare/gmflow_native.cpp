#include "prepare/gmflow_native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/ir/onnx_model.h"
#include "kernels/ref/half.h"

namespace kirari::prepare {
namespace {

namespace onnx = gpu::ir::onnx;

std::unexpected<Refusal> missing(std::string detail) {
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing, "GMFlow natif : squelette non reconnu : " + std::move(detail)});
}

constexpr std::string_view kNodes = "/flownet/backbone/";
constexpr std::string_view kWeights = "gmfss.flownet.backbone.";

struct Model {
  std::map<std::string, const onnx::Tensor*, std::less<>> constants;
  std::map<std::string, const onnx::Node*, std::less<>> nodes;
};

// Le nœud name s'il existe et fait op, sinon nul.
const onnx::Node* node_of(const Model& model, const std::string& name, std::string_view op) {
  const auto found = model.nodes.find(name);
  return found != model.nodes.end() && found->second->op_type == op ? found->second : nullptr;
}

const onnx::Attribute* attribute(const onnx::Node& node, std::string_view name) {
  const auto found = std::ranges::find(node.attributes, name, &onnx::Attribute::name);
  return found != node.attributes.end() ? &*found : nullptr;
}

// Un attribut INTS : celui du nœud, ou absent (implicit : la valeur par défaut d'ONNX qu'il prendrait).
bool ints_are(const onnx::Node& node, std::string_view name, const std::vector<std::int64_t>& want,
              const std::vector<std::int64_t>& implicit) {
  const onnx::Attribute* a = attribute(node, name);
  return a != nullptr ? a->ints == want : implicit == want;
}

// La Conv fait-elle une convolution carrée kernel × kernel de pas stride, de bord kernel / 2, sans groupes ni
// dilatation ? Rien : oui.
const char* conv_problem(const onnx::Node& node, int kernel, int stride) {
  const std::int64_t k = kernel, s = stride, pad = kernel / 2;
  if (const onnx::Attribute* shape = attribute(node, "kernel_shape");
      shape != nullptr && shape->ints != std::vector{k, k})
    return "kernel_shape";
  if (!ints_are(node, "strides", {s, s}, {1, 1})) return "strides";
  if (!ints_are(node, "pads", {pad, pad, pad, pad}, {0, 0, 0, 0})) return "pads";
  if (!ints_are(node, "dilations", {1, 1}, {1, 1})) return "dilations";
  if (const onnx::Attribute* group = attribute(node, "group"); group != nullptr && group->i != 1)
    return "group";
  if (const onnx::Attribute* pad_mode = attribute(node, "auto_pad");
      pad_mode != nullptr && pad_mode->s != "NOTSET")
    return "auto_pad";
  return nullptr;
}

// Les poids ONNX [cout][cin][k][k] rangés [cout][k²][cin].
std::vector<float> reorder(const std::vector<float>& from, int cout, int cin, int kernel) {
  const auto taps = static_cast<std::size_t>(kernel) * static_cast<std::size_t>(kernel);
  const auto in = static_cast<std::size_t>(cin);
  std::vector<float> out(from.size());
  for (std::size_t o = 0; o < static_cast<std::size_t>(cout); ++o)
    for (std::size_t c = 0; c < in; ++c)
      for (std::size_t t = 0; t < taps; ++t) out[(o * taps + t) * in + c] = from[(o * in + c) * taps + t];
  return out;
}

// La Conv node, de poids « <prefix><module>.weight » (et .bias si bias), qui lit input.
std::expected<GmflowConv, Refusal> read_conv(const Model& model, std::string_view node_name,
                                             std::string_view module, bool bias, int kernel, int stride,
                                             const std::string& input, std::string_view prefix = kWeights) {
  const onnx::Node* node = node_of(model, std::string(node_name), "Conv");
  const std::string weight_name = std::format("{}{}.weight", prefix, module);
  const std::string bias_name = std::format("{}{}.bias", prefix, module);
  if (node == nullptr) return missing(std::format("Conv {}", node_name));
  const std::vector<std::string> inputs =
      bias ? std::vector{input, weight_name, bias_name} : std::vector{input, weight_name};
  if (node->inputs != inputs || node->outputs.size() != 1)
    return missing(std::format("{} : entrées ou sorties inattendues", node_name));
  if (const char* why = conv_problem(*node, kernel, stride); why != nullptr)
    return missing(
        std::format("{} : {} hors de la {}×{} de pas {} attendue", node_name, why, kernel, kernel, stride));
  const auto weight = model.constants.find(weight_name);
  if (weight == model.constants.end()) return missing("poids " + weight_name);
  const std::vector<std::int64_t>& d = weight->second->dims;
  if (d.size() != 4 || d[2] != kernel || d[3] != kernel || d[0] < 1 || d[1] < 1)
    return missing(std::format("{} : poids {}×{} attendus", weight_name, kernel, kernel));
  GmflowConv conv{.kernel = kernel, .cin = static_cast<int>(d[1]), .cout = static_cast<int>(d[0])};
  const auto count = static_cast<std::size_t>(d[0] * d[1] * kernel * kernel);
  if (weight->second->floats.size() != count) return missing(weight_name + " : taille");
  conv.weights = reorder(weight->second->floats, conv.cout, conv.cin, kernel);
  conv.bias.assign(static_cast<std::size_t>(conv.cout), 0.0F);
  if (bias) {
    const auto b = model.constants.find(bias_name);
    if (b == model.constants.end() || b->second->floats.size() != static_cast<std::size_t>(conv.cout))
      return missing("biais " + bias_name);
    conv.bias = b->second->floats;
  } else if (model.constants.contains(bias_name)) {
    return missing(bias_name + " : biais inattendu");
  }
  return conv;
}

// Une constante de n valeurs, toutes égales à value si value est donné.
bool constant_is(const Model& model, const std::string& name, std::size_t n, const float* value) {
  const auto found = model.constants.find(name);
  if (found == model.constants.end() || found->second->floats.size() != n) return false;
  return value == nullptr || std::ranges::all_of(found->second->floats, [&](float v) { return v == *value; });
}

// L'InstanceNormalization node, sans paramètres (échelle 1, décalage 0), qui lit input : son ε.
std::expected<float, Refusal> read_norm(const Model& model, const std::string& node_name,
                                        const std::string& input, int channels) {
  const onnx::Node* node = node_of(model, node_name, "InstanceNormalization");
  if (node == nullptr) return missing("InstanceNormalization " + node_name);
  if (node->inputs.size() != 3 || node->inputs[0] != input || node->outputs.size() != 1)
    return missing(node_name + " : ne lit pas la convolution qui la précède");
  constexpr float kOne = 1.0F, kZero = 0.0F;
  const auto n = static_cast<std::size_t>(channels);
  if (!constant_is(model, node->inputs[1], n, &kOne) || !constant_is(model, node->inputs[2], n, &kZero))
    return missing(node_name + " : échelle ou décalage appris");
  const onnx::Attribute* eps = attribute(*node, "epsilon");
  return eps != nullptr ? eps->f : 1e-5F;
}

// Le nœud name, op, qui lit input en première entrée : sa sortie, ou vide.
std::string output_of(const Model& model, const std::string& name, std::string_view op,
                      const std::string& input) {
  const onnx::Node* node = node_of(model, name, op);
  if (node == nullptr || node->inputs.empty() || node->inputs[0] != input || node->outputs.size() != 1)
    return {};
  return node->outputs[0];
}

// La normalisation de l'image (/flownet/Sub et /flownet/Div, puis _1 pour la seconde image, mêmes
// constantes), et le Concat des deux qui entre dans le squelette : sa sortie, ou vide.
struct ImageNorm {
  std::array<float, 3> mean{};
  std::array<float, 3> deviation{};
};
std::string read_image(const Model& model, ImageNorm& b) {
  std::array<std::string, 2> divided;
  for (std::size_t side = 0; side < 2; ++side) {
    const std::string tail = side == 0 ? "" : "_1";
    const onnx::Node* sub = node_of(model, "/flownet/Sub" + tail, "Sub");
    const onnx::Node* div = node_of(model, "/flownet/Div" + tail, "Div");
    if (sub == nullptr || div == nullptr || sub->inputs.size() != 2 || div->inputs.size() != 2 ||
        sub->outputs.size() != 1 || div->inputs[0] != sub->outputs[0] ||
        !constant_is(model, sub->inputs[1], 3, nullptr) || !constant_is(model, div->inputs[1], 3, nullptr))
      return {};
    const std::vector<float>& mean = model.constants.find(sub->inputs[1])->second->floats;
    const std::vector<float>& deviation = model.constants.find(div->inputs[1])->second->floats;
    for (std::size_t c = 0; c < 3; ++c) {
      if (!(deviation[c] > 0.0F) || !std::isfinite(deviation[c]) || !std::isfinite(mean[c])) return {};
      if (side == 1 && (b.mean[c] != mean[c] || b.deviation[c] != deviation[c])) return {};
      b.mean[c] = mean[c];
      b.deviation[c] = deviation[c];
    }
    divided[side] = div->outputs.empty() ? std::string() : div->outputs[0];
  }
  const onnx::Node* concat = node_of(model, "/flownet/Concat", "Concat");
  const onnx::Attribute* axis = concat != nullptr ? attribute(*concat, "axis") : nullptr;
  if (concat == nullptr || concat->inputs != std::vector{divided[0], divided[1]} || axis == nullptr ||
      axis->i != 0 || concat->outputs.size() != 1)
    return {};
  return concat->outputs[0];
}

// La normalisation de l'image rangée dans la 7 × 7 (poids [cout][49][3]) : poids × 1/écart de leur plan,
// biais -Σ poids · pad, pad la moyenne arrondie en FP16 (la valeur exacte que les noyaux donnent aux pixels
// hors de l'image). Un pixel x donne alors poids · (x - pad) / écart ; l'écart de pad à la moyenne, constant
// par canal à l'intérieur, l'InstanceNorm qui suit l'efface.
void fold_image(const ImageNorm& norm, GmflowConv& stem, GmflowBackbone& b) {
  std::array<float, 3> inverse;
  for (std::size_t c = 0; c < inverse.size(); ++c) inverse[c] = 1.0F / norm.deviation[c];
  const std::size_t per_out = stem.weights.size() / static_cast<std::size_t>(stem.cout);
  for (std::size_t o = 0; o < static_cast<std::size_t>(stem.cout); ++o) {
    double bias = stem.bias[o];
    for (std::size_t i = 0; i < per_out; ++i) {
      float& w = stem.weights[o * per_out + i];
      w *= inverse[i % 3];
      bias -= static_cast<double>(w) * static_cast<double>(b.pad[i % 3]);
    }
    stem.bias[o] = static_cast<float>(bias);
  }
}

// Le plan en construction : les tenseurs, les convolutions et les opérations dans l'ordre du graphe.
class Planner {
 public:
  explicit Planner(GmflowBackbone& b) : b_(b) {}

  int add_conv(GmflowConv conv) {
    b_.convs.push_back(std::move(conv));
    return static_cast<int>(b_.convs.size()) - 1;
  }

  // La convolution conv au pas stride sur in ; normed : suivie d'une InstanceNorm (et d'une Relu, qui se
  // fait à la lecture suivante). Rend la lecture de ce qu'elle écrit.
  GmflowRead conv(int conv, int stride, GmflowRead in, bool normed) {
    const GmflowTensor from = b_.tensors[static_cast<std::size_t>(in.tensor)];
    const int out = tensor(from.level + (stride == 2 ? 1 : 0), b_.convs[static_cast<std::size_t>(conv)].cout);
    const int stats = normed ? b_.stats++ : -1;
    b_.ops.push_back(
        {.kind = GmflowOpKind::kConv, .conv = conv, .stride = stride, .in = in, .out = out, .stats = stats});
    return {.tensor = out, .stats = stats, .relu = normed};
  }

  // out = relu(skip + in), écrit à neuf.
  GmflowRead end(GmflowRead in, GmflowRead skip) {
    const GmflowTensor t = b_.tensors[static_cast<std::size_t>(in.tensor)];
    const int out = tensor(t.level, t.channels);
    b_.ops.push_back({.kind = GmflowOpKind::kBlockEnd, .in = in, .skip = skip, .out = out});
    return {.tensor = out};
  }

  int tensor(int level, int channels) {
    b_.tensors.push_back({.level = level, .channels = channels});
    return static_cast<int>(b_.tensors.size()) - 1;
  }

 private:
  GmflowBackbone& b_;
};

// Un bloc résiduel de CNNEncoder : layer<layer>.<index>, son pas, et s'il a un raccourci 1 × 1 (downsample).
struct BlockSpec {
  int layer = 1;
  int index = 0;
  int stride = 1;
  bool downsample = false;
};
constexpr std::array<BlockSpec, 6> kBlocks{{{1, 0, 1, false},
                                            {1, 1, 1, false},
                                            {2, 0, 2, true},
                                            {2, 1, 1, false},
                                            {3, 0, 1, true},
                                            {3, 1, 1, false}}};

// Lecture en cours : le plan, et le nom ONNX de la sortie courante (x) avec sa lecture dans le plan.
struct Reader {
  const Model& model;
  Planner plan;
  float epsilon = 0.0F;
  std::string x_name;
  GmflowRead x;
};

// Une convolution suivie de son InstanceNorm : la sortie ONNX de la norme, ou un refus.
std::expected<std::string, Refusal> conv_norm(Reader& r, const std::string& prefix,
                                              std::string_view conv_node, std::string_view norm_node,
                                              std::string_view module, bool bias, int kernel, int stride,
                                              const std::string& input, GmflowRead read, GmflowRead& out) {
  auto conv = read_conv(r.model, prefix + std::string(conv_node), module, bias, kernel, stride, input);
  if (!conv) return std::unexpected(std::move(conv.error()));
  const int cout = conv->cout;
  const std::string conv_out = r.model.nodes.find(prefix + std::string(conv_node))->second->outputs[0];
  const std::string norm_name = prefix + std::string(norm_node);
  auto eps = read_norm(r.model, norm_name, conv_out, cout);
  if (!eps) return std::unexpected(std::move(eps.error()));
  if (r.epsilon != 0.0F && *eps != r.epsilon) return missing(norm_name + " : ε différent des autres");
  r.epsilon = *eps;
  out = r.plan.conv(r.plan.add_conv(std::move(*conv)), stride, read, true);
  return r.model.nodes.find(norm_name)->second->outputs[0];
}

std::expected<void, Refusal> add_block(Reader& r, const BlockSpec& spec) {
  const std::string module = std::format("layer{}.{}", spec.layer, spec.index);
  const std::string prefix = std::format("{}layer{}/{}/", kNodes, spec.layer, module);
  GmflowRead y;
  auto n1 = conv_norm(r, prefix, "conv1/Conv", "norm1/InstanceNormalization", module + ".conv1", false, 3,
                      spec.stride, r.x_name, r.x, y);
  if (!n1) return std::unexpected(std::move(n1.error()));
  const std::string h1 = output_of(r.model, prefix + "relu/Relu", "Relu", *n1);
  if (h1.empty()) return missing(prefix + "relu/Relu");
  auto n2 = conv_norm(r, prefix, "conv2/Conv", "norm2/InstanceNormalization", module + ".conv2", false, 3, 1,
                      h1, y, y);
  if (!n2) return std::unexpected(std::move(n2.error()));
  const std::string h2 = output_of(r.model, prefix + "relu_1/Relu", "Relu", *n2);
  if (h2.empty()) return missing(prefix + "relu_1/Relu");
  std::string skip_name = r.x_name;
  GmflowRead skip = r.x;
  const std::string ds_module = module + ".downsample.0";
  if (spec.downsample) {
    auto n3 = conv_norm(r, prefix, "downsample/downsample.0/Conv", "downsample/norm3/InstanceNormalization",
                        ds_module, true, 1, spec.stride, r.x_name, r.x, skip);
    if (!n3) return std::unexpected(std::move(n3.error()));
    skip_name = *n3;
    skip.relu = false;  // le raccourci n'a pas de Relu
  } else if (r.model.constants.contains(std::format("{}{}.weight", kWeights, ds_module))) {
    return missing(module + " : raccourci inattendu");
  }
  const onnx::Node* add = node_of(r.model, prefix + "Add", "Add");
  if (add == nullptr || add->inputs != std::vector{skip_name, h2} || add->outputs.size() != 1)
    return missing(prefix + "Add : raccourci + bloc attendu");
  r.x_name = output_of(r.model, prefix + "relu_2/Relu", "Relu", add->outputs[0]);
  if (r.x_name.empty()) return missing(prefix + "relu_2/Relu");
  r.x = r.plan.end(y, skip);
  return {};
}

std::expected<void, Refusal> read_all(Reader& r, GmflowBackbone& b) {
  ImageNorm norm;
  const std::string image = read_image(r.model, norm);
  if (image.empty()) return missing("normalisation de l'image (/flownet/Sub, /flownet/Div, /flownet/Concat)");
  b.image = r.plan.tensor(0, 3);
  auto n = conv_norm(r, std::string(kNodes), "conv1/Conv", "norm1/InstanceNormalization", "conv1", false, 7,
                     2, image, {.tensor = b.image}, r.x);
  if (!n) return std::unexpected(std::move(n.error()));
  const auto as_half = [](float v) { return kernels::ref::half_to_float(kernels::ref::float_to_half(v)); };
  for (std::size_t c = 0; c < b.pad.size(); ++c) b.pad[c] = as_half(norm.mean[c]);
  if (b.stem_input == GmflowStemInput::kFolded) {
    fold_image(norm, b.convs.back(), b);
  } else {
    for (std::size_t c = 0; c < b.deviation.size(); ++c) b.deviation[c] = as_half(norm.deviation[c]);
  }
  r.x_name = output_of(r.model, std::string(kNodes) + "relu1/Relu", "Relu", *n);
  if (r.x_name.empty()) return missing("relu1");
  for (const BlockSpec& spec : kBlocks) {
    if (auto ok = add_block(r, spec); !ok) return ok;
  }
  auto conv2 = read_conv(r.model, std::string(kNodes) + "conv2/Conv", "conv2", true, 1, 1, r.x_name);
  if (!conv2) return std::unexpected(std::move(conv2.error()));
  const GmflowRead e = r.plan.conv(r.plan.add_conv(std::move(*conv2)), 1, r.x, false);
  const std::string e_name = r.model.nodes.find(std::string(kNodes) + "conv2/Conv")->second->outputs[0];
  auto quarter =
      read_conv(r.model, std::string(kNodes) + "trident_conv/Conv", "trident_conv", false, 3, 1, e_name);
  if (!quarter) return std::unexpected(std::move(quarter.error()));
  auto eighth =
      read_conv(r.model, std::string(kNodes) + "trident_conv/Conv_1", "trident_conv", false, 3, 2, e_name);
  if (!eighth) return std::unexpected(std::move(eighth.error()));
  const int trident = r.plan.add_conv(std::move(*quarter));  // les mêmes poids aux deux pas
  b.quarter = r.plan.conv(trident, 1, e, false).tensor;
  b.eighth = r.plan.conv(trident, 2, e, false).tensor;
  b.epsilon = r.epsilon;
  return {};
}

// Les tenseurs écrits et les statistiques faites, au fil des opérations.
struct Progress {
  std::vector<bool> written;
  std::vector<int> stats_of;  // le tenseur dont chaque statistique est faite ; -1 : pas encore
};

const char* read_problem(const GmflowBackbone& b, const Progress& p, const GmflowRead& r) {
  if (r.tensor < 0 || r.tensor >= static_cast<int>(b.tensors.size())) return "tenseur lu";
  if (!p.written[static_cast<std::size_t>(r.tensor)]) return "lit un tenseur pas encore écrit";
  if (r.stats >= b.stats || (r.stats >= 0 && p.stats_of[static_cast<std::size_t>(r.stats)] != r.tensor))
    return "statistiques pas encore faites, ou d'un autre tenseur";
  if (r.stats < 0 && r.relu) return "Relu sans InstanceNorm";
  return nullptr;
}

// Les poids et le biais de c ont-ils la taille de ses formes ?
bool weights_fit(const GmflowConv& c) {
  const auto taps = static_cast<std::size_t>(c.kernel) * static_cast<std::size_t>(c.kernel);
  return c.cin >= 1 &&
         c.weights.size() == static_cast<std::size_t>(c.cout) * taps * static_cast<std::size_t>(c.cin) &&
         c.bias.size() == static_cast<std::size_t>(c.cout);
}

const char* stem_problem(const GmflowBackbone& b, const GmflowOp& op, const GmflowConv& c) {
  if (op.in.tensor != b.image || op.stride != 2 || c.cin != 3 || op.in.stats >= 0 || c.cout % 32 != 0)
    return "la 7 × 7 ne lit que l'image, au pas 2, vers des canaux par 32";
  return nullptr;
}

const char* conv_op_problem(const GmflowBackbone& b, const GmflowOp& op) {
  if (op.conv < 0 || op.conv >= static_cast<int>(b.convs.size())) return "convolution";
  const GmflowConv& c = b.convs[static_cast<std::size_t>(op.conv)];
  const GmflowTensor& in = b.tensors[static_cast<std::size_t>(op.in.tensor)];
  const GmflowTensor& out = b.tensors[static_cast<std::size_t>(op.out)];
  if (!weights_fit(c)) return "poids";
  if (op.stride != 1 && op.stride != 2) return "pas";
  if (out.level != in.level + (op.stride == 2 ? 1 : 0) || out.channels != c.cout)
    return "taille ou canaux de sortie";
  if (c.cout % 16 != 0 || c.cout > kGmflowMaxChannels) return "canaux de sortie (par 16, 128 au plus)";
  if (c.kernel == 7) return stem_problem(b, op, c);
  if (op.in.tensor == b.image) return "l'image ne se lit qu'en 7 × 7";
  if (c.kernel != 3 && c.kernel != 1) return "noyau (7 × 7, 3 × 3 ou 1 × 1)";
  if (c.cin != in.channels || c.cin % 32 != 0) return "canaux d'entrée (par 32)";
  if (op.in.stats >= 0 && (c.kernel != 3 || op.stride != 1))
    return "normalisation à la lecture hors d'une 3 × 3 de pas 1";
  return nullptr;
}

const char* end_op_problem(const GmflowBackbone& b, const GmflowOp& op) {
  const GmflowTensor& in = b.tensors[static_cast<std::size_t>(op.in.tensor)];
  const GmflowTensor& skip = b.tensors[static_cast<std::size_t>(op.skip.tensor)];
  const GmflowTensor& out = b.tensors[static_cast<std::size_t>(op.out)];
  if (out.level < 1 || in.level != out.level || skip.level != out.level || in.channels != out.channels ||
      skip.channels != out.channels || op.stats >= 0)
    return "formes de la fin de bloc";
  return nullptr;
}

// Une opération : ses lectures, son écriture (un tenseur neuf), ses statistiques ; rien : bonne.
const char* op_problem(const GmflowBackbone& b, Progress& p, const GmflowOp& op) {
  const bool end = op.kind == GmflowOpKind::kBlockEnd;
  const char* why = read_problem(b, p, op.in);
  if (why == nullptr && end) why = read_problem(b, p, op.skip);
  if (why != nullptr) return why;
  if (op.out < 0 || op.out >= static_cast<int>(b.tensors.size()) || op.out == b.image ||
      p.written[static_cast<std::size_t>(op.out)])
    return "écrit un tenseur déjà écrit";
  if (op.stats >= b.stats || (op.stats >= 0 && p.stats_of[static_cast<std::size_t>(op.stats)] >= 0))
    return "statistiques déjà faites";
  why = end ? end_op_problem(b, op) : conv_op_problem(b, op);
  if (why != nullptr) return why;
  p.written[static_cast<std::size_t>(op.out)] = true;
  if (op.stats >= 0) p.stats_of[static_cast<std::size_t>(op.stats)] = op.out;
  return nullptr;
}

bool output_is(const GmflowBackbone& b, const Progress& p, int id, int level) {
  if (id < 0 || id >= static_cast<int>(b.tensors.size()) || !p.written[static_cast<std::size_t>(id)])
    return false;
  const GmflowTensor& t = b.tensors[static_cast<std::size_t>(id)];
  return t.level == level && t.channels == kGmflowMaxChannels;
}

Model index(const onnx::Graph& graph) {
  Model model;
  for (const onnx::Tensor& tensor : graph.initializers) model.constants.emplace(tensor.name, &tensor);
  for (const onnx::Node& node : graph.nodes) {
    model.nodes.emplace(node.name, &node);
    if (node.op_type != "Constant" || node.outputs.size() != 1) continue;
    for (const onnx::Attribute& a : node.attributes) {
      if (a.name == "value" && a.tensors.size() == 1)
        model.constants.emplace(node.outputs[0], a.tensors.data());
    }
  }
  return model;
}
}  // namespace

std::expected<GmflowBackbone, Refusal> gmflow_backbone(std::span<const std::byte> pair,
                                                       GmflowStemInput input) {
  auto parsed = onnx::read_onnx(pair);
  if (!parsed) return std::unexpected(Refusal{RefusalCode::ModelUnreadable, parsed.error().detail});
  const Model model = index(parsed->graph);
  GmflowBackbone backbone;
  backbone.stem_input = input;
  Reader reader{.model = model, .plan = Planner(backbone)};
  if (auto ok = read_all(reader, backbone); !ok) return std::unexpected(std::move(ok.error()));
  if (auto ok = gmflow_backbone_check(backbone); !ok) return std::unexpected(std::move(ok.error()));
  return backbone;
}

std::expected<GmflowUpsampler, Refusal> gmflow_upsampler(std::span<const std::byte> pair) {
  auto parsed = onnx::read_onnx(pair);
  if (!parsed) return std::unexpected(Refusal{RefusalCode::ModelUnreadable, parsed.error().detail});
  const Model model = index(parsed->graph);
  constexpr std::string_view kPrefix = "gmfss.flownet.upsampler.";
  const std::string first_name = "/flownet/upsampler/upsampler.0/Conv";
  const onnx::Node* first_node = node_of(model, first_name, "Conv");
  const onnx::Node* flow = node_of(model, "/flownet/feature_flow_attn_1/Transpose_6", "Transpose");
  const onnx::Node* features = node_of(model, "/flownet/transformer_1/Transpose_3", "Transpose");
  if (first_node == nullptr || first_node->inputs.empty() || flow == nullptr || flow->outputs.size() != 1 ||
      features == nullptr || features->outputs.size() != 1)
    return missing("upsampler : flot ou caractéristiques au quart");
  const std::string& in = first_node->inputs.front();
  const auto producer = std::ranges::find_if(
      parsed->graph.nodes, [&](const onnx::Node& node) { return node.outputs == std::vector{in}; });
  if (producer == parsed->graph.nodes.end() || producer->op_type != "Concat" ||
      producer->inputs != std::vector{flow->outputs.front(), features->outputs.front()})
    return missing("upsampler : concaténation flot xy puis caractéristiques");
  const onnx::Attribute* axis = attribute(*producer, "axis");
  if (axis == nullptr || axis->i != 1) return missing("upsampler : concaténation par canal");
  auto first = read_conv(model, first_name, "0", true, 3, 1, in, kPrefix);
  if (!first) return std::unexpected(std::move(first.error()));
  if (first->cin != 130 || first->cout != 256) return missing("upsampler.0 : 130 → 256 canaux");
  const std::string relu =
      output_of(model, "/flownet/upsampler/upsampler.1/Relu", "Relu", first_node->outputs.front());
  if (relu.empty()) return missing("upsampler.1 : Relu attendue");
  auto mask = read_conv(model, "/flownet/upsampler/upsampler.2/Conv", "2", true, 1, 1, relu, kPrefix);
  if (!mask) return std::unexpected(std::move(mask.error()));
  if (mask->cin != 256 || mask->cout != 144) return missing("upsampler.2 : 256 → 144 canaux");
  return GmflowUpsampler{.first = std::move(*first), .mask = std::move(*mask)};
}

std::expected<void, Refusal> gmflow_backbone_check(const GmflowBackbone& b) {
  if (b.image < 0 || b.image >= static_cast<int>(b.tensors.size()) ||
      b.tensors[static_cast<std::size_t>(b.image)].level != 0 ||
      b.tensors[static_cast<std::size_t>(b.image)].channels != 3)
    return missing("l'entrée n'est pas une image de trois plans");
  if (b.stem_input != GmflowStemInput::kFolded) {
    for (std::size_t c = 0; c < b.pad.size(); ++c) {
      if (!std::isfinite(b.pad[c]) || !std::isfinite(b.deviation[c]) || !(b.deviation[c] > 0.F))
        return missing("normalisation FP16 de l'image : moyenne ou écart");
    }
  }
  if (b.stats < 0 || !(b.epsilon > 0.0F)) return missing("InstanceNorm");
  Progress p{.written = std::vector<bool>(b.tensors.size(), false),
             .stats_of = std::vector<int>(static_cast<std::size_t>(b.stats), -1)};
  p.written[static_cast<std::size_t>(b.image)] = true;
  for (std::size_t i = 0; i < b.ops.size(); ++i) {
    if (const char* why = op_problem(b, p, b.ops[i]); why != nullptr)
      return missing(std::format("opération {} : {}", i, why));
  }
  if (std::ranges::find(p.stats_of, -1) != p.stats_of.end())
    return missing("une InstanceNorm n'est jamais faite");
  if (!output_is(b, p, b.quarter, 2) || !output_is(b, p, b.eighth, 3))
    return missing("les sorties ne sont pas 128 caractéristiques au quart et au huitième");
  return {};
}

}  // namespace kirari::prepare
