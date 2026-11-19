#include "prepare/gmflow_mlp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/ir/onnx_model.h"

namespace kirari::prepare {
namespace {

namespace onnx = gpu::ir::onnx;

constexpr int kLayers = 6;
constexpr int kScales = 2;  // le huitième, puis le quart (second appel du transformeur, suffixe « _1 »)
constexpr float kFp16Max = 65504.0F;
constexpr float kFp16Rounding = 1.0F / 2048.0F;  // l'écart relatif d'un arrondi FP16 au plus proche

std::unexpected<Refusal> missing(std::string detail) {
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing, "GMFlow natif : MLP non reconnu : " + std::move(detail)});
}

// Le graphe indexé : constantes (initialiseurs et sorties des nœuds Constant), nœuds par nom, producteur de
// chaque tenseur, lecteurs de chaque tenseur.
struct Model {
  std::map<std::string, const onnx::Tensor*, std::less<>> initializers;
  std::map<std::string, const onnx::Tensor*, std::less<>> constants;
  std::map<std::string, const onnx::Node*, std::less<>> nodes;
  std::map<std::string, const onnx::Node*, std::less<>> producers;
  std::map<std::string, std::vector<const onnx::Node*>, std::less<>> readers;
};

void index_model(const onnx::Model& parsed, Model& model) {
  for (const onnx::Tensor& tensor : parsed.graph.initializers) {
    model.initializers.emplace(tensor.name, &tensor);
    model.constants.emplace(tensor.name, &tensor);
  }
  for (const onnx::Node& node : parsed.graph.nodes) {
    model.nodes.emplace(node.name, &node);
    for (const std::string& out : node.outputs) model.producers.emplace(out, &node);
    for (const std::string& in : node.inputs) model.readers[in].push_back(&node);
    if (node.op_type != "Constant" || node.outputs.size() != 1) continue;
    for (const onnx::Attribute& a : node.attributes) {
      if (a.name == "value" && a.tensors.size() == 1)
        model.constants.emplace(node.outputs[0], a.tensors.data());
    }
  }
}

// Le nœud name s'il existe, fait op et n'a qu'une sortie ; sinon nul.
const onnx::Node* node_of(const Model& model, const std::string& name, std::string_view op) {
  const auto found = model.nodes.find(name);
  if (found == model.nodes.end() || found->second->op_type != op || found->second->outputs.size() != 1)
    return nullptr;
  return found->second;
}

const onnx::Attribute* attribute(const onnx::Node& node, std::string_view name) {
  const auto found = std::ranges::find(node.attributes, name, &onnx::Attribute::name);
  return found != node.attributes.end() ? &*found : nullptr;
}

// Un attribut entier : le sien, ou implicit s'il est absent.
std::int64_t int_attribute(const onnx::Node& node, std::string_view name, std::int64_t implicit) {
  const onnx::Attribute* a = attribute(node, name);
  return a != nullptr ? a->i : implicit;
}

// La constante name : une seule valeur, à tolerance près de want.
bool scalar_is(const Model& model, const std::string& name, float want, float tolerance) {
  const auto found = model.constants.find(name);
  return found != model.constants.end() && found->second->floats.size() == 1 &&
         std::abs(found->second->floats[0] - want) <= tolerance;
}

// Le tenseur name n'est-il lu que par ces nœuds, chacun une fois ?
bool only_read_by(const Model& model, const std::string& name, std::vector<const onnx::Node*> want) {
  const auto found = model.readers.find(name);
  if (found == model.readers.end()) return false;
  std::vector<const onnx::Node*> got = found->second;
  std::ranges::sort(got);
  std::ranges::sort(want);
  return got == want;
}

// Les nœuds d'un emploi : les deux MatMul, les cinq de la GELU, et les deux LayerNorm de la couche.
struct Nodes {
  const onnx::Node* mm1 = nullptr;
  const onnx::Node* div = nullptr;
  const onnx::Node* erf = nullptr;
  const onnx::Node* add = nullptr;
  const onnx::Node* mul = nullptr;
  const onnx::Node* mul1 = nullptr;
  const onnx::Node* mm2 = nullptr;
  const onnx::Node* norm1 = nullptr;
  const onnx::Node* norm2 = nullptr;
};

std::string prefix_of(int layer) {
  return std::format("/flownet/transformer/layers.{}/cross_attn_ffn/", layer);
}

std::expected<Nodes, Refusal> find_nodes(const Model& model, int layer, int scale) {
  const std::string p = prefix_of(layer);
  const std::string tail = scale == 0 ? "" : "_1";
  const std::string gelu = p + "mlp/mlp.1" + tail + "/";
  Nodes n;
  struct Want {
    const onnx::Node** slot;
    std::string name;
    std::string_view op;
  };
  const std::array<Want, 9> wants{
      {{&n.mm1, p + "mlp/mlp.0" + tail + "/MatMul", "MatMul"},
       {&n.div, gelu + "Div", "Div"},
       {&n.erf, gelu + "Erf", "Erf"},
       {&n.add, gelu + "Add", "Add"},
       {&n.mul, gelu + "Mul", "Mul"},
       {&n.mul1, gelu + "Mul_1", "Mul"},
       {&n.mm2, p + "mlp/mlp.2" + tail + "/MatMul", "MatMul"},
       {&n.norm1, p + "norm1" + tail + "/LayerNormalization", "LayerNormalization"},
       {&n.norm2, p + "norm2" + tail + "/LayerNormalization", "LayerNormalization"}}};
  for (const Want& w : wants) {
    *w.slot = node_of(model, w.name, w.op);
    if (*w.slot == nullptr) return missing(std::format("{} {}", w.op, w.name));
  }
  return n;
}

// La GELU exacte du graphe : Mul_1(Mul(h, Add(Erf(Div(h, √2)), 1)), 0,5), lue par le second MatMul. √2 à un
// arrondi FP16 près : la paire du lecteur est en FP16 (1,4140625), le noyau prend 1/√2 exact. Rien : bonne.
const char* gelu_problem(const Model& model, const Nodes& n) {
  const std::string& h = n.mm1->outputs[0];
  const float sqrt2 = std::sqrt(2.0F);
  if (n.div->inputs.size() != 2 || n.div->inputs[0] != h ||
      !scalar_is(model, n.div->inputs[1], sqrt2, sqrt2 * kFp16Rounding))
    return "Div par √2 de la GELU exacte";
  if (n.erf->inputs != std::vector{n.div->outputs[0]}) return "Erf";
  if (n.add->inputs.size() != 2 || n.add->inputs[0] != n.erf->outputs[0] ||
      !scalar_is(model, n.add->inputs[1], 1.0F, 0.0F))
    return "Add de 1 après Erf";
  if (n.mul->inputs != std::vector{h, n.add->outputs[0]}) return "Mul de la GELU";
  if (n.mul1->inputs.size() != 2 || n.mul1->inputs[0] != n.mul->outputs[0] ||
      !scalar_is(model, n.mul1->inputs[1], 0.5F, 0.0F))
    return "Mul par 0,5 de la GELU";
  if (n.mm2->inputs.size() != 2 || n.mm2->inputs[0] != n.mul1->outputs[0])
    return "le second MatMul ne lit pas la GELU";
  if (n.norm2->inputs.empty() || n.norm2->inputs[0] != n.mm2->outputs[0])
    return "norm2 ne lit pas le second MatMul (un biais ?)";
  return nullptr;
}

// La couche cachée et ce qui la suit ne sont lus que dans le MLP : le noyau ne les écrit jamais. Rien :
// bonne, sinon le nom du tenseur lu ailleurs.
std::string hidden_problem(const Model& model, const Nodes& n) {
  const std::array<std::pair<const onnx::Node*, std::vector<const onnx::Node*>>, 7> chain{{
      {n.mm1, {n.div, n.mul}},
      {n.div, {n.erf}},
      {n.erf, {n.add}},
      {n.add, {n.mul}},
      {n.mul, {n.mul1}},
      {n.mul1, {n.mm2}},
      {n.mm2, {n.norm2}},
  }};
  for (const auto& [from, to] : chain) {
    if (!only_read_by(model, from->outputs[0], to)) return from->outputs[0];
  }
  return {};
}

// Le Concat [x, message] sur le dernier axe, message la sortie de norm1, lu par le premier MatMul seul : la
// source x, ou vide.
std::string source_of(const Model& model, const Nodes& n) {
  const auto found = model.producers.find(n.mm1->inputs[0]);
  if (found == model.producers.end() || found->second->op_type != "Concat") return {};
  const onnx::Node& concat = *found->second;
  const onnx::Attribute* axis = attribute(concat, "axis");
  if (axis == nullptr || axis->i != -1 || concat.inputs.size() != 2 || concat.outputs.size() != 1 ||
      concat.inputs[1] != n.norm1->outputs[0] || concat.inputs[0] == concat.inputs[1] ||
      !only_read_by(model, concat.outputs[0], {n.mm1}))
    return {};
  return concat.inputs[0];
}

// L'Add du résiduel, seul lecteur de norm2 : source + LayerNorm ; sa sortie, ou vide.
std::string residual_of(const Model& model, const Nodes& n, const std::string& source) {
  const auto found = model.readers.find(n.norm2->outputs[0]);
  if (found == model.readers.end() || found->second.size() != 1) return {};
  const onnx::Node& add = *found->second.front();
  if (add.op_type != "Add" || add.inputs != std::vector{source, n.norm2->outputs[0]} ||
      add.outputs.size() != 1)
    return {};
  return add.outputs[0];
}

// La LayerNorm norm2 de la couche : dernier axe, statistiques en f32, échelle et décalage nommés de channels
// valeurs. Son ε, ou un refus.
std::expected<float, Refusal> read_norm(const Model& model, const onnx::Node& norm, int layer, int channels,
                                        GmflowMlpLayer* out) {
  const std::string module = std::format("gmfss.flownet.transformer.layers.{}.cross_attn_ffn.norm2.", layer);
  if (norm.inputs.size() != 3 || norm.inputs[1] != module + "weight" || norm.inputs[2] != module + "bias")
    return missing(norm.name + " : échelle et décalage " + module + "weight, bias attendus");
  if (int_attribute(norm, "axis", -1) != -1 || int_attribute(norm, "stash_type", 1) != 1)
    return missing(norm.name + " : axe ou type des statistiques");
  const auto gamma = model.initializers.find(norm.inputs[1]);
  const auto beta = model.initializers.find(norm.inputs[2]);
  const auto n = static_cast<std::size_t>(channels);
  if (gamma == model.initializers.end() || beta == model.initializers.end() ||
      gamma->second->floats.size() != n || beta->second->floats.size() != n)
    return missing(norm.name + std::format(" : {} canaux attendus", channels));
  if (out != nullptr) {
    out->gamma = gamma->second->floats;
    out->beta = beta->second->floats;
  }
  const onnx::Attribute* eps = attribute(norm, "epsilon");
  return eps != nullptr ? eps->f : 1e-5F;
}

// Le poids d'un MatMul : un initialiseur, ou un Identity (en chaîne) d'un initialiseur ; resolved reçoit le
// nom de l'initialiseur.
const onnx::Tensor* weight_of(const Model& model, std::string name, std::string& resolved) {
  for (int hops = 0; hops < 4; ++hops) {
    if (const auto init = model.initializers.find(name); init != model.initializers.end()) {
      resolved = name;
      return init->second;
    }
    const auto producer = model.producers.find(name);
    if (producer == model.producers.end() || producer->second->op_type != "Identity" ||
        producer->second->inputs.size() != 1)
      return nullptr;
    name = producer->second->inputs[0];
  }
  return nullptr;
}

// Les poids ONNX [rows][cols] transposés : [cols][rows].
std::vector<float> transpose(const std::vector<float>& from, std::size_t rows, std::size_t cols) {
  std::vector<float> out(from.size());
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c) out[c * rows + r] = from[r * cols + c];
  return out;
}

// Les deux poids d'un emploi : W1 [2C][hidden], W2 [hidden][C], et les noms de leurs initialiseurs.
struct Weights {
  const onnx::Tensor* w1 = nullptr;
  const onnx::Tensor* w2 = nullptr;
  std::array<std::string, 2> names;
};

std::expected<Weights, Refusal> read_weights(const Model& model, const Nodes& n, int channels, int& hidden) {
  Weights w;
  w.w1 = weight_of(model, n.mm1->inputs[1], w.names[0]);
  w.w2 = weight_of(model, n.mm2->inputs[1], w.names[1]);
  if (w.w1 == nullptr || w.w2 == nullptr) return missing(n.mm1->name + " : poids constants attendus");
  const std::vector<std::int64_t>& d1 = w.w1->dims;
  const std::vector<std::int64_t>& d2 = w.w2->dims;
  if (d1.size() != 2 || d1[0] != 2 * std::int64_t{channels} || d1[1] < 1 || d2.size() != 2 ||
      d2[0] != d1[1] || d2[1] != channels)
    return missing(n.mm1->name +
                   std::format(" : poids [{}][caché] puis [caché][{}] attendus", 2 * channels, channels));
  if (hidden != 0 && d1[1] != hidden) return missing(n.mm1->name + " : couche cachée d'une autre largeur");
  hidden = static_cast<int>(d1[1]);
  const auto h = static_cast<std::size_t>(hidden), c = static_cast<std::size_t>(channels);
  if (w.w1->floats.size() != 2 * c * h || w.w2->floats.size() != h * c)
    return missing(n.mm1->name + " : taille");
  return w;
}

// Un emploi lu : sa couche (à la première échelle) ou sa vérification (à la seconde, mêmes poids).
struct Reading {
  const Model& model;
  GmflowMlp& mlp;
  std::array<std::array<std::string, 2>, kLayers> names;  // les initialiseurs de W1 et W2 de chaque couche
};

std::expected<void, Refusal> read_use(Reading& r, int scale, int layer) {
  auto nodes = find_nodes(r.model, layer, scale);
  if (!nodes) return std::unexpected(std::move(nodes.error()));
  const Nodes& n = *nodes;
  const std::string where = n.mm1->name;
  if (const char* why = gelu_problem(r.model, n); why != nullptr)
    return missing(std::format("{} : {}", where, why));
  if (const std::string read = hidden_problem(r.model, n); !read.empty())
    return missing(std::format("{} : lu hors du MLP", read));
  const std::string source = source_of(r.model, n);
  if (source.empty()) return missing(where + " : Concat [x, message de norm1] sur le dernier axe attendu");
  const std::string output = residual_of(r.model, n, source);
  if (output.empty()) return missing(n.norm2->name + " : Add(x, norm2) du résiduel attendu");
  auto weights = read_weights(r.model, n, r.mlp.channels, r.mlp.hidden);
  if (!weights) return std::unexpected(std::move(weights.error()));
  const auto at = static_cast<std::size_t>(layer);
  GmflowMlpLayer fresh;
  auto eps = read_norm(r.model, *n.norm2, layer, r.mlp.channels, scale == 0 ? &fresh : nullptr);
  if (!eps) return std::unexpected(std::move(eps.error()));
  if (scale == 0) {
    const auto h = static_cast<std::size_t>(r.mlp.hidden), c = static_cast<std::size_t>(r.mlp.channels);
    fresh.w1 = transpose(weights->w1->floats, 2 * c, h);
    fresh.w2 = transpose(weights->w2->floats, h, c);
    fresh.epsilon = *eps;
    r.mlp.layers.push_back(std::move(fresh));
    r.names[at] = weights->names;
  } else if (weights->names != r.names[at] || *eps != r.mlp.layers[at].epsilon) {
    return missing(where + " : pas les poids ni l'ε du huitième");
  }
  r.mlp.uses.push_back(
      {.scale = scale, .layer = layer, .source = source, .message = n.norm1->outputs[0], .output = output});
  return {};
}

// Une couche se tient-elle ? Rien : oui.
const char* layer_problem(const GmflowMlp& mlp, const GmflowMlpLayer& l) {
  const auto c = static_cast<std::size_t>(mlp.channels), h = static_cast<std::size_t>(mlp.hidden);
  if (l.w1.size() != h * 2 * c || l.w2.size() != c * h) return "poids";
  if (l.gamma.size() != c || l.beta.size() != c) return "LayerNorm";
  if (!std::isfinite(l.epsilon) || !(l.epsilon > 0.0F)) return "ε";
  const auto fp16 = [](float v) { return std::isfinite(v) && std::abs(v) <= kFp16Max; };
  if (!std::ranges::all_of(l.w1, fp16) || !std::ranges::all_of(l.w2, fp16)) return "poids hors du FP16";
  const auto finite = [](float v) { return std::isfinite(v); };
  if (!std::ranges::all_of(l.gamma, finite) || !std::ranges::all_of(l.beta, finite))
    return "LayerNorm non finie";
  return nullptr;
}

}  // namespace

std::expected<GmflowMlp, Refusal> gmflow_mlp(std::span<const std::byte> pair) {
  auto parsed = onnx::read_onnx(pair);
  if (!parsed) return std::unexpected(Refusal{RefusalCode::ModelUnreadable, parsed.error().detail});
  Model model;
  index_model(*parsed, model);
  GmflowMlp mlp;
  Reading reading{.model = model, .mlp = mlp, .names = {}};
  for (int scale = 0; scale < kScales; ++scale)
    for (int layer = 0; layer < kLayers; ++layer) {
      if (auto ok = read_use(reading, scale, layer); !ok) return std::unexpected(std::move(ok.error()));
    }
  if (auto ok = gmflow_mlp_check(mlp); !ok) return std::unexpected(std::move(ok.error()));
  return mlp;
}

std::expected<void, Refusal> gmflow_mlp_check(const GmflowMlp& mlp) {
  if (mlp.channels != kGmflowMlpChannels)
    return missing(std::format("{} canaux attendus", kGmflowMlpChannels));
  if (mlp.hidden < kGmflowMlpHiddenTile || mlp.hidden % kGmflowMlpHiddenTile != 0 ||
      mlp.hidden > kGmflowMlpMaxHidden)
    return missing(std::format("couche cachée de {} : par {}, {} au plus", mlp.hidden, kGmflowMlpHiddenTile,
                               kGmflowMlpMaxHidden));
  if (mlp.layers.empty() || mlp.uses.empty()) return missing("aucune couche");
  for (std::size_t i = 0; i < mlp.layers.size(); ++i) {
    if (const char* why = layer_problem(mlp, mlp.layers[i]); why != nullptr)
      return missing(std::format("couche {} : {}", i, why));
  }
  std::set<std::pair<int, int>> seen;
  for (const GmflowMlpUse& use : mlp.uses) {
    if (use.layer < 0 || use.layer >= static_cast<int>(mlp.layers.size()) || use.scale < 0 ||
        use.scale >= kScales || !seen.emplace(use.scale, use.layer).second)
      return missing(
          std::format("emploi (échelle {}, couche {}) hors des couches, ou en double", use.scale, use.layer));
  }
  return {};
}

}  // namespace kirari::prepare
