#include "prepare/gmflow_attention_native.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/ir/onnx_model.h"
#include "prepare/gmflow_onnx_view.h"

namespace kirari::prepare {
namespace {

namespace onnx = gpu::ir::onnx;
using Node = onnx::Node;

std::unexpected<Refusal> missing(std::string detail) {
  return std::unexpected(
      Refusal{RefusalCode::FieldMissing, "GMFlow natif : attention non reconnue : " + std::move(detail)});
}

constexpr std::string_view kLayers = "/flownet/transformer/layers.";
constexpr std::string_view kWeights = "gmfss.flownet.transformer.layers.";
constexpr int kD = kGmflowModel;

// Une échelle du transformeur : le suffixe de ses nœuds (vide au huitième, « _1 » au quart), ses fenêtres par
// côté, ses décalages (la base des tenseurs du décalage en hauteur, axe 1, puis en largeur, axe 2) et son
// masque.
struct Scale {
  std::string suffix;
  std::string top;
  int splits = 0;
  std::array<std::string, 2> shift;
  std::string mask;
};

// L'entier constant de la seconde entrée d'un nœud à deux entrées.
std::optional<std::int64_t> operand(const OnnxView& g, const Node* n) {
  if (n == nullptr || n->inputs.size() != 2) return std::nullopt;
  return g.integer(n->inputs[1]);
}

bool ints_are(const OnnxView& g, std::string_view tensor, const std::vector<std::int64_t>& want) {
  const onnx::Tensor* t = g.constant(g.base(tensor));
  return t != nullptr && t->ints == want;
}

// Le masque : régions de l'image décalée (ligne · 3 + colonne, chacune comptée par deux comparaisons au bord
// moins une fenêtre et au bord moins un décalage), différence de deux régions, value où elle n'est pas nulle,
// 0 ailleurs.
const char* mask_problem(const OnnxView& g, const Scale& s, std::array<std::string, 2> window, float& value) {
  const Node* where = g.node(s.top + "Where", "Where");
  const Node* where1 = g.node(s.top + "Where_1", "Where");
  const Node* equal = g.node(s.top + "Equal", "Equal");
  const Node* equal1 = g.node(s.top + "Equal_1", "Equal");
  const Node* rows = g.node(s.top + "Mul", "Mul");
  if (where == nullptr || where1 == nullptr || equal == nullptr || equal1 == nullptr || rows == nullptr ||
      where->inputs.size() != 3 || where1->inputs.size() != 3 || where1->inputs[2] != where->outputs[0])
    return "Where du masque";
  const auto fill = g.scalar(where->inputs[1]);
  if (!fill || !(*fill < 0.F) || !std::isfinite(*fill) || g.scalar(where1->inputs[1]) != 0.F ||
      g.scalar(equal->inputs[1]) != 0.F || g.scalar(equal1->inputs[1]) != 0.F || operand(g, rows) != 3)
    return "valeurs du masque (-100 entre régions, 0 dans une région, trois régions par ligne)";
  for (int i = 0; i < 4; ++i) {
    const std::string name = s.top + (i == 0 ? "GreaterOrEqual" : std::format("GreaterOrEqual_{}", i));
    const Node* ge = g.node(name, "GreaterOrEqual");
    const Node* edge = ge != nullptr && ge->inputs.size() == 2 ? g.made_by(ge->inputs[1], "Sub") : nullptr;
    const std::string& want =
        (i % 2 == 0) ? window[static_cast<std::size_t>(i / 2)] : s.shift[static_cast<std::size_t>(i / 2)];
    if (edge == nullptr || edge->inputs.size() != 2 || g.base(edge->inputs[1]) != want)
      return "régions du masque (bord moins une fenêtre, bord moins un décalage)";
  }
  value = *fill;
  return nullptr;
}

std::expected<Scale, Refusal> read_scale(const OnnxView& g, int index, float& mask) {
  Scale s{.suffix = index == 0 ? "" : "_1",
          .top = index == 0 ? "/flownet/transformer/" : "/flownet/transformer_1/"};
  const Node* split_h = g.node(s.top + "Div", "Div");
  const Node* split_w = g.node(s.top + "Div_1", "Div");
  const auto sh = operand(g, split_h), sw = operand(g, split_w);
  if (!sh || !sw || *sh != *sw || *sh < 1 || *sh > 64)
    return missing(s.top + "Div : autant de fenêtres par côté en hauteur et en largeur");
  s.splits = static_cast<int>(*sh);
  const std::array<std::string, 2> window{split_h->outputs[0], split_w->outputs[0]};
  for (std::size_t axis = 0; axis < 2; ++axis) {
    const Node* half = g.node(s.top + (axis == 0 ? "Div_2" : "Div_3"), "Div");
    if (operand(g, half) != 2 || g.base(half->inputs[0]) != window[axis])
      return missing(s.top + "Div_2, Div_3 : décalage d'une demi-fenêtre");
    s.shift[axis] = half->outputs[0];
  }
  float value = 0.F;
  if (const char* why = mask_problem(g, s, window, value); why != nullptr) return missing(s.top + why);
  if (index == 1 && value != mask) return missing(s.top + " : masque différent du huitième");
  mask = value;
  s.mask = g.node(s.top + "Where_1", "Where")->outputs[0];
  return s;
}

// Un Concat à deux entrées sur l'axe axis (-1 : 1 ou 2, rendu dans axis).
bool concat_on(const Node* n, std::int64_t& axis) {
  const onnx::Attribute* a = n != nullptr && n->op_type == "Concat" ? onnx_attribute(*n, "axis") : nullptr;
  if (a == nullptr || n->inputs.size() != 2) return false;
  if (axis == -1 && (a->i == 1 || a->i == 2)) axis = a->i;
  return a->i == axis;
}

// Deux Slice de la même donnée data sur l'axe axis, le second depuis 0 : ceux d'un torch.roll ou de chunk(2).
bool two_slices(const OnnxView& g, const Node* a, const Node* b, std::string_view data, std::int64_t axis) {
  return a != nullptr && b != nullptr && a->op_type == "Slice" && b->op_type == "Slice" &&
         a->inputs.size() == 4 && b->inputs.size() == 4 && a->inputs[0] == data && b->inputs[0] == data &&
         ints_are(g, a->inputs[3], {axis}) && ints_are(g, b->inputs[3], {axis}) &&
         ints_are(g, b->inputs[1], {0});
}

// tensor = Add(…) dont une entrée est what (et l'autre 0 si zero).
bool sum_of(const OnnxView& g, std::string_view tensor, std::string_view what, bool zero) {
  const Node* add = g.made_by(tensor, "Add");
  if (add == nullptr || add->inputs.size() != 2) return false;
  const bool holds = g.base(add->inputs[0]) == what || g.base(add->inputs[1]) == what;
  return holds && (!zero || g.integer(add->inputs[0]) == 0 || g.integer(add->inputs[1]) == 0);
}

// Un décalage de Swin : Concat sur l'axe 1 ou 2 de Slice(x, a, a + reste) et Slice(x, 0, 0 + a), a = le
// décalage (à gauche, avant l'attention) ou la taille moins le décalage (à droite, après). Rend x et l'axe.
const char* roll_problem(const OnnxView& g, const Scale& s, const Node& concat, bool right, std::string& data,
                         int& axis) {
  std::int64_t ax = -1;
  if (!concat_on(&concat, ax)) return "Concat du décalage";
  const Node* a = g.producer(concat.inputs[0]);
  const Node* b = g.producer(concat.inputs[1]);
  if (a == nullptr || !two_slices(g, a, b, a->inputs.empty() ? "" : a->inputs[0], ax))
    return "décalage : deux Slice de la même donnée sur l'axe du Concat";
  const std::string& shift = s.shift[static_cast<std::size_t>(ax - 1)];
  const std::string_view start = g.base(a->inputs[1]);
  const Node* sub = g.made_by(a->inputs[1], "Sub");
  const bool start_ok =
      right ? sub != nullptr && sub->inputs.size() == 2 && g.base(sub->inputs[1]) == shift : start == shift;
  if (!start_ok || !sum_of(g, a->inputs[2], start, false) || !sum_of(g, b->inputs[2], start, true))
    return right ? "décalage de retour (taille moins le décalage de l'échelle)"
                 : "décalage (celui de l'échelle)";
  data = a->inputs[0];
  axis = static_cast<int>(ax);
  return nullptr;
}

// Le Reshape qui découpe en fenêtres (split_feature : [b, s, h/s, s, w/s, c], splits en 1 et 3) ou qui les
// rassemble (merge_splits : [b, s, s, h/s, w/s, c], splits en 1 et 2), juste avant le Transpose (0, 1, 3, 2,
// 4, 5).
bool windows_are(const OnnxView& g, const Scale& s, const Node& transpose, bool merge) {
  const Node* reshape = g.producer(transpose.inputs[0]);
  if (reshape == nullptr || reshape->op_type != "Reshape" || reshape->inputs.size() != 2) return false;
  const Node* shape = g.producer(reshape->inputs[1]);
  if (shape == nullptr || shape->op_type != "Concat" || shape->inputs.size() != 6) return false;
  const std::size_t second = merge ? 2 : 3;
  return g.integer(shape->inputs[1]) == s.splits && g.integer(shape->inputs[second]) == s.splits;
}

// Le chemin des données de tensor jusqu'à from, en remontant les Reshape, les Transpose des fenêtres (et
// celui de kᵀ) et les décalages. Rien d'autre : ni biais, ni autre calcul. windows : les découpes en fenêtres
// croisées (merge : ce sont des rassemblements) ; rolls : les axes décalés (bit 1, bit 2) ; transposed : kᵀ
// croisé.
struct Path {
  int windows = 0;
  int rolls = 0;
  int transposed = 0;
};

const char* transpose_step(const OnnxView& g, const Scale& s, const Node& n, bool merge, Path& path) {
  const onnx::Attribute* perm = onnx_attribute(n, "perm");
  if (perm == nullptr) return "Transpose sans perm";
  if (perm->ints == std::vector<std::int64_t>{0, 2, 1}) {
    ++path.transposed;
    return nullptr;
  }
  if (perm->ints != std::vector<std::int64_t>{0, 1, 3, 2, 4, 5} || !windows_are(g, s, n, merge))
    return "fenêtres (Transpose et Reshape de split_feature ou merge_splits, fenêtres de l'échelle)";
  ++path.windows;
  return nullptr;
}

const char* trace(const OnnxView& g, const Scale& s, std::string tensor, std::string_view from, bool merge,
                  Path& path) {
  for (int hops = 0; hops < 32; ++hops) {
    if (tensor == from) return nullptr;
    const Node* n = g.producer(tensor);
    if (n == nullptr || n->inputs.empty()) return "chemin interrompu";
    if (n->op_type == "Concat") {
      int axis = 0;
      if (const char* why = roll_problem(g, s, *n, merge, tensor, axis); why != nullptr) return why;
      if ((path.rolls & (1 << axis)) != 0) return "décalé deux fois sur le même axe";
      path.rolls |= 1 << axis;
      continue;
    }
    if (n->op_type == "Transpose") {
      if (const char* why = transpose_step(g, s, *n, merge, path); why != nullptr) return why;
    } else if (n->op_type != "Reshape") {
      return "opération inattendue entre la projection et l'attention";
    }
    tensor = n->inputs[0];
  }
  return "chemin trop long";
}

// Une projection 128 → 128 sans biais : le MatMul name qui lit input (vide : n'importe quoi, rendu dans
// input), poids [128][128] [entrée][sortie] ajoutés à rows en [sortie][entrée].
const char* projection(const OnnxView& g, const std::string& name, std::string& input, std::string& output,
                       std::vector<float>& rows) {
  const Node* n = g.node(name, "MatMul");
  if (n == nullptr || n->inputs.size() != 2 || n->outputs.size() != 1) return "MatMul absent";
  if (!input.empty() && n->inputs[0] != input) return "ne lit pas le tenseur attendu";
  const onnx::Tensor* w = g.constant(n->inputs[1]);
  if (w == nullptr || w->dims != std::vector<std::int64_t>{kD, kD} ||
      w->floats.size() != static_cast<std::size_t>(kD) * kD)
    return "poids 128 × 128 constants attendus";
  for (std::size_t o = 0; o < kD; ++o)
    for (std::size_t i = 0; i < kD; ++i) rows.push_back(w->floats[i * kD + o]);
  input = n->inputs[0];
  output = n->outputs[0];
  return nullptr;
}

// Le facteur des scores : Div(q kᵀ, Pow(d, 0.5)), d lu dans la forme de q (axe 2). Puis l'éventuel masque :
// Add(scores, masque de l'échelle répété par Expand et Tile). Rend shifted.
const char* scores_problem(const OnnxView& g, const Scale& s, const Node& softmax, const Node& scores,
                           const std::string& q, bool& shifted) {
  const onnx::Attribute* axis = onnx_attribute(softmax, "axis");
  if (softmax.inputs.size() != 1 || (axis != nullptr && axis->i != -1 && axis->i != 2))
    return "Softmax hors du dernier axe";
  const Node* div = g.producer(softmax.inputs[0]);
  shifted = div != nullptr && div->op_type == "Add";
  if (shifted) {
    const Node* tile = div->inputs.size() == 2 ? g.made_by(div->inputs[1], "Tile") : nullptr;
    const Node* expand = tile != nullptr ? g.made_by(tile->inputs[0], "Expand") : nullptr;
    if (expand == nullptr || expand->inputs[0] != s.mask) return "masque des fenêtres décalées";
    div = g.producer(div->inputs[0]);
  }
  if (div == nullptr || div->op_type != "Div" || div->inputs.size() != 2 ||
      div->inputs[0] != scores.outputs[0])
    return "scores divisés par √d";
  const Node* pow = g.made_by(div->inputs[1], "Pow");
  const Node* gather =
      pow != nullptr && pow->inputs.size() == 2 ? g.made_by(pow->inputs[0], "Gather") : nullptr;
  const Node* shape =
      gather != nullptr && gather->inputs.size() == 2 ? g.made_by(gather->inputs[0], "Shape") : nullptr;
  if (shape == nullptr || g.scalar(pow->inputs[1]) != 0.5F || g.integer(gather->inputs[1]) != 2 ||
      shape->inputs[0] != q)
    return "facteur √d (Pow 0.5 des canaux de q)";
  return nullptr;
}

// La LayerNorm norm1 qui lit la fusion : axe -1, ε, échelle et décalage de 128 par leurs noms.
const char* norm_problem(const OnnxView& g, const Node* ln, const std::string& merged,
                         const std::string& weights, GmflowAttentionLayer& layer, float& epsilon) {
  if (ln == nullptr || ln->inputs.size() != 3 || ln->outputs.empty() || ln->inputs[0] != merged ||
      ln->inputs[1] != weights + "norm1.weight" || ln->inputs[2] != weights + "norm1.bias")
    return "norm1 : LayerNormalization de la fusion, poids par leurs noms";
  const onnx::Attribute* axis = onnx_attribute(*ln, "axis");
  const onnx::Attribute* eps = onnx_attribute(*ln, "epsilon");
  const onnx::Tensor* gamma = g.constant(ln->inputs[1]);
  const onnx::Tensor* beta = g.constant(ln->inputs[2]);
  if ((axis != nullptr && axis->i != -1) || gamma == nullptr || beta == nullptr ||
      gamma->floats.size() != kD || beta->floats.size() != kD)
    return "norm1 : axe -1, 128 canaux";
  const float e = eps != nullptr ? eps->f : 1e-5F;
  if (!(e > 0.F) || (epsilon != 0.F && e != epsilon)) return "norm1 : ε différent des autres couches";
  epsilon = e;
  layer.gamma = gamma->floats;
  layer.beta = beta->floats;
  return nullptr;
}

struct Reading {
  const OnnxView& g;
  float scale = 0.F;
  float epsilon = 0.F;
};

// Les noms d'une couche : module (les Linear et LayerNorm, suffixés à l'échelle), cœur (l'attention), poids.
struct Names {
  std::string module;
  std::string core;
  std::string weights;
  std::string suffix;
};

Names names_of(const Scale& s, int block, bool cross) {
  const std::string_view kind = cross ? "cross_attn_ffn" : "self_attn";
  return {.module = std::format("{}{}/{}/", kLayers, block, kind),
          .core = std::format("{}{}/{}{}/", kLayers, block, kind, s.suffix),
          .weights = std::format("{}{}.{}.", kWeights, block, kind),
          .suffix = s.suffix};
}

// Le cœur d'une couche : scores, Softmax, mélange, chemins depuis q, k, v et vers la fusion.
const char* core_problem(Reading& r, const Scale& s, const Names& n, const std::array<std::string, 3>& qkv,
                         std::string& merged_input, GmflowAttentionLayer& layer) {
  const Node* scores = r.g.node(n.core + "MatMul", "MatMul");
  const Node* softmax = r.g.node(n.core + "Softmax", "Softmax");
  const Node* mix = r.g.node(n.core + "MatMul_1", "MatMul");
  if (scores == nullptr || softmax == nullptr || mix == nullptr || scores->inputs.size() != 2 ||
      mix->inputs.size() != 2 || mix->inputs[0] != softmax->outputs[0])
    return "MatMul, Softmax, MatMul_1 de l'attention";
  if (const char* why = scores_problem(r.g, s, *softmax, *scores, qkv[0], layer.shifted); why != nullptr)
    return why;
  const int rolls = layer.shifted ? 0b110 : 0;
  const std::array<std::pair<std::string, std::string>, 3> paths{
      {{scores->inputs[0], qkv[0]}, {scores->inputs[1], qkv[1]}, {mix->inputs[1], qkv[2]}}};
  for (std::size_t i = 0; i < paths.size(); ++i) {
    Path path;
    if (const char* why = trace(r.g, s, paths[i].first, paths[i].second, false, path); why != nullptr)
      return why;
    if (path.windows != 1 || path.rolls != rolls || path.transposed != (i == 1 ? 1 : 0))
      return "q, k, v : une découpe en fenêtres, kᵀ, décalés si la couche l'est";
  }
  Path back;
  if (const char* why = trace(r.g, s, merged_input, mix->outputs[0], true, back); why != nullptr) return why;
  if (back.windows != 1 || back.rolls != rolls || back.transposed != 0)
    return "sortie de l'attention : fenêtres rassemblées, décalage rendu";
  return nullptr;
}

// Une couche : q lit src, k et v lisent target. Rend la sortie (self : src + LN ; cross : LN, que le MLP
// lit).
std::expected<std::string, Refusal> read_layer(Reading& r, const Scale& s, int block, bool cross,
                                               const std::string& src, const std::string& target,
                                               GmflowAttentionLayer& layer) {
  const Names n = names_of(s, block, cross);
  layer.cross = cross;
  std::array<std::string, 3> out;
  constexpr std::array<std::string_view, 3> kProj{"q_proj", "k_proj", "v_proj"};
  for (std::size_t i = 0; i < 3; ++i) {
    const std::string name = std::format("{}{}{}/MatMul", n.module, kProj[i], n.suffix);
    std::string input = i == 0 ? src : target;
    if (const char* why = projection(r.g, name, input, out[i], layer.qkv); why != nullptr)
      return missing(name + " : " + why);
  }
  const std::string merge = n.module + "merge" + n.suffix + "/MatMul";
  std::string merge_in, merged;
  if (const char* why = projection(r.g, merge, merge_in, merged, layer.merge); why != nullptr)
    return missing(merge + " : " + why);
  if (const char* why = core_problem(r, s, n, out, merge_in, layer); why != nullptr)
    return missing(n.core + " : " + why);
  const Node* ln = r.g.node(n.module + "norm1" + n.suffix + "/LayerNormalization", "LayerNormalization");
  if (const char* why = norm_problem(r.g, ln, merged, n.weights, layer, r.epsilon); why != nullptr)
    return missing(n.module + why);
  std::vector<const Node*> readers;
  for (const Node* reader : r.g.readers(ln->outputs[0])) {
    if (reader->op_type != "Shape") readers.push_back(reader);
  }
  const Node* next = readers.size() == 1 ? readers[0] : nullptr;
  const onnx::Attribute* axis = next != nullptr ? onnx_attribute(*next, "axis") : nullptr;
  const std::vector<std::string> pair{src, ln->outputs[0]};
  if (!cross && (next == nullptr || next->op_type != "Add" || next->inputs != pair))
    return missing(n.module + " : résiduel Add(source, norm1) attendu");
  if (cross && (next == nullptr || next->op_type != "Concat" || next->inputs != pair || axis == nullptr ||
                (axis->i != -1 && axis->i != 2)))
    return missing(n.module + " : Concat(source, message) du MLP attendu");
  return cross ? ln->outputs[0] : next->outputs[0];
}

// La sortie du bloc : Add(source, norm2) du MLP de la cross (norm2 lit le MLP, que la PR 3 lit).
std::string block_output(const OnnxView& g, const Scale& s, int block, const std::string& source) {
  const Names n = names_of(s, block, true);
  const Node* ln = g.node(n.module + "norm2" + n.suffix + "/LayerNormalization", "LayerNormalization");
  if (ln == nullptr || ln->outputs.empty()) return {};
  for (const Node* reader : g.readers(ln->outputs[0])) {
    if (reader->op_type == "Add" && reader->inputs == std::vector<std::string>{source, ln->outputs[0]})
      return reader->outputs[0];
  }
  return {};
}

// La moitié du lot de chunk(2) : (lot + 1) / 2, le lot lu dans la forme de x (axe 0) ; times : 1 ou 2.
bool half_batch(const OnnxView& g, std::string_view tensor, const std::string& x, std::int64_t times) {
  const Node* mul = g.made_by(tensor, "Mul");
  const Node* div = mul != nullptr && mul->inputs.size() == 2 ? g.made_by(mul->inputs[0], "Div") : nullptr;
  const Node* add = div != nullptr && div->inputs.size() == 2 ? g.made_by(div->inputs[0], "Add") : nullptr;
  const Node* gather =
      add != nullptr && add->inputs.size() == 2 ? g.made_by(add->inputs[0], "Gather") : nullptr;
  const Node* shape =
      gather != nullptr && gather->inputs.size() == 2 ? g.made_by(gather->inputs[0], "Shape") : nullptr;
  return shape != nullptr && shape->inputs[0] == x && ints_are(g, gather->inputs[1], {0}) &&
         ints_are(g, add->inputs[1], {1}) && g.integer(div->inputs[1]) == 2 &&
         g.integer(mul->inputs[1]) == times;
}

// La cible de la cross-attention du bloc : les deux moitiés du lot de x échangées. Au premier bloc,
// Concat(f1, f0) quand x = Concat(f0, f1) ; ensuite Concat(Slice(x, m, 2m), Slice(x, 0, m)), m = moitié du
// lot.
bool swapped(const OnnxView& g, const std::string& target, const std::string& x, int block) {
  const Node* concat = g.producer(target);
  std::int64_t axis = 0;
  if (!concat_on(concat, axis)) return false;
  if (block == 0) {
    const Node* first = g.producer(x);
    return concat_on(first, axis) && first->inputs[0] == concat->inputs[1] &&
           first->inputs[1] == concat->inputs[0];
  }
  const Node* upper = g.producer(concat->inputs[0]);
  const Node* lower = g.producer(concat->inputs[1]);
  return two_slices(g, upper, lower, x, 0) && upper->inputs[1] == lower->inputs[2] &&
         half_batch(g, upper->inputs[1], x, 1) && half_batch(g, upper->inputs[2], x, 2);
}

bool same_layer(const GmflowAttentionLayer& a, const GmflowAttentionLayer& b) {
  return a.cross == b.cross && a.shifted == b.shifted && a.qkv == b.qkv && a.merge == b.merge &&
         a.gamma == b.gamma && a.beta == b.beta;
}

// Les six blocs d'une échelle ; au quart, mêmes couches qu'au huitième.
std::expected<void, Refusal> read_blocks(Reading& r, const Scale& s,
                                         std::vector<GmflowAttentionLayer>& layers) {
  const Node* input = r.g.node(s.top + "Concat_8", "Concat");
  if (input == nullptr) return missing(s.top + "Concat_8 : l'entrée du transformeur");
  std::string x = input->outputs[0];
  for (int b = 0; b < kGmflowBlocks; ++b) {
    std::array<GmflowAttentionLayer, 2> read;
    auto self = read_layer(r, s, b, false, x, x, read[0]);
    if (!self) return std::unexpected(std::move(self.error()));
    const Node* k =
        r.g.node(std::format("{}{}/cross_attn_ffn/k_proj{}/MatMul", kLayers, b, s.suffix), "MatMul");
    if (k == nullptr || k->inputs.empty() || !swapped(r.g, k->inputs[0], x, b))
      return missing(
          std::format("bloc {} : la cross ne lit pas l'autre moitié du lot à l'entrée du bloc", b));
    auto cross = read_layer(r, s, b, true, *self, k->inputs[0], read[1]);
    if (!cross) return std::unexpected(std::move(cross.error()));
    if (read[0].shifted != read[1].shifted) return missing(std::format("bloc {} : décalages différents", b));
    for (std::size_t i = 0; i < 2; ++i) {
      const std::size_t at = 2 * static_cast<std::size_t>(b) + i;
      if (s.suffix.empty()) {
        layers.push_back(std::move(read[i]));
      } else if (!same_layer(layers[at], read[i])) {
        return missing(std::format("couche {} : poids ou décalage différents d'une échelle à l'autre", at));
      }
    }
    x = block_output(r.g, s, b, *self);
    if (x.empty()) return missing(std::format("bloc {} : sortie Add(source, norm2) attendue", b));
  }
  return {};
}

// Le plan : l'entrée 0 ; puis par bloc la sortie de la self, le message de la cross, la sortie du bloc.
void build_plan(GmflowAttention& a) {
  int x = a.input;
  a.tensors = 1;
  for (int b = 0; b < kGmflowBlocks; ++b) {
    const int s = a.tensors++, m = a.tensors++, next = a.tensors++;
    a.ops.push_back({.layer = 2 * b, .query = x, .keys = x, .residual = true, .out = s});
    a.ops.push_back({.layer = 2 * b + 1, .query = s, .keys = x, .other_half = true, .out = m});
    a.ops.push_back({.kind = GmflowAttnOpKind::kMlp, .query = s, .message = m, .out = next});
    x = next;
  }
  a.output = x;
}

}  // namespace

std::expected<GmflowAttention, Refusal> gmflow_attention(std::span<const std::byte> pair) {
  auto parsed = onnx::read_onnx(pair);
  if (!parsed) return std::unexpected(Refusal{RefusalCode::ModelUnreadable, parsed.error().detail});
  const OnnxView view(*parsed);
  Reading reading{.g = view};
  GmflowAttention a;
  for (int index = 0; index < 2; ++index) {
    auto scale = read_scale(view, index, a.mask);
    if (!scale) return std::unexpected(std::move(scale.error()));
    a.splits[static_cast<std::size_t>(index)] = scale->splits;
    if (auto ok = read_blocks(reading, *scale, a.layers); !ok) return std::unexpected(std::move(ok.error()));
  }
  a.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kD)));
  a.epsilon = reading.epsilon;
  build_plan(a);
  if (auto ok = gmflow_attention_check(a); !ok) return std::unexpected(std::move(ok.error()));
  return a;
}

}  // namespace kirari::prepare
