// L'attention du transformeur de GMFlow lue pour les noyaux natifs (prepare/gmflow_attention_native.h) : un
// faux transformeur aux noms, opérateurs, attributs et liens de la paire exportée (fenêtres, décalages de
// Swin et leur masque, facteur √d, cible de la cross-attention, résiduel), lu et rangé, ses fautes refusées,
// puis le plan et ses fautes. La vraie paire se lit dans gmflow_native_test.cpp, avec le squelette.
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/gmflow_attention_native.h"
#include "rife_fake.h"

namespace kirari::prepare {
namespace {

constexpr int kD = kGmflowModel;

// Le faux transformeur ; chaque faute casse un maillon que la lecture doit voir.
struct FakeTransformer {
  std::int64_t splits_w = 2;  // fenêtres par côté en largeur au huitième (2 en hauteur)
  std::int64_t half = 2;      // diviseur du décalage en hauteur
  float fill = -100.F;
  bool edge_swapped = false;      // une région comptée au bord moins une fenêtre au lieu du décalage
  bool roll_by_window = false;    // décalé d'une fenêtre au lieu d'une demi-fenêtre
  bool residual_swapped = false;  // Add(norm1, source)
  bool cross_same_half = false;   // la cross lit k et v dans sa propre moitié du lot
  bool q_bias = false;            // un biais après q_proj
  float root = 0.5F;              // l'exposant de √d
  bool quarter_differs = false;   // une couche aux poids différents au quart
  float last_epsilon = 1e-5F;
  std::int64_t softmax_axis = -1;
  bool unmasked = false;            // fenêtres décalées sans masque
  std::int64_t reshape_splits = 0;  // fenêtres du Reshape de split_feature (0 : celles de l'échelle)
  bool mask_not_negated = false;    // -100 là où les régions sont égales (Where sans Not)
  bool regions_swapped = false;     // la ligne de la région comptée par les comparaisons des colonnes
  bool mask_unwindowed = false;     // la carte des régions comparée sans être découpée en fenêtres
  std::int64_t norm1_stash = 1;     // statistiques de norm1 (1 : f32)
};

// Le poids ONNX [entrée][sortie] de la partie part (q, k, v, fusion) de la couche layer : exact en FP32.
float weight(int layer, int part, int in, int out) {
  return static_cast<float>((in * 5 + out * 3 + layer * 7 + part * 11) % 23 - 11) * 0.0625F;
}
float gamma(int c) {
  return 1.0F + static_cast<float>(c) * 0.0078125F;
}

Bytes ints_attribute(std::string_view name, std::initializer_list<std::int64_t> values) {
  Bytes out;
  delimited(out, 1, text(name));
  for (const std::int64_t v : values) integer(out, 8, static_cast<std::uint64_t>(v));
  return out;
}
Bytes int_attribute(std::string_view name, std::int64_t value) {
  return attribute({.name = std::string(name), .num = value});
}
Bytes float_attribute(std::string_view name, float value) {
  Bytes out;
  delimited(out, 1, text(name));
  varint(out, (2U << 3U) | 5U);
  const auto bits = std::bit_cast<std::uint32_t>(value);
  for (unsigned i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>((bits >> (8U * i)) & 0xffU));
  return out;
}
Bytes raw(const std::vector<float>& values) {
  Bytes out(values.size() * 4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    std::memcpy(out.data() + 4 * i, &bits, 4);
  }
  return out;
}

class Graph {
 public:
  explicit Graph(const FakeTransformer& f) : f_(f) {}

  const FakeTransformer& fake() const { return f_; }

  // Un nœud à une sortie ; sans nom, un nom neuf.
  std::string op(std::string name, std::string_view type, std::vector<std::string> in,
                 const std::vector<Bytes>& attrs = {}) {
    if (name.empty()) name = std::format("/flownet/n{}", count_++);
    Bytes n = node(name, type, std::move(in), {name + "_output_0"});
    for (const Bytes& a : attrs) delimited(n, 5, a);
    delimited(graph_, 1, n);
    return name + "_output_0";
  }
  std::string ints(std::int64_t value) {
    const std::string name = std::format("c{}", count_++);
    delimited(graph_, 5, tensor(name, 7, 1, int64s({value})));
    return name;
  }
  std::string scalar(float value) {
    const std::string name = std::format("c{}", count_++);
    delimited(graph_, 5, tensor(name, 1, 1, floats({value})));
    return name;
  }
  // Les poids de la partie part de la couche layer, une fois pour les deux échelles (sauf la faute).
  std::string matrix(int layer, int part, int scale) {
    const bool differs = f_.quarter_differs && scale == 1 && layer == 4 && part == 1;
    const std::string name = std::format("w{}_{}{}", layer, part, differs ? "_quart" : "");
    if (!names_.insert(name).second) return name;
    std::vector<float> values;
    for (int i = 0; i < kD; ++i)
      for (int o = 0; o < kD; ++o) values.push_back(weight(layer, part, i, o) + (differs ? 0.5F : 0.F));
    delimited(graph_, 5, tensor_dims(name, {kD, kD}, raw(values)));
    return name;
  }
  // norm1 de la couche : weights + « norm1.weight » et « norm1.bias », une fois.
  void norm(const std::string& weights) {
    if (!names_.insert(weights).second) return;
    std::vector<float> g, b;
    for (int c = 0; c < kD; ++c) {
      g.push_back(gamma(c));
      b.push_back(static_cast<float>(c % 7 - 3) * 0.125F);
    }
    delimited(graph_, 5, tensor_dims(weights + "norm1.weight", {kD}, raw(g)));
    delimited(graph_, 5, tensor_dims(weights + "norm1.bias", {kD}, raw(b)));
  }
  Bytes model() const {
    Bytes out;
    integer(out, 1, 8);
    delimited(out, 7, graph_);
    return out;
  }

 private:
  const FakeTransformer& f_;
  Bytes graph_;
  std::set<std::string> names_;
  int count_ = 0;
};

struct Scale {
  int index = 0;
  std::string top, sfx;
  std::int64_t splits = 2;
  std::string wh, ww, sh, sw, mask;
};

// Fenêtres, décalages d'une demi-fenêtre et masque, comme l'export : régions de chaque ligne et de chaque
// colonne comptées au bord moins une fenêtre et au bord moins un décalage, carte ligne · 3 + colonne découpée
// en fenêtres, différence de deux régions, -100 où elle n'est pas nulle, 0 où elle l'est.
void geometry(Graph& g, Scale& s) {
  const FakeTransformer& f = g.fake();
  const std::string h = s.top + "height", w = s.top + "width";
  s.wh = g.op(s.top + "Div", "Div", {h, g.ints(s.splits)});
  s.ww = g.op(s.top + "Div_1", "Div", {w, g.ints(s.index == 0 ? f.splits_w : s.splits)});
  s.sh = g.op(s.top + "Div_2", "Div", {s.wh, g.ints(f.half)});
  s.sw = g.op(s.top + "Div_3", "Div", {s.ww, g.ints(2)});
  const std::array<std::string, 4> edges{s.wh, f.edge_swapped ? s.wh : s.sh, s.ww, s.sw};
  std::array<std::string, 4> ge;
  for (std::size_t i = 0; i < 4; ++i) {
    const std::string sub = g.op("", "Sub", {i < 2 ? h : w, edges[i]});
    ge[i] = g.op(s.top + (i == 0 ? "GreaterOrEqual" : std::format("GreaterOrEqual_{}", i)), "GreaterOrEqual",
                 {i < 2 ? s.top + "rows" : s.top + "cols", sub});
  }
  const auto count = [&](std::size_t a) {
    const std::string sum = g.op("", "Add", {g.op("", "Cast", {ge[a]}), g.op("", "Cast", {ge[a + 1]})});
    return g.op("", "Reshape", {sum, "shape"});
  };
  const std::string rows = g.op(s.top + "Mul", "Mul", {count(f.regions_swapped ? 2 : 0), g.ints(3)});
  std::string map = g.op("", "Cast", {g.op("", "Add", {rows, count(f.regions_swapped ? 0 : 2)})});
  if (!f.mask_unwindowed) {
    map = g.op("", "Transpose", {g.op("", "Reshape", {map, "split"})},
               {ints_attribute("perm", {0, 1, 3, 2, 4, 5})});
  }
  map = g.op("", "Reshape", {map, "flat"});
  const std::string diff =
      g.op("", "Sub", {g.op("", "Unsqueeze", {map, g.ints(1)}), g.op("", "Unsqueeze", {map, g.ints(2)})});
  const std::string equal = g.op(s.top + "Equal", "Equal", {diff, g.scalar(0.F)});
  const std::string equal1 = g.op(s.top + "Equal_1", "Equal", {diff, g.scalar(0.F)});
  const std::string differs = f.mask_not_negated ? equal : g.op("", "Not", {equal});
  const std::string where =
      g.op(s.top + "Where", "Where", {g.op("", "Cast", {differs}), g.scalar(f.fill), diff});
  s.mask = g.op(s.top + "Where_1", "Where", {g.op("", "Cast", {equal1}), g.scalar(0.F), where});
}

// torch.roll sur l'axe 1 ou 2 : Concat(Slice(x, a, a + reste), Slice(x, 0, 0 + a)), a = le décalage (aller)
// ou la taille moins le décalage (retour).
std::string roll(Graph& g, const Scale& s, const std::string& x, int axis, bool back) {
  const std::string& shift = axis == 1 ? s.sh : s.sw;
  const std::string& window = axis == 1 ? s.wh : s.ww;
  const std::string start =
      back ? g.op("", "Sub", {"size", shift}) : (g.fake().roll_by_window ? window : shift);
  const std::string axes = g.ints(axis);
  const std::string a = g.op("", "Slice", {x, start, g.op("", "Add", {start, "rest"}), axes});
  const std::string b = g.op("", "Slice", {x, g.ints(0), g.op("", "Add", {g.ints(0), start}), axes});
  return g.op("", "Concat", {a, b}, {int_attribute("axis", axis)});
}

// split_feature (merge : merge_splits) : Reshape à six axes, Transpose (0, 1, 3, 2, 4, 5), Reshape.
std::string windows(Graph& g, const Scale& s, std::string x, bool shifted, bool merge) {
  if (shifted && !merge) x = roll(g, s, roll(g, s, x, 1, false), 2, false);
  const std::int64_t wrong = g.fake().reshape_splits;
  const std::string n = g.ints(s.splits), m = g.ints(!merge && wrong > 0 ? wrong : s.splits);
  const std::vector<std::string> dims =
      merge ? std::vector<std::string>{g.ints(2), n, m, "rows", "cols", g.ints(kD)}
            : std::vector<std::string>{g.ints(2), n, "rows", m, "cols", g.ints(kD)};
  const std::string shape = g.op("", "Concat", dims, {int_attribute("axis", 0)});
  const std::string r = g.op("", "Reshape", {x, shape});
  const std::string t = g.op("", "Transpose", {r}, {ints_attribute("perm", {0, 1, 3, 2, 4, 5})});
  x = g.op("", "Reshape", {t, "flat"});
  if (shifted && merge) x = roll(g, s, roll(g, s, x, 1, true), 2, true);
  return x;
}

// Une couche : q lit src, k et v lisent target. Rend la sortie de la self (src + LN) ou l'entrée du MLP.
std::string layer(Graph& g, const Scale& s, int block, bool cross, const std::string& src,
                  const std::string& target) {
  const FakeTransformer& f = g.fake();
  const std::string_view kind = cross ? "cross_attn_ffn" : "self_attn";
  const std::string module = std::format("/flownet/transformer/layers.{}/{}/", block, kind);
  const std::string core = std::format("/flownet/transformer/layers.{}/{}{}/", block, kind, s.sfx);
  const std::string weights = std::format("gmfss.flownet.transformer.layers.{}.{}.", block, kind);
  const int index = 2 * block + (cross ? 1 : 0);
  const bool shifted = block % 2 == 1;
  constexpr std::array<std::string_view, 3> kProj{"q_proj", "k_proj", "v_proj"};
  std::array<std::string, 3> qkv;
  for (int p = 0; p < 3; ++p) {
    qkv[static_cast<std::size_t>(p)] =
        g.op(std::format("{}{}{}/MatMul", module, kProj[static_cast<std::size_t>(p)], s.sfx), "MatMul",
             {p == 0 ? src : target, g.matrix(index, p, s.index)});
  }
  const std::string q = f.q_bias ? g.op("", "Add", {qkv[0], "bias"}) : qkv[0];
  const std::string k =
      g.op("", "Transpose", {windows(g, s, qkv[1], shifted, false)}, {ints_attribute("perm", {0, 2, 1})});
  const std::string scores = g.op(core + "MatMul", "MatMul", {windows(g, s, q, shifted, false), k});
  const std::string channels = g.op("", "Gather", {g.op("", "Shape", {qkv[0]}), g.ints(2)});
  std::string x = g.op("", "Div", {scores, g.op("", "Pow", {channels, g.scalar(f.root)})});
  if (shifted && !f.unmasked)
    x = g.op("", "Add", {x, g.op("", "Tile", {g.op("", "Expand", {s.mask, "shape"}), "repeats"})});
  const std::string p = g.op(core + "Softmax", "Softmax", {x}, {int_attribute("axis", f.softmax_axis)});
  const std::string mix = g.op(core + "MatMul_1", "MatMul", {p, windows(g, s, qkv[2], shifted, false)});
  const std::string merged = g.op(module + "merge" + s.sfx + "/MatMul", "MatMul",
                                  {windows(g, s, mix, shifted, true), g.matrix(index, 3, s.index)});
  g.norm(weights);
  const float epsilon = index == kGmflowLayers - 1 ? f.last_epsilon : 1e-5F;
  const std::string ln = g.op(module + "norm1" + s.sfx + "/LayerNormalization", "LayerNormalization",
                              {merged, weights + "norm1.weight", weights + "norm1.bias"},
                              {int_attribute("axis", -1), int_attribute("stash_type", f.norm1_stash),
                               float_attribute("epsilon", epsilon)});
  if (cross) return g.op("", "Concat", {src, ln}, {int_attribute("axis", -1)});
  return g.op("", "Add", f.residual_swapped ? std::vector{ln, src} : std::vector{src, ln});
}

// La cible de la cross : les moitiés du lot à l'entrée du bloc échangées (Slice par la moitié du lot lue dans
// la forme de x, ou Concat(f1, f0) au premier bloc).
std::string swapped(Graph& g, const Scale& s, int block, const std::string& x) {
  if (g.fake().cross_same_half) return x;
  if (block == 0) return g.op("", "Concat", {s.top + "f1", s.top + "f0"}, {int_attribute("axis", 0)});
  const std::string batch = g.op("", "Gather", {g.op("", "Shape", {x}), g.ints(0)});
  const std::string half = g.op("", "Div", {g.op("", "Add", {batch, g.ints(1)}), g.ints(2)});
  const std::string m1 = g.op("", "Mul", {half, g.ints(1)});
  const std::string m2 = g.op("", "Mul", {half, g.ints(2)});
  const std::string axes = g.ints(0);
  const std::string upper = g.op("", "Slice", {x, m1, m2, axes});
  const std::string lower = g.op("", "Slice", {x, g.ints(0), m1, axes});
  return g.op("", "Concat", {upper, lower}, {int_attribute("axis", 0)});
}

Bytes fake_transformer(const FakeTransformer& f) {
  Graph g(f);
  for (int index = 0; index < 2; ++index) {
    Scale s{.index = index,
            .top = index == 0 ? "/flownet/transformer/" : "/flownet/transformer_1/",
            .sfx = index == 0 ? "" : "_1",
            .splits = index == 0 ? 2 : 8};
    geometry(g, s);
    std::string x =
        g.op(s.top + "Concat_8", "Concat", {s.top + "f0", s.top + "f1"}, {int_attribute("axis", 0)});
    for (int b = 0; b < kGmflowBlocks; ++b) {
      const std::string target = swapped(g, s, b, x);
      const std::string self = layer(g, s, b, false, x, x);
      const std::string mlp = layer(g, s, b, true, self, target);
      const std::string ln2 = g.op(
          std::format("/flownet/transformer/layers.{}/cross_attn_ffn/norm2{}/LayerNormalization", b, s.sfx),
          "LayerNormalization", {mlp, "norm2.weight", "norm2.bias"});
      x = g.op("", "Add", {self, ln2});
    }
  }
  return g.model();
}

}  // namespace

TEST_CASE("GMFlow natif : le faux transformeur se lit, rangé pour les noyaux", "[prepare][gmflow]") {
  const auto read = gmflow_attention(fake_transformer(FakeTransformer{}));
  INFO((read ? std::string() : read.error().detail));
  REQUIRE(read);
  const GmflowAttention& a = *read;
  CHECK(a.splits == std::array<int, 2>{2, 8});
  CHECK(a.mask == -100.F);
  CHECK(a.epsilon == 1e-5F);
  CHECK(std::abs(a.scale - 1.F / std::sqrt(128.F)) < 1e-7F);
  REQUIRE(a.layers.size() == static_cast<std::size_t>(kGmflowLayers));
  for (std::size_t i = 0; i < a.layers.size(); ++i) {
    CHECK(a.layers[i].cross == (i % 2 == 1));
    CHECK(a.layers[i].shifted == (i / 2 % 2 == 1));  // blocs impairs
  }
  // [sortie][entrée] depuis les MatMul [entrée][sortie] : k de la couche 5 (lignes 128 à 255), sortie 7,
  // entrée 3 ; la fusion de la couche 2, sortie 9, entrée 4 ; v de la couche 0, sortie 127, entrée 126.
  CHECK(a.layers[5].qkv[(kD + 7) * kD + 3] == weight(5, 1, 3, 7));
  CHECK(a.layers[2].merge[9 * kD + 4] == weight(2, 3, 4, 9));
  CHECK(a.layers[0].qkv[(2 * kD + 127) * kD + 126] == weight(0, 2, 126, 127));
  CHECK(a.layers[7].gamma[100] == gamma(100));
  // Le plan : self, cross, MLP par bloc ; la cross lit k et v à l'entrée du bloc, dans l'autre moitié du lot.
  REQUIRE(a.ops.size() == 18);
  CHECK(a.tensors == 19);
  CHECK(a.output == 18);
  const GmflowAttnOp& self = a.ops[3];
  const GmflowAttnOp& cross = a.ops[4];
  CHECK((self.residual && !self.other_half && self.query == self.keys));
  CHECK((!cross.residual && cross.other_half && cross.keys == self.query && cross.query == self.out));
  CHECK((a.ops[5].kind == GmflowAttnOpKind::kMlp && a.ops[5].message == cross.out));
  CHECK(gmflow_attention_check(a));
}

TEST_CASE("GMFlow natif : les fautes du faux transformeur sont refusées", "[prepare][gmflow]") {
  const auto refused = [](const FakeTransformer& f) {
    const auto read = gmflow_attention(fake_transformer(f));
    return !read && read.error().code == RefusalCode::FieldMissing;
  };
  CHECK(refused({.splits_w = 3}));  // fenêtres différentes en hauteur et en largeur
  CHECK(refused({.half = 3}));      // un décalage d'un tiers de fenêtre
  CHECK(refused({.fill = 100.F}));
  CHECK(refused({.edge_swapped = true}));
  CHECK(refused({.roll_by_window = true}));
  CHECK(refused({.residual_swapped = true}));
  CHECK(refused({.cross_same_half = true}));
  CHECK(refused({.q_bias = true}));
  CHECK(refused({.root = 0.25F}));
  CHECK(refused({.quarter_differs = true}));
  CHECK(refused({.last_epsilon = 1e-3F}));
  CHECK(refused({.softmax_axis = 1}));
  CHECK(refused({.unmasked = true}));
  CHECK(refused({.reshape_splits = 3}));
  // Refusées pour ce qu'elles faussent, pas par un contrôle plus tôt.
  const auto refused_for = [](const FakeTransformer& f, std::string_view why) {
    const auto read = gmflow_attention(fake_transformer(f));
    return !read && read.error().detail.find(why) != std::string::npos;
  };
  CHECK(refused_for({.mask_not_negated = true}, "conditions du masque"));
  CHECK(refused_for({.regions_swapped = true}, "carte des régions"));
  CHECK(refused_for({.mask_unwindowed = true}, "carte des régions"));
  CHECK(refused_for({.norm1_stash = 0}, "norm1"));
  const FakeTransformer none;
  const auto empty = gmflow_attention(Graph(none).model());  // aucun transformeur
  REQUIRE_FALSE(empty);
  CHECK(empty.error().code == RefusalCode::FieldMissing);
  const auto garbage = gmflow_attention(std::vector<std::byte>(16, std::byte{0x7f}));
  REQUIRE_FALSE(garbage);
  CHECK(garbage.error().code == RefusalCode::ModelUnreadable);
}

TEST_CASE("GMFlow natif : le plan de l'attention se tient, ses fautes sont refusées", "[prepare][gmflow]") {
  const auto read = gmflow_attention(fake_transformer(FakeTransformer{}));
  REQUIRE(read);
  const auto broken = [&](auto&& change) {
    GmflowAttention a = *read;
    change(a);
    const auto ok = gmflow_attention_check(a);
    return !ok && ok.error().code == RefusalCode::FieldMissing;
  };
  CHECK(broken([](GmflowAttention& a) { a.ops[1].other_half = false; }));
  CHECK(broken([](GmflowAttention& a) { a.ops[0].residual = false; }));
  CHECK(broken([](GmflowAttention& a) { a.ops[1].keys = a.ops[0].out; }));  // k, v de la cross après la self
  CHECK(broken([](GmflowAttention& a) { a.ops[2].message = a.ops[0].out; }));
  CHECK(broken([](GmflowAttention& a) { a.ops[3].query = a.ops[5].out; }));  // pas encore écrit
  CHECK(broken([](GmflowAttention& a) { a.ops[4].layer = 2; }));             // la cross sur une self
  CHECK(broken([](GmflowAttention& a) { a.ops[5].out = a.ops[2].out; }));    // écrit deux fois
  CHECK(broken([](GmflowAttention& a) { a.ops.pop_back(); }));
  CHECK(broken([](GmflowAttention& a) { a.layers[3].merge.pop_back(); }));
  CHECK(broken([](GmflowAttention& a) { a.output = 5; }));
  CHECK(broken([](GmflowAttention& a) { a.splits[1] = 0; }));
  CHECK(broken([](GmflowAttention& a) { a.mask = 0.F; }));
  CHECK(broken([](GmflowAttention& a) { a.layers[2].qkv[7] = 1e5F; }));  // hors du FP16 des noyaux
  CHECK(broken([](GmflowAttention& a) { a.layers[9].merge[3] = std::numeric_limits<float>::infinity(); }));
  CHECK(broken([](GmflowAttention& a) { a.layers[4].gamma[0] = std::numeric_limits<float>::quiet_NaN(); }));
}

}  // namespace kirari::prepare
