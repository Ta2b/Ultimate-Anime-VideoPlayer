// Le MLP des couches de cross-attention de GMFlow lu pour le noyau natif (prepare/gmflow_mlp.h) : un faux
// transformeur aux noms, opérateurs, constantes et attributs de la paire exportée (couche cachée réduite à
// 64), lu et rangé, ses fautes refusées, le plan et ses fautes, puis la vraie paire quand elle est là
// (KIRARI_RIFE_MODEL_DIR).
#include "prepare/gmflow_mlp.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/model_source.h"
#include "rife_fake.h"

#ifndef KIRARI_RIFE_MODEL_DIR
#define KIRARI_RIFE_MODEL_DIR ""
#endif

namespace kirari::prepare {
namespace {

constexpr int kC = 128;

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

// Un poids ONNX de la couche layer, ligne r, colonne c : des valeurs distinctes, exactes en FP32 et en FP16.
float weight(int layer, int r, int c) {
  return static_cast<float>((layer * 7 + r * 3 + c * 5) % 17 - 8) * 0.0625F;
}

// Le faux transformeur : six couches aux deux échelles. Chaque faute casse un maillon que la lecture doit
// voir.
struct FakeMlp {
  int hidden = 64;
  float sqrt2 = std::sqrt(2.0F);
  float add_one = 1.0F;
  float half = 0.5F;
  std::int64_t concat_axis = -1;
  std::int64_t norm_axis = -1;
  float quarter_epsilon = 1e-5F;
  bool concat_swapped = false;
  bool residual_swapped = false;
  bool hidden_read_elsewhere = false;
  bool quarter_weights_differ = false;
  bool bias_after_mlp = false;
  bool w2_transposed = false;
  bool no_quarter = false;
};

class Builder {
 public:
  void init(const std::string& name, const std::vector<std::int64_t>& dims,
            const std::vector<float>& values) {
    delimited(graph_, 5, tensor_dims(name, dims, raw(values)));
  }
  void add(const Bytes& n) { delimited(graph_, 1, n); }
  std::string simple(const std::string& name, std::string_view op, std::vector<std::string> in,
                     const std::vector<Attr>& attrs = {}) {
    add(node(name, op, std::move(in), {name + "_output_0"}, attrs));
    return name + "_output_0";
  }
  std::string constant(const std::string& name, float value) {
    add(node(name, "Constant", {}, {name + "_output_0"},
             {Attr{.name = "value", .tensor = tensor("", 1, 1, raw({value}))}}));
    return name + "_output_0";
  }
  std::string norm(const std::string& name, const std::string& in, const std::string& module, float epsilon,
                   std::int64_t axis) {
    Bytes n = node(name, "LayerNormalization", {in, module + "weight", module + "bias"}, {name + "_output_0"},
                   {Attr{.name = "axis", .num = axis}});
    delimited(n, 5, float_attribute("epsilon", epsilon));
    add(n);
    return name + "_output_0";
  }
  Bytes model() const {
    Bytes out;
    integer(out, 1, 8);
    delimited(out, 7, graph_);
    return out;
  }

 private:
  Bytes graph_;
};

void layer_weights(Builder& b, const FakeMlp& f, int layer) {
  std::vector<float> w1, w2;
  for (int r = 0; r < 2 * kC; ++r)
    for (int c = 0; c < f.hidden; ++c) w1.push_back(weight(layer, r, c));
  for (int r = 0; r < f.hidden; ++r)
    for (int c = 0; c < kC; ++c) w2.push_back(weight(layer + 1, r, c));
  b.init(std::format("onnx::MatMul_{}0", layer), {2 * kC, f.hidden}, w1);
  b.init(std::format("onnx::MatMul_{}1", layer),
         f.w2_transposed ? std::vector<std::int64_t>{kC, f.hidden} : std::vector<std::int64_t>{f.hidden, kC},
         w2);
  b.init(std::format("onnx::MatMul_{}9", layer), {2 * kC, f.hidden}, w1);  // une copie sous un autre nom
  const std::string module = std::format("gmfss.flownet.transformer.layers.{}.cross_attn_ffn.", layer);
  for (const std::string norm : {"norm1.", "norm2."}) {
    std::vector<float> gamma(kC), beta(kC);
    for (int c = 0; c < kC; ++c) {
      gamma[static_cast<std::size_t>(c)] = 1.0F + static_cast<float>(c) * 0.0078125F;
      beta[static_cast<std::size_t>(c)] = static_cast<float>(c % 5) * 0.25F;
    }
    b.init(module + norm + "weight", {kC}, gamma);
    b.init(module + norm + "bias", {kC}, beta);
  }
}

// Une couche à une échelle : norm1 (le message), Concat, MatMul, GELU, MatMul, norm2, résiduel.
void layer_use(Builder& b, const FakeMlp& f, int layer, int scale) {
  const std::string tail = scale == 0 ? "" : "_1";
  const std::string p = std::format("/flownet/transformer/layers.{}/cross_attn_ffn/", layer);
  const std::string module = std::format("gmfss.flownet.transformer.layers.{}.cross_attn_ffn.", layer);
  const std::string source = std::format("source_{}_{}", scale, layer);
  const std::string message = b.norm(p + "norm1" + tail + "/LayerNormalization",
                                     std::format("merge_{}_{}", scale, layer), module + "norm1.", 1e-5F, -1);
  const std::string concat =
      b.simple(std::format("/flownet/transformer/layers.{}/cross_attn_ffn{}/Concat_16", layer, tail),
               "Concat", f.concat_swapped ? std::vector{message, source} : std::vector{source, message},
               {Attr{.name = "axis", .num = f.concat_axis}});
  std::string w1 = std::format("onnx::MatMul_{}{}", layer, scale == 1 && f.quarter_weights_differ ? 9 : 0);
  std::string w2 = std::format("onnx::MatMul_{}1", layer);
  if (scale == 1) {  // le second appel lit les poids par un Identity, comme l'export
    w1 = b.simple(std::format("Identity_{}_0", layer), "Identity", {w1});
    w2 = b.simple(std::format("Identity_{}_1", layer), "Identity", {w2});
  }
  const std::string h = b.simple(p + "mlp/mlp.0" + tail + "/MatMul", "MatMul", {concat, w1});
  const std::string gelu = p + "mlp/mlp.1" + tail + "/";
  const std::string div = b.simple(gelu + "Div", "Div", {h, b.constant(gelu + "Constant", f.sqrt2)});
  const std::string erf = b.simple(gelu + "Erf", "Erf", {div});
  const std::string one = b.simple(gelu + "Add", "Add", {erf, b.constant(gelu + "Constant_1", f.add_one)});
  const std::string mul = b.simple(gelu + "Mul", "Mul", {h, one});
  const std::string g = b.simple(gelu + "Mul_1", "Mul", {mul, b.constant(gelu + "Constant_2", f.half)});
  if (f.hidden_read_elsewhere) b.simple(p + "extra" + tail + "/Relu", "Relu", {h});
  std::string o = b.simple(p + "mlp/mlp.2" + tail + "/MatMul", "MatMul", {g, w2});
  if (f.bias_after_mlp) o = b.simple(p + "mlp/mlp.2" + tail + "/Add", "Add", {o, module + "norm2.bias"});
  const std::string n2 = b.norm(p + "norm2" + tail + "/LayerNormalization", o, module + "norm2.",
                                scale == 1 ? f.quarter_epsilon : 1e-5F, f.norm_axis);
  b.simple(std::format("/flownet/transformer/layers.{}/cross_attn_ffn{}/Add", layer, tail), "Add",
           f.residual_swapped ? std::vector{n2, source} : std::vector{source, n2});
}

Bytes fake_mlp(const FakeMlp& f) {
  Builder b;
  for (int layer = 0; layer < 6; ++layer) layer_weights(b, f, layer);
  for (int scale = 0; scale < (f.no_quarter ? 1 : 2); ++scale)
    for (int layer = 0; layer < 6; ++layer) layer_use(b, f, layer, scale);
  return b.model();
}

}  // namespace

TEST_CASE("GMFlow natif : le MLP du faux transformeur se lit, rangé pour le noyau", "[prepare][gmflow]") {
  const auto mlp = gmflow_mlp(fake_mlp(FakeMlp{}));
  INFO((mlp ? std::string() : mlp.error().detail));
  REQUIRE(mlp);
  CHECK(mlp->channels == kC);
  CHECK(mlp->hidden == 64);
  REQUIRE(mlp->layers.size() == 6);
  REQUIRE(mlp->uses.size() == 12);
  // w1 [caché][256] depuis [256][caché] ; w2 [128][caché] depuis [caché][128].
  const GmflowMlpLayer& l3 = mlp->layers[3];
  CHECK(l3.w1[static_cast<std::size_t>(17 * 256 + 200)] == weight(3, 200, 17));
  CHECK(l3.w2[static_cast<std::size_t>(100 * 64 + 9)] == weight(4, 9, 100));
  CHECK(l3.gamma[2] == 1.015625F);
  CHECK(l3.beta[7] == 0.5F);
  CHECK(l3.epsilon == 1e-5F);
  const GmflowMlpUse& use = mlp->uses[7];  // le quart, couche 1
  CHECK((use.scale == 1 && use.layer == 1));
  CHECK(use.source == "source_1_1");
  CHECK(use.message == "/flownet/transformer/layers.1/cross_attn_ffn/norm1_1/LayerNormalization_output_0");
  CHECK(use.output == "/flownet/transformer/layers.1/cross_attn_ffn_1/Add_output_0");
  // La paire du lecteur est en FP16 : √2 y vaut 1,4140625, la GELU reste la même.
  CHECK(gmflow_mlp(fake_mlp({.sqrt2 = 1.4140625F})));
}

TEST_CASE("GMFlow natif : les fautes du faux MLP sont refusées", "[prepare][gmflow]") {
  const auto refused = [](const FakeMlp& f) {
    const auto mlp = gmflow_mlp(fake_mlp(f));
    return !mlp && mlp.error().code == RefusalCode::FieldMissing;
  };
  CHECK(refused({.sqrt2 = 1.5F}));  // une autre GELU
  CHECK(refused({.add_one = 2.0F}));
  CHECK(refused({.half = 0.25F}));
  CHECK(refused({.concat_axis = 0}));
  CHECK(refused({.norm_axis = 0}));
  CHECK(refused({.quarter_epsilon = 1e-3F}));
  CHECK(refused({.concat_swapped = true}));  // [message, x] : W1 ne lirait plus le bon côté
  CHECK(refused({.residual_swapped = true}));
  CHECK(refused({.hidden_read_elsewhere = true}));  // la couche cachée doit rester sur la puce
  CHECK(refused({.quarter_weights_differ = true}));
  CHECK(refused({.bias_after_mlp = true}));
  CHECK(refused({.w2_transposed = true}));
  CHECK(refused({.no_quarter = true}));
  CHECK(refused({.hidden = 96}));  // pas par tranches de 64
  const auto garbage = gmflow_mlp(std::vector<std::byte>(16, std::byte{0x7f}));
  REQUIRE_FALSE(garbage);
  CHECK(garbage.error().code == RefusalCode::ModelUnreadable);
}

TEST_CASE("GMFlow natif : le MLP se tient, ses fautes sont refusées", "[prepare][gmflow]") {
  const auto read = gmflow_mlp(fake_mlp(FakeMlp{}));
  REQUIRE(read);
  CHECK(gmflow_mlp_check(*read));
  const auto broken = [&](auto&& change) {
    GmflowMlp m = *read;
    change(m);
    const auto ok = gmflow_mlp_check(m);
    return !ok && ok.error().code == RefusalCode::FieldMissing;
  };
  CHECK(broken([](GmflowMlp& m) { m.channels = 64; }));
  CHECK(broken([](GmflowMlp& m) { m.hidden = 0; }));
  CHECK(broken([](GmflowMlp& m) { m.layers[2].w1.pop_back(); }));
  CHECK(broken([](GmflowMlp& m) { m.layers[1].w2[5] = std::numeric_limits<float>::quiet_NaN(); }));
  CHECK(broken([](GmflowMlp& m) { m.layers[0].w1[3] = 1e5F; }));  // hors du FP16
  CHECK(broken([](GmflowMlp& m) { m.layers[4].epsilon = 0.0F; }));
  CHECK(broken([](GmflowMlp& m) { m.layers[5].gamma.resize(64); }));
  CHECK(broken([](GmflowMlp& m) { m.uses[3].layer = 6; }));
  CHECK(broken([](GmflowMlp& m) { m.uses[4] = m.uses[5]; }));  // deux fois le même emploi
  CHECK(broken([](GmflowMlp& m) { m.uses.clear(); }));
}

TEST_CASE("GMFlow natif : le MLP de la paire exportée se lit", "[prepare][gmflow]") {
  const std::filesystem::path model =
      std::filesystem::path(KIRARI_RIFE_MODEL_DIR) / "gmfss_fortuna_union_pair.onnx";
  if (!std::filesystem::exists(model)) SKIP("gmfss_fortuna_union_pair.onnx absent");
  auto bytes = read_onnx_bytes(model);
  REQUIRE(bytes);
  const auto mlp = gmflow_mlp(*bytes);
  INFO((mlp ? std::string() : mlp.error().detail));
  REQUIRE(mlp);
  CHECK(mlp->hidden == 1024);
  CHECK(mlp->layers.size() == 6);
  REQUIRE(mlp->uses.size() == 12);
  CHECK(std::abs(mlp->layers[0].epsilon - 1e-5F) < 1e-9F);
  CHECK(mlp->uses[0].source == "/flownet/transformer/layers.0/self_attn/Add_output_0");
  CHECK(mlp->uses[11].output == "/flownet/transformer/layers.5/cross_attn_ffn_1/Add_17_output_0");
}

}  // namespace kirari::prepare
