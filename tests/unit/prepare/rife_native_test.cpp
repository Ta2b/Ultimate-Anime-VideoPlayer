// Les poids d'un morceau de RIFE lus pour les noyaux natifs (prepare/rife_native.h) : un petit bloc écrit à
// la main (Conv de pas 2 et LeakyRelu, ResConv à bêta, ConvTranspose), le rangement des poids, et les refus.
#include "prepare/rife_native.h"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "rife_fake.h"

namespace kirari::prepare {
namespace {

// Un attribut INTS (AttributeProto.ints = 8, non empaqueté) ou FLOAT (f = 2, fixed32).
Bytes ints_attribute(std::string_view name, std::initializer_list<std::int64_t> values) {
  Bytes out;
  delimited(out, 1, text(name));
  for (const std::int64_t v : values) integer(out, 8, static_cast<std::uint64_t>(v));
  return out;
}
Bytes float_attribute(std::string_view name, float value) {
  Bytes out;
  delimited(out, 1, text(name));
  varint(out, (2U << 3U) | 5U);
  const auto bits = std::bit_cast<std::uint32_t>(value);
  for (unsigned i = 0; i < 4U; ++i) out.push_back(static_cast<std::byte>((bits >> (8U * i)) & 0xffU));
  return out;
}
Bytes int_attribute(std::string_view name, std::int64_t value) {
  Bytes out;
  delimited(out, 1, text(name));
  integer(out, 3, static_cast<std::uint64_t>(value));
  return out;
}
Bytes with_attributes(Bytes n, const std::vector<Bytes>& attrs) {
  for (const Bytes& a : attrs) delimited(n, 5, a);
  return n;
}

std::vector<float> ramp(std::size_t n, float step) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i) * step - 1.0F;
  return v;
}
Bytes raw(const std::vector<float>& values) {
  Bytes out(values.size() * 4);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto bits = std::bit_cast<std::uint32_t>(values[i]);
    std::memcpy(out.data() + 4 * i, &bits, 4);
  }
  return out;
}

struct Block {
  std::vector<float> w0 = ramp(8 * 4 * 9, 0.01F);  // [8][4][3][3]
  std::vector<float> w1 = ramp(8 * 8 * 9, 0.002F);
  std::vector<float> wt = ramp(8 * 4 * 16, 0.003F);  // [8 entrées][4 sorties][4][4]
  std::vector<float> bias0 = ramp(8, 0.1F);
  std::vector<float> beta = ramp(8, 0.05F);
  std::string middle = "LeakyRelu";  // l'activation de la première convolution
};

Bytes block_model(const Block& b) {
  Bytes graph;
  const auto add = [&](const Bytes& n) { delimited(graph, 1, n); };
  const auto conv = [&](std::string_view name, std::string_view op, std::vector<std::string> in,
                        std::string out, std::int64_t stride) {
    add(with_attributes(node(name, op, std::move(in), {std::move(out)}),
                        {ints_attribute("strides", {stride, stride}), ints_attribute("pads", {1, 1, 1, 1})}));
  };
  conv("c0", "Conv", {"x", "w0", "b0"}, "a0", 2);
  if (b.middle == "LeakyRelu")
    add(with_attributes(node("r0", "LeakyRelu", {"a0"}, {"h0"}), {float_attribute("alpha", 0.2F)}));
  else
    add(node("r0", b.middle, {"a0"}, {"h0"}));
  conv("c1", "Conv", {"h0", "w1"}, "a1", 1);
  add(node("m1", "Mul", {"a1", "beta"}, {"s1"}));
  add(node("p1", "Add", {"s1", "h0"}, {"t1"}));
  add(with_attributes(node("r1", "LeakyRelu", {"t1"}, {"h1"}), {float_attribute("alpha", 0.2F)}));
  conv("ct", "ConvTranspose", {"h1", "wt"}, "y", 2);
  delimited(graph, 5, tensor_dims("w0", {8, 4, 3, 3}, raw(b.w0)));
  delimited(graph, 5, tensor_dims("b0", {8}, raw(b.bias0)));
  delimited(graph, 5, tensor_dims("w1", {8, 8, 3, 3}, raw(b.w1)));
  delimited(graph, 5, tensor_dims("beta", {1, 8, 1, 1}, raw(b.beta)));
  delimited(graph, 5, tensor_dims("wt", {8, 4, 4, 4}, raw(b.wt)));
  delimited(graph, 11, value_info("x", 1, {1, 4, 16, 16}));
  delimited(graph, 12, value_info("y", 1, {1, 4, 16, 16}));
  Bytes model;
  integer(model, 1, 8);
  delimited(model, 7, graph);
  return model;
}

TEST_CASE("Les couches d'un bloc de RIFE se lisent pour les noyaux natifs", "[prepare][onnx][rife_native]") {
  const Block b;
  const auto layers = rife_native_layers(block_model(b));
  REQUIRE(layers);
  REQUIRE(layers->size() == 3);
  const RifeNativeLayer& first = (*layers)[0];
  CHECK_FALSE(first.transposed);
  CHECK(first.cin == 4);
  CHECK(first.cout == 8);
  CHECK(first.stride == 2);
  CHECK(first.scale.empty());
  REQUIRE(first.slope.size() == 8);
  CHECK(first.slope[3] == 0.2F);
  CHECK(first.bias == b.bias0);
  // [cout][prise][cin] depuis [cout][cin][ky][kx].
  CHECK(first.weights[(5 * 9 + 7) * 4 + 2] == b.w0[(5 * 4 + 2) * 9 + 7]);
  const RifeNativeLayer& res = (*layers)[1];
  CHECK(res.stride == 1);
  CHECK(res.scale == b.beta);
  CHECK(res.slope.size() == 8);
  CHECK(res.bias == std::vector<float>(8, 0.0F));
  const RifeNativeLayer& last = (*layers)[2];
  CHECK(last.transposed);
  CHECK(last.cin == 8);
  CHECK(last.cout == 4);
  CHECK(last.kernel == 4);
  CHECK(last.slope.empty());
  // [cout][ky·4 + kx][cin] depuis [cin][cout][ky][kx].
  CHECK(last.weights[(3 * 16 + 9) * 8 + 6] == b.wt[(6 * 4 + 3) * 16 + 9]);
  CHECK(rife_native_check(*layers, 4, 4));
  CHECK(rife_native_check(*layers, 5, 4).error().code == RefusalCode::FieldMissing);
  CHECK(rife_native_check(*layers, 4, 52).error().code == RefusalCode::FieldMissing);
}

TEST_CASE("Un morceau hors de la chaîne de convolutions est refusé", "[prepare][onnx][rife_native]") {
  Block b;
  b.middle = "Relu";
  const auto relu = rife_native_layers(block_model(b));
  REQUIRE(relu);
  CHECK((*relu)[0].slope == std::vector<float>(8, 0.0F));
  b.middle = "Sigmoid";
  const auto sigmoid = rife_native_layers(block_model(b));
  REQUIRE_FALSE(sigmoid);
  CHECK(sigmoid.error().code == RefusalCode::FieldMissing);
  const std::vector<std::byte> garbage(16, std::byte{0xff});
  CHECK_FALSE(rife_native_layers(garbage));
}

// Un ResConv séparable de TemporalFix : Conv par canal (4 groupes) puis Conv 1 × 1, bêta, résiduel,
// LeakyRelu.
Bytes separable_model(const std::vector<float>& dw, const std::vector<float>& dwb,
                      const std::vector<float>& pw, const std::vector<float>& pwb) {
  Bytes graph;
  const auto add = [&](const Bytes& n) { delimited(graph, 1, n); };
  add(with_attributes(node("dw", "Conv", {"x", "wd", "bd"}, {"a"}),
                      {ints_attribute("pads", {1, 1, 1, 1}), int_attribute("group", 4)}));
  add(with_attributes(node("pw", "Conv", {"a", "wp", "bp"}, {"b"}), {ints_attribute("pads", {0, 0, 0, 0})}));
  add(node("m", "Mul", {"b", "beta"}, {"s"}));
  add(node("p", "Add", {"s", "x"}, {"t"}));
  add(with_attributes(node("r", "LeakyRelu", {"t"}, {"y"}), {float_attribute("alpha", 0.2F)}));
  delimited(graph, 5, tensor_dims("wd", {4, 1, 3, 3}, raw(dw)));
  delimited(graph, 5, tensor_dims("bd", {4}, raw(dwb)));
  delimited(graph, 5, tensor_dims("wp", {4, 4, 1, 1}, raw(pw)));
  delimited(graph, 5, tensor_dims("bp", {4}, raw(pwb)));
  delimited(graph, 5, tensor_dims("beta", {1, 4, 1, 1}, raw(ramp(4, 0.5F))));
  delimited(graph, 11, value_info("x", 1, {1, 4, 8, 8}));
  delimited(graph, 12, value_info("y", 1, {1, 4, 8, 8}));
  Bytes model;
  integer(model, 1, 8);
  delimited(model, 7, graph);
  return model;
}

TEST_CASE("Une convolution séparable se replie en une seule 3 × 3", "[prepare][onnx][rife_native]") {
  const std::vector<float> dw = ramp(4 * 9, 0.05F);
  const std::vector<float> dwb = ramp(4, 0.25F);
  const std::vector<float> pw = ramp(16, 0.125F);
  const std::vector<float> pwb = ramp(4, 0.5F);
  const auto layers = rife_native_layers(separable_model(dw, dwb, pw, pwb));
  REQUIRE(layers);
  REQUIRE(layers->size() == 1);
  const RifeNativeLayer& l = (*layers)[0];
  CHECK(l.cin == 4);
  CHECK(l.cout == 4);
  CHECK(l.stride == 1);
  CHECK(l.scale == ramp(4, 0.5F));
  CHECK(l.slope.size() == 4);
  REQUIRE(l.weights.size() == 4U * 9U * 4U);
  // W'[o][t][c] = P[o][c] · D[c][t] ; b'[o] = Σ_c P[o][c] · d[c] + p[o].
  for (std::size_t o = 0; o < 4; ++o) {
    double b = pwb[o];
    for (std::size_t c = 0; c < 4; ++c) {
      b += static_cast<double>(pw[o * 4 + c]) * static_cast<double>(dwb[c]);
      for (std::size_t t = 0; t < 9; ++t) {
        const double want = static_cast<double>(pw[o * 4 + c]) * static_cast<double>(dw[c * 9 + t]);
        CHECK(std::abs(static_cast<double>(l.weights[(o * 9 + t) * 4 + c]) - want) < 1e-6);
      }
    }
    CHECK(std::abs(static_cast<double>(l.bias[o]) - b) < 1e-6);
  }
  CHECK(conv_chain_check(*layers, 4, 4));
  CHECK(conv_chain_check(*layers, 4, 5).error().code == RefusalCode::FieldMissing);
  CHECK(conv_chain_check(*layers, 3, 4).error().code == RefusalCode::FieldMissing);
}

}  // namespace
}  // namespace kirari::prepare
