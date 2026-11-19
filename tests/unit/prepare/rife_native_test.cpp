// Les poids d'un morceau de RIFE lus pour les noyaux natifs (prepare/rife_native.h) : un petit bloc écrit à
// la main (Conv de pas 2 et LeakyRelu, ResConv à bêta, ConvTranspose), le rangement des poids, et les refus.
#include "prepare/rife_native.h"

#include <bit>
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
  for (unsigned i = 0; i < 4; ++i) out.push_back(static_cast<std::byte>((bits >> (8U * i)) & 0xffU));
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
  std::vector<std::int64_t> beta_dims{1, 8, 1, 1};
  bool output_shape = false;  // la transposée porte output_shape (ONNX ignore alors ses bords)
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
  std::vector<Bytes> transposed{ints_attribute("strides", {2, 2}), ints_attribute("pads", {1, 1, 1, 1})};
  if (b.output_shape) transposed.push_back(ints_attribute("output_shape", {16, 16}));
  add(with_attributes(node("ct", "ConvTranspose", {"h1", "wt"}, {"y"}), transposed));
  delimited(graph, 5, tensor_dims("w0", {8, 4, 3, 3}, raw(b.w0)));
  delimited(graph, 5, tensor_dims("b0", {8}, raw(b.bias0)));
  delimited(graph, 5, tensor_dims("w1", {8, 8, 3, 3}, raw(b.w1)));
  delimited(graph, 5, tensor_dims("beta", b.beta_dims, raw(b.beta)));
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

TEST_CASE("Un bêta ou une transposée que la diffusion d'ONNX lirait autrement est refusé",
          "[prepare][onnx][rife_native]") {
  Block b;
  b.beta_dims = {8, 1, 1};
  CHECK(rife_native_layers(block_model(b)));
  // Une forme C (ou 1×1×1×C) se diffuse le long de la largeur, pas des canaux.
  for (const auto& dims : {std::vector<std::int64_t>{8}, std::vector<std::int64_t>{1, 1, 1, 8}}) {
    b.beta_dims = dims;
    const auto layers = rife_native_layers(block_model(b));
    REQUIRE_FALSE(layers);
    CHECK(layers.error().code == RefusalCode::FieldMissing);
  }
  b.beta_dims = {1, 8, 1, 1};
  b.output_shape = true;
  CHECK_FALSE(rife_native_layers(block_model(b)));
}

}  // namespace
}  // namespace kirari::prepare
