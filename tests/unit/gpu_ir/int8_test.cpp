// INT8 : arrondi, décodage des initialiseurs ONNX INT8, fusion des paires Q/DQ en FakeQuantInt8, poids rangés
// en entiers (DequantizeLinear seul) et poids retrouvés pour les noyaux à matrices INT8.
#include "gpu/ir/int8.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/fp8.h"
#include "gpu/ir/fp8_weights.h"
#include "gpu/ir/passes.h"
#include "kernels/ref/half.h"
#include "test_util.h"

using namespace kirari::gpu::ir;
using namespace kirari::gpu::ir::test;

namespace {

constexpr std::int32_t kInt8Type = onnx::kInt8;

// Zéro INT8 scalaire (octet 0) en raw_data.
ProtoWriter int8_zero(std::string_view name) {
  return tensor_header(name, kInt8Type, {}).bytes(9, std::string(1, '\0'));
}

// Des entiers INT8 en raw_data (complément à deux).
ProtoWriter int8_values(std::string_view name, const std::vector<std::int64_t>& dims,
                        const std::vector<int>& values) {
  std::string raw;
  for (const int v : values) raw.push_back(static_cast<char>(static_cast<std::uint8_t>(v & 0xFF)));
  return tensor_header(name, kInt8Type, dims).bytes(9, raw);
}

ModelBuilder quantized_model(float scale) {
  ModelBuilder m(19);
  m.input("x", {1, 1, 2, 4});
  m.weight("s", {}, {scale}).initializer(int8_zero("zq")).initializer(int8_zero("zd"));
  m.node("QuantizeLinear", {"x", "s", "zq"}, {"q"});
  m.node("DequantizeLinear", {"q", "s", "zd"}, {"y"});
  m.output("y");
  return m;
}

const ConvAttrs& first_conv(const Graph& graph) {
  for (const Node& node : graph.nodes) {
    if (node.op == OpKind::Conv) return std::get<ConvAttrs>(node.attrs);
  }
  FAIL("Conv absente");
  return std::get<ConvAttrs>(graph.nodes[0].attrs);
}

}  // namespace

TEST_CASE("int8 : arrondi au plus proche pair, saturation à [-128, 127], NaN à 0", "[gpu_ir][int8]") {
  CHECK(round_to_int8(0.5F) == 0.0F);
  CHECK(round_to_int8(1.5F) == 2.0F);
  CHECK(round_to_int8(-2.5F) == -2.0F);
  CHECK(round_to_int8(126.6F) == 127.0F);
  CHECK(round_to_int8(300.0F) == 127.0F);
  CHECK(round_to_int8(-128.4F) == -128.0F);
  CHECK(round_to_int8(-1000.0F) == -128.0F);
  CHECK(round_to_int8(std::numeric_limits<float>::infinity()) == 127.0F);
  CHECK(round_to_int8(std::numeric_limits<float>::quiet_NaN()) == 0.0F);
  CHECK(quant_round(QuantGrid::Int8, 0.26F, 0.1F) == round_to_int8(0.26F / 0.1F) * 0.1F);
  CHECK(quant_round(QuantGrid::E4m3, 0.26F, 0.1F) == round_to_e4m3(0.26F / 0.1F) * 0.1F);
}

TEST_CASE("int8 : une paire Q/DQ devient un FakeQuantInt8 et calcule l'arrondi", "[gpu_ir][int8]") {
  const Graph graph = build(quantized_model(0.05F));
  REQUIRE(graph.nodes.size() == 1);
  CHECK(graph.nodes[0].op == OpKind::FakeQuantInt8);
  const std::vector<float> x = {0.0F, 0.024F, 0.026F, -0.075F, 3.0F, 6.36F, -7.0F, 1.0F};
  const Value y = execute1(graph, {{"x", floats({1, 1, 2, 4}, x)}});
  REQUIRE(y.f.size() == x.size());
  for (std::size_t k = 0; k < x.size(); ++k) {
    INFO(x[k]);
    CHECK(y.f[k] == round_to_int8(x[k] / 0.05F) * 0.05F);
  }
  CHECK(y.f[6] == -128.0F * 0.05F);  // saturé
  CHECK(y.f[5] == 127.0F * 0.05F);
}

TEST_CASE("int8 : un DequantizeLinear sans zéro après un QuantizeLinear INT8 forme la paire",
          "[gpu_ir][int8]") {
  ModelBuilder m(19);
  m.input("x", {1, 1, 2, 4});
  m.weight("s", {}, {0.1F}).initializer(int8_zero("zq"));
  m.node("QuantizeLinear", {"x", "s", "zq"}, {"q"});
  m.node("DequantizeLinear", {"q", "s"}, {"y"});
  m.output("y");
  const Graph graph = build(m);
  CHECK(count_ops(graph, OpKind::FakeQuantInt8) == 1);
}

TEST_CASE("int8 : des poids rangés en entiers par canal se plient en q × échelle", "[gpu_ir][int8]") {
  for (const bool with_zero : {true, false}) {
    ModelBuilder m(19);
    m.input("x", {1, 2, 4, 4});
    const std::vector<int> q = {-128, 127, 5, -3, 0, 64};
    const std::vector<float> scale = {0.01F, 0.1F, 0.002F};
    m.initializer(int8_values("wq", {3, 2, 1, 1}, q)).weight("s", {3}, scale);
    m.initializer(int8_values("z", {3}, {0, 0, 0}));
    if (with_zero) {
      m.node("DequantizeLinear", {"wq", "s", "z"}, {"wd"}, {attr_int("axis", 0)});
    } else {
      m.node("DequantizeLinear", {"wq", "s"}, {"wd"}, {attr_int("axis", 0)});
    }
    m.node("Conv", {"x", "wd"}, {"y"});
    m.output("y");
    const Graph graph = build(m);
    const TensorId id = find_tensor(graph, "wd").value_or(kNoTensor);
    REQUIRE(id != kNoTensor);
    REQUIRE(graph.tensors[id].kind == TensorKind::Constant);
    for (std::size_t k = 0; k < q.size(); ++k)
      CHECK(graph.tensors[id].value.f[k] == static_cast<float>(q[k]) * scale[k / 2]);
    // Les entiers retrouvés sont ceux de l'initialiseur, l'échelle celle de chaque canal.
    const auto found = recover_int8_weights(graph.tensors[id].value);
    REQUIRE(found);
    for (std::size_t k = 0; k < q.size(); ++k) {
      CHECK(static_cast<std::int8_t>(found->bytes[k]) * found->scale[k / 2] == graph.tensors[id].value.f[k]);
    }
  }
}

TEST_CASE("int8 : les FakeQuantInt8 entrent dans les Conv sur la grille INT8, sans changer le calcul",
          "[gpu_ir][int8]") {
  ModelBuilder m(19);
  m.input("x", {1, 4, 6, 6});
  m.weight("s1", {}, {0.02F}).weight("s2", {}, {0.05F});
  for (const char* z : {"z1", "z2", "z3", "z4"}) m.initializer(int8_zero(z));
  m.weight("w1", {4, 4, 3, 3}, sequence(4 * 4 * 3 * 3, -0.5F, 0.5F));
  m.weight("b1", {4}, {0.1F, -0.2F, 0.05F, 0.0F});
  m.node("QuantizeLinear", {"x", "s1", "z1"}, {"xq"});
  m.node("DequantizeLinear", {"xq", "s1", "z2"}, {"xd"});
  m.node("Conv", {"xd", "w1", "b1"}, {"t1"}, {attr_ints("pads", {1, 1, 1, 1})});
  m.node("QuantizeLinear", {"t1", "s2", "z3"}, {"t1q"});
  m.node("DequantizeLinear", {"t1q", "s2", "z4"}, {"y"});
  m.output("y");
  const Graph before = build(m);
  REQUIRE(count_ops(before, OpKind::FakeQuantInt8) == 2);
  Graph after = before;
  REQUIRE(optimize(after));
  CHECK(count_ops(after, OpKind::FakeQuantInt8) == 0);
  CHECK(first_conv(after).quant_grid == QuantGrid::Int8);
  CHECK(first_conv(after).in_quant == 0.02F);
  CHECK(first_conv(after).out_quant == 0.05F);
  const std::vector<NamedValue> inputs{{"x", random_floats({1, 4, 6, 6}, 3, -3.0F, 3.0F)}};
  CHECK(max_abs_diff(execute1(before, inputs).f, execute1(after, inputs).f) == 0.0F);
}

TEST_CASE("int8 : une Conv ne mêle pas les grilles FP8 et INT8", "[gpu_ir][int8]") {
  ModelBuilder m(19);
  m.input("x", {1, 4, 6, 6});
  m.weight("s1", {}, {0.02F}).weight("s2", {}, {0.05F});
  m.initializer(int8_zero("z1")).initializer(int8_zero("z2"));
  m.initializer(tensor_header("f1", onnx::kFloat8E4M3Fn, {}).bytes(9, std::string(1, '\0')));
  m.initializer(tensor_header("f2", onnx::kFloat8E4M3Fn, {}).bytes(9, std::string(1, '\0')));
  m.weight("w1", {4, 4, 1, 1}, sequence(16, -0.5F, 0.5F));
  m.node("QuantizeLinear", {"x", "s1", "z1"}, {"xq"});
  m.node("DequantizeLinear", {"xq", "s1", "z2"}, {"xd"});
  m.node("Conv", {"xd", "w1"}, {"t1"});
  m.node("QuantizeLinear", {"t1", "s2", "f1"}, {"t1q"});
  m.node("DequantizeLinear", {"t1q", "s2", "f2"}, {"y"});
  m.output("y");
  const Graph before = build(m);
  Graph after = before;
  REQUIRE(optimize(after));
  CHECK(first_conv(after).quant_grid == QuantGrid::Int8);
  CHECK(first_conv(after).in_quant == 0.02F);
  CHECK(first_conv(after).out_quant == 0.0F);  // l'arrondi FP8 de sortie reste un nœud à part
  CHECK(count_ops(after, OpKind::FakeQuantE4M3) == 1);
  const std::vector<NamedValue> inputs{{"x", random_floats({1, 4, 6, 6}, 5, -3.0F, 3.0F)}};
  CHECK(max_abs_diff(execute1(before, inputs).f, execute1(after, inputs).f) == 0.0F);
}

TEST_CASE("int8 : les entiers et les échelles des poids se retrouvent", "[gpu_ir][int8]") {
  const Dims dims{3, 2, 3, 3};
  const std::vector<float> raw = sequence(count_of(dims), -2.0F, 2.0F, 9);
  const std::size_t per = raw.size() / 3;
  // Canal 0 : amax / 127 ; canal 1 : une échelle par tenseur, son plus grand entier sous 127 ; canal 2 :
  // le plus grand en valeur absolue vaut -128 × échelle.
  for (const bool half : {false, true}) {
    Value w = floats(dims, raw);
    const std::vector<float> scales = {0.013F, 0.031F, 0.0077F};
    for (std::size_t k = 0; k < raw.size(); ++k) {
      const float s = scales[k / per];
      w.f[k] = round_to_int8(raw[k] / s) * s;
      if (half) w.f[k] = kirari::kernels::ref::round_to_half(w.f[k]);
    }
    w.f[2 * per] = -128.0F * scales[2];
    if (half) w.f[2 * per] = kirari::kernels::ref::round_to_half(w.f[2 * per]);
    const auto found = recover_int8_weights(w);
    REQUIRE(found);
    for (std::size_t k = 0; k < w.f.size(); ++k) {
      const float rebuilt =
          static_cast<float>(static_cast<std::int8_t>(found->bytes[k])) * found->scale[k / per];
      CHECK(std::fabs(rebuilt - w.f[k]) <= std::ldexp(std::fabs(w.f[k]), -10) + 0x1p-25F);
    }
    CHECK(static_cast<std::int8_t>(found->bytes[2 * per]) == -128);
  }
  // Des poids quelconques (hors grille) ne sont pas retrouvés, ni des poids qui ne sont pas de rang 4.
  CHECK_FALSE(recover_int8_weights(floats(dims, raw)));
  CHECK_FALSE(recover_int8_weights(floats({4}, {1.0F, 2.0F, 3.0F, 4.0F})));
  // Un canal nul se retrouve, d'échelle 1.
  const auto zeros = recover_int8_weights(floats({2, 1, 1, 1}, {0.0F, 0.0F}));
  REQUIRE(zeros);
  CHECK(zeros->scale[0] == 1.0F);
}
