// FP8 E4M3 : arrondi, décodage des zéros ONNX et fusion des paires Q/DQ en FakeQuantE4M3.
#include "gpu/ir/fp8.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/passes.h"
#include "test_util.h"

using ajp::RefusalCode;
using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

constexpr std::int32_t kFp8Type = onnx::kFloat8E4M3Fn;

// Zéro FP8 scalaire (octet 0) en raw_data.
ProtoWriter fp8_zero(std::string_view name) {
  return tensor_header(name, kFp8Type, {}).bytes(9, std::string(1, '\0'));
}

ModelBuilder quantized_model(float q_scale, float dq_scale) {
  ModelBuilder m(19);
  m.input("x", {1, 1, 2, 4});
  m.weight("sq", {}, {q_scale})
      .weight("sd", {}, {dq_scale})
      .initializer(fp8_zero("zq"))
      .initializer(fp8_zero("zd"));
  m.node("QuantizeLinear", {"x", "sq", "zq"}, {"q"});
  m.node("DequantizeLinear", {"q", "sd", "zd"}, {"y"});
  m.output("y");
  return m;
}

}  // namespace

TEST_CASE("fp8 : les 254 valeurs finies d'E4M3 font l'aller-retour", "[gpu_ir][fp8]") {
  for (unsigned bits = 0; bits < 256; ++bits) {
    if ((bits & 0x7FU) == 0x7FU) continue;  // NaN
    const float value = e4m3_to_float(static_cast<std::uint8_t>(bits));
    const std::uint8_t back = float_to_e4m3(value);
    if (value == 0.0F) {
      CHECK((back & 0x7FU) == 0);
    } else {
      CHECK(back == bits);
    }
  }
  CHECK(e4m3_to_float(0x7E) == 448.0F);
  CHECK(e4m3_to_float(0x01) == std::ldexp(1.0F, -9));
}

TEST_CASE("fp8 : arrondi au plus proche pair, saturation à 448", "[gpu_ir][fp8]") {
  CHECK(round_to_e4m3(1.06F) == 1.0F);
  CHECK(round_to_e4m3(1.07F) == 1.125F);
  CHECK(round_to_e4m3(1.0625F) == 1.0F);   // à égale distance : mantisse paire
  CHECK(round_to_e4m3(1.1875F) == 1.25F);  // idem, vers le haut
  CHECK(round_to_e4m3(500.0F) == 448.0F);
  CHECK(round_to_e4m3(-1.0e9F) == -448.0F);
  CHECK(round_to_e4m3(std::ldexp(1.0F, -11)) == 0.0F);
  CHECK(std::isnan(round_to_e4m3(std::nanf(""))));
}

TEST_CASE("fp8 : une paire Q/DQ devient un FakeQuantE4M3 et calcule l'arrondi", "[gpu_ir][fp8]") {
  const Graph graph = build(quantized_model(0.5F, 0.5F));
  REQUIRE(graph.nodes.size() == 1);
  CHECK(graph.nodes[0].op == OpKind::FakeQuantE4M3);
  CHECK(count_ops(graph, OpKind::FakeQuantE4M3) == 1);
  for (const Tensor& t : graph.tensors) CHECK(t.name != "zq");

  const std::vector<float> in = {0.0F, 0.1F, 0.26F, 1.0F, 3.3F, -7.7F, 300.0F, 1.0e6F};
  const Value out = execute1(graph, {{"x", floats({1, 1, 2, 4}, in)}});
  for (std::size_t k = 0; k < in.size(); ++k) {
    CHECK(out.f[k] == round_to_e4m3(in[k] / 0.5F) * 0.5F);
  }
  CHECK(out.f[7] == 224.0F);  // 448 × 0,5
}

TEST_CASE("fp8 : des échelles différentes laissent les Q/DQ, refusés à l'import", "[gpu_ir][fp8]") {
  auto graph = load_onnx(quantized_model(0.5F, 0.25F).bytes());
  REQUIRE_FALSE(graph);
  CHECK(graph.error().code == RefusalCode::OperatorUnknown);
  CHECK(graph.error().detail == "QuantizeLinear");
}

TEST_CASE("fp8 : les poids quantifiés se plient en constantes arrondies", "[gpu_ir][fp8]") {
  ModelBuilder m(19);
  m.input("x", {1, 1, 4, 4});
  const std::vector<float> w = {0.3F, -0.7F, 1.9F, 0.01F};
  m.weight("w", {1, 1, 2, 2}, w)
      .weight("s", {}, {0.01F})
      .initializer(fp8_zero("z1"))
      .initializer(fp8_zero("z2"));
  m.node("QuantizeLinear", {"w", "s", "z1"}, {"wq"});
  m.node("DequantizeLinear", {"wq", "s", "z2"}, {"wd"});
  m.node("Conv", {"x", "wd"}, {"y"});
  m.output("y");
  Graph graph = build(m);
  REQUIRE(optimize(graph));
  CHECK(count_ops(graph, OpKind::FakeQuantE4M3) == 0);
  const TensorId id = find_tensor(graph, "wd").value_or(kNoTensor);
  REQUIRE(id != kNoTensor);
  REQUIRE(graph.tensors[id].kind == TensorKind::Constant);
  for (std::size_t k = 0; k < w.size(); ++k) {
    CHECK(graph.tensors[id].value.f[k] == round_to_e4m3(w[k] / 0.01F) * 0.01F);
  }
}

TEST_CASE("fp8 : un zéro FP8 en int32_data se décode comme en raw_data", "[gpu_ir][fp8]") {
  const std::vector<std::int64_t> words = {0x38, 0x7E};  // 1,0 et 448
  ProtoWriter tensor = tensor_header("z", kFp8Type, std::vector<std::int64_t>{2});
  tensor.packed_varints(5, words);
  ModelBuilder m(19);
  m.input("x", {1}).initializer(tensor).node("Relu", {"x"}, {"y"}).output("y");
  auto model = onnx::read_onnx(m.bytes());
  REQUIRE(model);
  REQUIRE(model->graph.initializers.size() == 1);
  CHECK(model->graph.initializers[0].floats == std::vector<float>{1.0F, 448.0F});
}
