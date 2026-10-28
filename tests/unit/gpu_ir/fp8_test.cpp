// FP8 E4M3 : arrondi, décodage des zéros ONNX et fusion des paires Q/DQ en FakeQuantE4M3.
#include "gpu/ir/fp8.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/fp8_weights.h"
#include "gpu/ir/passes.h"
#include "kernels/ref/half.h"
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

TEST_CASE("fp8 : des poids quantifiés par canal se plient avec l'échelle de leur canal", "[gpu_ir][fp8]") {
  ModelBuilder m(19);
  m.input("x", {1, 2, 4, 4});
  const std::vector<float> w = sequence(3 * 2 * 1 * 1, -2.0F, 2.0F);
  const std::vector<float> scale = {0.01F, 0.1F, 0.002F};
  m.weight("w", {3, 2, 1, 1}, w).weight("s", {3}, scale);
  m.initializer(tensor_header("z1", kFp8Type, std::vector<std::int64_t>{3}).bytes(9, std::string(3, '\0')));
  m.initializer(tensor_header("z2", kFp8Type, std::vector<std::int64_t>{3}).bytes(9, std::string(3, '\0')));
  m.node("QuantizeLinear", {"w", "s", "z1"}, {"wq"}, {attr_int("axis", 0)});
  m.node("DequantizeLinear", {"wq", "s", "z2"}, {"wd"}, {attr_int("axis", 0)});
  m.node("Conv", {"x", "wd"}, {"y"});
  m.output("y");
  const Graph graph = build(m);
  CHECK(count_ops(graph, OpKind::FakeQuantE4M3) == 0);
  const TensorId id = find_tensor(graph, "wd").value_or(kNoTensor);
  REQUIRE(id != kNoTensor);
  REQUIRE(graph.tensors[id].kind == TensorKind::Constant);
  for (std::size_t k = 0; k < w.size(); ++k) {
    const float s = scale[k / 2];
    CHECK(graph.tensors[id].value.f[k] == round_to_e4m3(w[k] / s) * s);
  }
}

namespace {

// QDQ(x) sous le nom out, de l'échelle scalaire scale_name (zéros FP8 posés à part).
void quant_dequant(ModelBuilder& m, std::string_view in, std::string_view scale, std::string_view zero_q,
                   std::string_view zero_d, std::string_view mid, std::string_view out) {
  m.node("QuantizeLinear", {in, scale, zero_q}, {mid});
  m.node("DequantizeLinear", {mid, scale, zero_d}, {out});
}

ModelBuilder two_convs() {
  ModelBuilder m(19);
  m.input("x", {1, 4, 6, 6});
  m.weight("s1", {}, {0.02F}).weight("s2", {}, {0.05F});
  for (const char* z : {"z1", "z2", "z3", "z4"}) m.initializer(fp8_zero(z));
  m.weight("w1", {4, 4, 3, 3}, sequence(4 * 4 * 3 * 3, -0.5F, 0.5F));
  m.weight("b1", {4}, {0.1F, -0.2F, 0.05F, 0.0F});
  m.weight("w2", {4, 4, 1, 1}, sequence(16, -1.0F, 1.0F, 5));
  quant_dequant(m, "x", "s1", "z1", "z2", "xq", "xd");
  m.node("Conv", {"xd", "w1", "b1"}, {"t1"}, {attr_ints("pads", {1, 1, 1, 1})});
  quant_dequant(m, "t1", "s2", "z3", "z4", "t1q", "t1d");
  m.node("Conv", {"t1d", "w2"}, {"t2"});
  m.node("Add", {"t2", "t1d"}, {"sum"});
  m.node("Relu", {"sum"}, {"y"});
  m.output("y");
  return m;
}

const ConvAttrs& conv_attrs(const Graph& graph, std::size_t which) {
  std::size_t seen = 0;
  for (const Node& node : graph.nodes) {
    if (node.op != OpKind::Conv) continue;
    if (seen++ == which) return std::get<ConvAttrs>(node.attrs);
  }
  FAIL("Conv absente");
  return std::get<ConvAttrs>(graph.nodes[0].attrs);
}

}  // namespace

TEST_CASE("fp8 : les FakeQuantE4M3 entrent dans les Conv et le résultat ne change pas", "[gpu_ir][fp8]") {
  const Graph before = build(two_convs());
  REQUIRE(count_ops(before, OpKind::FakeQuantE4M3) == 2);
  Graph after = before;
  const PassStats stats = *optimize(after);
  CHECK(stats.quants == 2);
  CHECK(count_ops(after, OpKind::FakeQuantE4M3) == 0);
  // Premier FakeQuant : seul lecteur de xd, la première Conv arrondit son entrée. Second : lu par
  // la Conv et l'Add, donc la première Conv arrondit sa sortie, après son biais.
  CHECK(conv_attrs(after, 0).in_quant == 0.02F);
  CHECK(conv_attrs(after, 0).out_quant == 0.05F);
  CHECK(conv_attrs(after, 1).in_quant == 0.0F);
  CHECK(conv_attrs(after, 1).epilogue.residual);
  CHECK(conv_attrs(after, 1).epilogue.act == Activation::Relu);

  const std::vector<NamedValue> inputs{{"x", random_floats({1, 4, 6, 6}, 3, -3.0F, 3.0F)}};
  const Value ref = execute1(before, inputs);
  const Value fused = execute1(after, inputs);
  CHECK(max_abs_diff(ref.f, fused.f) == 0.0F);
}

TEST_CASE("fp8 : les octets et les échelles des poids se retrouvent", "[gpu_ir][fp8]") {
  const Dims dims{3, 2, 3, 3};
  const std::vector<float> raw = sequence(count_of(dims), -2.0F, 2.0F, 9);
  const auto quantize = [&](const std::vector<float>& scales, bool half) {
    Value w = floats(dims, raw);
    const std::size_t per = raw.size() / 3;
    for (std::size_t k = 0; k < raw.size(); ++k) {
      const float s = scales[k / per];
      w.f[k] = round_to_e4m3(raw[k] / s) * s;
      if (half) w.f[k] = ajp::kernels::ref::round_to_half(w.f[k]);
    }
    // Le plus grand poids de chaque canal ou du tenseur sur 448 : amax / 448 = l'échelle.
    return w;
  };
  // Échelle par canal choisie comme amax / 448 : retrouvée exactement, en float comme en float16.
  std::vector<float> scales;
  for (std::size_t c = 0; c < 3; ++c) {
    float amax = 0.0F;
    for (std::size_t k = 0; k < raw.size() / 3; ++k)
      amax = std::max(amax, std::fabs(raw[(c * (raw.size() / 3)) + k]));
    scales.push_back(amax / 448.0F);
  }
  for (const bool half : {false, true}) {
    const Value w = quantize(scales, half);
    const auto found = recover_fp8_weights(w);
    REQUIRE(found);
    for (std::size_t k = 0; k < w.f.size(); ++k) {
      const float rebuilt = e4m3_to_float(found->bytes[k]) * found->scale[k / (w.f.size() / 3)];
      CHECK(std::fabs(rebuilt - w.f[k]) <= std::ldexp(std::fabs(w.f[k]), -10));
    }
  }
  // Des poids quelconques (hors grille) ne sont pas retrouvés.
  CHECK_FALSE(recover_fp8_weights(floats(dims, raw)));
  CHECK_FALSE(recover_fp8_weights(floats({4}, {1.0F, 2.0F, 3.0F, 4.0F})));
}
