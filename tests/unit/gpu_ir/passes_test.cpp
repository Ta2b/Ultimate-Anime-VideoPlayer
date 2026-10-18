// Passes : le graphe optimisé rend les mêmes sorties que le graphe importé
// (interpréteur, 1e-5), sur un réseau façon SPAN et un réseau façon RIFE.
#include "gpu/ir/passes.h"

#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/describe.h"
#include "gpu/ir/half.h"
#include "test_util.h"

using ajp::Refusal;
using ajp::RefusalCode;
using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

ProtoWriter scalar_i64(std::string_view name, std::int64_t value) {
  return tensor_i64(name, {}, std::vector<std::int64_t>{value});
}

// SPAN « unshuffle » d'AnimeJaNai en miniature : marges dynamiques au multiple
// de 2 calculées depuis Shape, pixel_unshuffle en Reshape/Transpose/Reshape,
// trois Conv 3x3 (échelle, biais, PRelu, résiduel, Relu), pixel_shuffle, rognage.
ModelBuilder span_like() {
  ModelBuilder m(17);
  m.input("x", {1, 3, -1, -1})
      .initializer(scalar_i64("i2", 2))
      .initializer(scalar_i64("i3", 3))
      .initializer(scalar_i64("two", 2))
      .ints("axis0", {0})
      .ints("z6", {0, 0, 0, 0, 0, 0})
      .ints("c13", {1, 3})
      .ints("c112", {1, 12})
      .ints("two1", {2})
      .ints("starts", {0, 0})
      .ints("axes23", {2, 3});
  m.node("Shape", {"x"}, {"sx"})
      .node("Gather", {"sx", "i2"}, {"h"})
      .node("Gather", {"sx", "i3"}, {"w"})
      .node("Mod", {"h", "two"}, {"hm"})
      .node("Sub", {"two", "hm"}, {"hs"})
      .node("Mod", {"hs", "two"}, {"ph"})
      .node("Mod", {"w", "two"}, {"wm"})
      .node("Sub", {"two", "wm"}, {"ws"})
      .node("Mod", {"ws", "two"}, {"pw"})
      .node("Unsqueeze", {"ph", "axis0"}, {"ph1"})
      .node("Unsqueeze", {"pw", "axis0"}, {"pw1"})
      .node("Concat", {"z6", "ph1", "pw1"}, {"pads"}, {attr_int("axis", 0)})
      .node("Pad", {"x", "pads"}, {"xp"}, {attr_string("mode", "reflect")})
      .node("Shape", {"xp"}, {"sp"})
      .node("Gather", {"sp", "i2"}, {"hp"})
      .node("Gather", {"sp", "i3"}, {"wp"})
      .node("Div", {"hp", "two"}, {"hh"})
      .node("Div", {"wp", "two"}, {"wh"})
      .node("Unsqueeze", {"hh", "axis0"}, {"hh1"})
      .node("Unsqueeze", {"wh", "axis0"}, {"wh1"})
      .node("Concat", {"c13", "hh1", "two1", "wh1", "two1"}, {"s6"}, {attr_int("axis", 0)})
      .node("Reshape", {"xp", "s6"}, {"t6"})
      .node("Transpose", {"t6"}, {"tt"}, {attr_ints("perm", {0, 1, 3, 5, 2, 4})})
      .node("Concat", {"c112", "hh1", "wh1"}, {"s4"}, {attr_int("axis", 0)})
      .node("Reshape", {"tt", "s4"}, {"u"});
  m.weight("w1", {16, 12, 3, 3}, sequence(16 * 12 * 9, -0.2F, 0.2F, 1))
      .weight("b1", {16}, sequence(16, -0.1F, 0.1F, 2))
      .weight("s1", {1, 16, 1, 1}, sequence(16, 0.5F, 1.5F, 3))
      .weight("a1", {16, 1, 1}, sequence(16, -0.1F, 0.1F, 4))
      .weight("slope", {16, 1, 1}, sequence(16, 0.05F, 0.3F, 5))
      .weight("w2", {16, 16, 3, 3}, sequence(16 * 16 * 9, -0.1F, 0.1F, 6))
      .weight("b2", {16}, sequence(16, -0.1F, 0.1F, 7))
      .weight("w3", {12, 16, 3, 3}, sequence(12 * 16 * 9, -0.1F, 0.1F, 8))
      .weight("b3", {12}, sequence(12, -0.1F, 0.1F, 9));
  const ProtoWriter pad1 = attr_ints("pads", {1, 1, 1, 1});
  m.node("Conv", {"u", "w1", "b1"}, {"c1"}, {pad1})
      .node("Mul", {"c1", "s1"}, {"c1s"})
      .node("Add", {"a1", "c1s"}, {"c1b"})  // constante en premier : Add commute
      .node("PRelu", {"c1b", "slope"}, {"p1"})
      .node("Conv", {"p1", "w2", "b2"}, {"c2"}, {pad1})
      .node("Add", {"p1", "c2"}, {"r2"})
      .node("Relu", {"r2"}, {"a2"})
      .node("Conv", {"a2", "w3", "b3"}, {"c3"}, {pad1});
  m.node("Concat", {"c13", "two1", "two1", "hh1", "wh1"}, {"d6"}, {attr_int("axis", 0)})
      .node("Reshape", {"c3", "d6"}, {"dt"})
      .node("Transpose", {"dt"}, {"dtt"}, {attr_ints("perm", {0, 1, 4, 2, 5, 3})})
      .node("Mul", {"hh1", "two1"}, {"H1"})
      .node("Mul", {"wh1", "two1"}, {"W1"})
      .node("Concat", {"c13", "H1", "W1"}, {"o4"}, {attr_int("axis", 0)})
      .node("Reshape", {"dtt", "o4"}, {"o"})
      .node("Unsqueeze", {"h", "axis0"}, {"h1"})
      .node("Unsqueeze", {"w", "axis0"}, {"w1d"})
      .node("Concat", {"h1", "w1d"}, {"ends"}, {attr_int("axis", 0)})
      .node("Slice", {"o", "starts", "ends", "axes23"}, {"crop"})
      .node("Identity", {"crop"}, {"y"})
      .output("y");
  return m;
}

// Synthèse de trame façon RIFE : flot estimé à basse résolution, suréchantillonné,
// deux GridSample (bord, align_corners), masque par ConvTranspose et Sigmoid.
ModelBuilder rife_like() {
  std::vector<float> base;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      base.push_back(-1.0F + (2.0F * static_cast<float>(x) / 7.0F));
      base.push_back(-1.0F + (2.0F * static_cast<float>(y) / 7.0F));
    }
  }
  const ProtoWriter pad1 = attr_ints("pads", {1, 1, 1, 1});
  const std::vector<std::int64_t> to_nhwc{0, 2, 3, 1};
  const ProtoWriter align = attr_int("align_corners", 1);
  const ProtoWriter border = attr_string("padding_mode", "border");
  ModelBuilder m(17);
  m.input("img0", {1, 3, 8, 8})
      .input("img1", {1, 3, 8, 8})
      .weight("base", {1, 8, 8, 2}, base)
      .weight("down", {4}, {1, 1, 0.5F, 0.5F})
      .weight("up", {4}, {1, 1, 2, 2})
      .weight("wf", {8, 6, 3, 3}, sequence(8 * 6 * 9, -0.3F, 0.3F, 11))
      .weight("bf", {8}, sequence(8, -0.1F, 0.1F, 12))
      .weight("wo", {4, 8, 3, 3}, sequence(4 * 8 * 9, -0.5F, 0.5F, 13))
      .weight("bo", {4}, sequence(4, -0.2F, 0.2F, 14))
      .weight("wm", {8, 1, 4, 4}, sequence(8 * 16, -0.5F, 0.5F, 15))
      .weight("k", {}, {2.0F / 7.0F})
      .weight("one", {}, {1})
      .ints("split", {2, 2});
  m.node("Concat", {"img0", "img1"}, {"cat"}, {attr_int("axis", 1)})
      .node("Resize", {"cat", "", "down"}, {"small"}, {attr_string("mode", "linear")})
      .node("Conv", {"small", "wf", "bf"}, {"f"}, {pad1})
      .node("LeakyRelu", {"f"}, {"fa"}, {attr_float("alpha", 0.2F)})
      .node("Conv", {"fa", "wo", "bo"}, {"flow_s"}, {pad1})
      .node("Resize", {"flow_s", "", "up"}, {"flow_u"}, {attr_string("mode", "linear")})
      .node("Mul", {"flow_u", "k"}, {"flow"})
      .node("Split", {"flow", "split"}, {"f0", "f1"}, {attr_int("axis", 1)})
      .node("Transpose", {"f0"}, {"g0t"}, {attr_ints("perm", to_nhwc)})
      .node("Add", {"base", "g0t"}, {"g0"})
      .node("Transpose", {"f1"}, {"g1t"}, {attr_ints("perm", to_nhwc)})
      .node("Add", {"base", "g1t"}, {"g1"})
      .node("GridSample", {"img0", "g0"}, {"w0"}, {align, border})
      .node("GridSample", {"img1", "g1"}, {"w1"}, {align, border})
      .node("ConvTranspose", {"fa", "wm"}, {"mu"}, {attr_ints("strides", {2, 2}), pad1})
      .node("Sigmoid", {"mu"}, {"mask"})
      .node("Mul", {"w0", "mask"}, {"a"})
      .node("Sub", {"one", "mask"}, {"inv"})
      .node("Mul", {"w1", "inv"}, {"b"})
      .node("Add", {"a", "b"}, {"out"})
      .output("out");
  return m;
}

const ConvAttrs& conv_attrs(const Node& node) {
  const auto* attrs = std::get_if<ConvAttrs>(&node.attrs);
  REQUIRE(attrs != nullptr);
  return *attrs;
}

// Optimise une copie et compare les sorties de l'interpréteur avant et après.
PassStats check_preserved(const Graph& before, Graph& after, const std::vector<NamedValue>& inputs) {
  after = before;
  auto stats = optimize(after);
  if (!stats) FAIL("refus " << ajp::name(stats.error().code) << " : " << stats.error().detail);
  const std::vector<NamedValue> expected = execute(before, inputs);
  const std::vector<NamedValue> got = execute(after, inputs);
  REQUIRE(got.size() == expected.size());
  for (std::size_t k = 0; k < got.size(); ++k) {
    CHECK(got[k].name == expected[k].name);
    REQUIRE(got[k].value.dims == expected[k].value.dims);
    CHECK(max_abs_diff(got[k].value.f, expected[k].value.f) < 1e-5F);
  }
  return *stats;
}

}  // namespace

TEST_CASE("passes : SPAN miniature aux dimensions paires, réduit à cinq nœuds", "[gpu_ir][passes]") {
  const Graph before = build(span_like(), {{.name = "x", .dims = {1, 3, 6, 10}}});
  Graph after;
  const PassStats stats =
      check_preserved(before, after, {{.name = "x", .value = random_floats({1, 3, 6, 10}, 21)}});

  CHECK(stats.pixel_shuffles == 2);
  CHECK(stats.bias == 2);
  CHECK(stats.residuals == 1);
  CHECK(stats.activations == 2);
  CHECK(stats.identities >= 3);  // Pad nul, Slice entier, Identity
  REQUIRE(after.nodes.size() == 5);
  CHECK(after.nodes[0].op == OpKind::SpaceToDepth);
  CHECK(std::get<BlockAttrs>(after.nodes[0].attrs).mode == BlockMode::Crd);
  CHECK(after.nodes[4].op == OpKind::DepthToSpace);
  CHECK(std::get<BlockAttrs>(after.nodes[4].attrs).mode == BlockMode::Crd);
  CHECK(count_ops(after, OpKind::Conv) == 3);

  const ConvAttrs& first = conv_attrs(after.nodes[1]);
  CHECK(first.epilogue.act == Activation::PRelu);
  CHECK_FALSE(first.epilogue.residual);
  const ConvAttrs& second = conv_attrs(after.nodes[2]);
  CHECK(second.epilogue.residual);
  CHECK(second.epilogue.act == Activation::Relu);
  CHECK(conv_attrs(after.nodes[3]).epilogue.act == Activation::None);

  // La sortie garde son nom, l'entrée et la sortie restent en Nchw.
  CHECK(after.tensors[after.outputs[0]].name == "y");
  CHECK(after.tensors[after.inputs[0]].layout == Layout::Nchw);
  CHECK(after.tensors[after.outputs[0]].layout == Layout::Nchw);
  CHECK(stats.nhwc == 4);
  const Tensor& unshuffled = after.tensors[after.nodes[0].outputs[0]];
  CHECK(unshuffled.layout == Layout::Nhwc16);
  CHECK(storage_bytes(unshuffled) == 3 * 5 * 16 * 4);  // 12 canaux rangés sur 16
}

TEST_CASE("passes : SPAN miniature aux dimensions impaires, marges et rognage gardés", "[gpu_ir][passes]") {
  const Graph before = build(span_like(), {{.name = "x", .dims = {1, 3, 5, 7}}});
  Graph after;
  const PassStats stats =
      check_preserved(before, after, {{.name = "x", .value = random_floats({1, 3, 5, 7}, 22)}});
  CHECK(stats.pixel_shuffles == 2);
  REQUIRE(after.nodes.size() == 7);
  CHECK(after.nodes[0].op == OpKind::Pad);
  CHECK(after.nodes[6].op == OpKind::Slice);
  CHECK(after.tensors[after.outputs[0]].dims == Dims{1, 3, 5, 7});
  CHECK(count_ops(after, OpKind::Shape) == 0);
  CHECK(count_ops(after, OpKind::Reshape) == 0);
}

TEST_CASE("passes : RIFE miniature, GridSample et ConvTranspose préservés", "[gpu_ir][passes]") {
  const Graph before = build(rife_like());
  Graph after;
  const PassStats stats = check_preserved(before, after,
                                          {{.name = "img0", .value = random_floats({1, 3, 8, 8}, 31, 0, 1)},
                                           {.name = "img1", .value = random_floats({1, 3, 8, 8}, 32, 0, 1)}});
  CHECK(stats.activations == 1);  // Conv + LeakyRelu ; Sigmoid suit une ConvTranspose
  CHECK(count_ops(after, OpKind::GridSample) == 2);
  CHECK(count_ops(after, OpKind::LeakyRelu) == 0);
  CHECK(count_ops(after, OpKind::Sigmoid) == 1);
  // La grille [N, H, W, 2] n'a pas de canaux : elle reste en Nchw.
  for (const Node& node : after.nodes) {
    if (node.op == OpKind::GridSample) CHECK(after.tensors[node.inputs[1]].layout == Layout::Nchw);
  }
}

TEST_CASE("passes : constantes F16 fondues, type gardé", "[gpu_ir][passes]") {
  ModelBuilder m;
  m.input("x", {1, 2, 4, 4}, onnx::kFloat16)
      .initializer(tensor_f16("w", {4, 2, 3, 3}, sequence(72, -0.5F, 0.5F, 41)))
      .initializer(tensor_f16("s", {1, 4, 1, 1}, std::vector<float>{0.5F, 2, -1, 0.25F}))
      .node("Conv", {"x", "w"}, {"c"}, {attr_ints("pads", {1, 1, 1, 1})})
      .node("Mul", {"c", "s"}, {"m"})
      .node("Sigmoid", {"m"}, {"y"})
      .output("y");
  const Graph before = build(m);
  Graph after;
  std::vector<float> input = sequence(32, -1, 1, 42);
  for (float& v : input) v = round_to_half(v);
  const PassStats stats =
      check_preserved(before, after, {{.name = "x", .value = floats({1, 2, 4, 4}, input, DataType::F16)}});
  CHECK(stats.bias == 1);
  CHECK(stats.activations == 1);
  REQUIRE(after.nodes.size() == 1);
  CHECK(after.tensors[after.nodes[0].inputs[kConvW]].type == DataType::F16);
  CHECK(after.tensors[after.outputs[0]].type == DataType::F16);
}

TEST_CASE("passes : Sub, Div, Clip et Tanh fondus, résiduel suivi d'une activation", "[gpu_ir][passes]") {
  ModelBuilder m;
  m.input("x", {1, 3, 5, 5})
      .weight("w1", {4, 3, 3, 3}, sequence(108, -0.4F, 0.4F, 51))
      .weight("w2", {4, 4, 1, 1}, sequence(16, -0.4F, 0.4F, 52))
      .weight("w3", {4, 4, 3, 3}, sequence(144, -0.4F, 0.4F, 53))
      .weight("sub", {4, 1, 1}, {0.1F, -0.2F, 0.3F, 0})
      .weight("div", {}, {0.5F})
      .weight("lo", {}, {-0.25F})
      .weight("hi", {}, {0.25F})
      .node("Conv", {"x", "w1"}, {"c1"}, {attr_ints("pads", {1, 1, 1, 1})})
      .node("Sub", {"c1", "sub"}, {"s1"})
      .node("Div", {"s1", "div"}, {"d1"})
      .node("Clip", {"d1", "lo", "hi"}, {"k1"})
      .node("Conv", {"k1", "w2"}, {"c2"})
      .node("Add", {"c2", "k1"}, {"r2"})
      .node("Tanh", {"r2"}, {"t2"})
      .node("Conv", {"t2", "w3"}, {"c3"}, {attr_ints("pads", {1, 1, 1, 1})})
      .node("Sub", {"sub", "c3"}, {"y"})  // constante à gauche d'un Sub : non commutatif, gardé
      .output("y");
  const Graph before = build(m);
  Graph after;
  const PassStats stats =
      check_preserved(before, after, {{.name = "x", .value = random_floats({1, 3, 5, 5}, 54)}});
  CHECK(stats.bias == 2);
  CHECK(stats.residuals == 1);
  CHECK(stats.activations == 2);
  REQUIRE(after.nodes.size() == 4);
  CHECK(conv_attrs(after.nodes[0]).epilogue.act == Activation::Clip);
  CHECK(conv_attrs(after.nodes[0]).epilogue.hi == 0.25F);
  CHECK(conv_attrs(after.nodes[1]).epilogue.residual);
  CHECK(conv_attrs(after.nodes[1]).epilogue.act == Activation::Tanh);
  CHECK(after.nodes[3].op == OpKind::Sub);
  const std::string text = describe_nodes(after);
  CHECK(text.find("k3x3 s1x1 g1 +biais +Clip") != std::string::npos);
  CHECK(text.find("+résiduel +Tanh") != std::string::npos);
}

TEST_CASE("passes : code mort, nœud sans effet entre entrée et sortie", "[gpu_ir][passes]") {
  ModelBuilder m;
  m.input("x", {2, 3}).node("Relu", {"x"}, {"y"}).node("Tanh", {"x"}, {"dead"}).output("y");
  Graph graph = build(m);
  CHECK(eliminate_dead_code(graph) == 1);
  CHECK(graph.nodes.size() == 1);
  CHECK_FALSE(find_tensor(graph, "dead"));

  // Identity d'une entrée vers une sortie : rien ne peut écrire la sortie à sa place.
  ModelBuilder copy;
  copy.input("x", {2}).node("Identity", {"x"}, {"y"}).output("y");
  Graph kept = build(copy);
  REQUIRE(optimize(kept));
  CHECK(kept.nodes.size() == 1);
  CHECK(execute1(kept, {{.name = "x", .value = floats({2}, {1, 2})}}).f == std::vector<float>{1, 2});
}

TEST_CASE("passes : un graphe entièrement constant se plie", "[gpu_ir][passes]") {
  ModelBuilder m;
  m.input("x", {3})
      .weight("a", {3}, {1, 2, 3})
      .weight("b", {3}, {4, 5, 6})
      .node("Mul", {"a", "b"}, {"ab"})
      .node("Add", {"x", "ab"}, {"y"})
      .output("y");
  Graph graph = build(m);
  auto stats = optimize(graph);
  REQUIRE(stats);
  CHECK(stats->folded == 1);
  REQUIRE(graph.nodes.size() == 1);
  CHECK(graph.tensors[graph.nodes[0].inputs[1]].value.f == std::vector<float>{4, 10, 18});
}

TEST_CASE("passes : optimize exige des formes inférées", "[gpu_ir][passes]") {
  ModelBuilder m;
  m.input("x", {1, -1}).node("Relu", {"x"}, {"y"}).output("y");
  auto graph = load_onnx(m.bytes());
  REQUIRE(graph);
  CHECK(optimize(*graph).error().code == RefusalCode::ShapeInvalid);
}
