// Opérations élément par élément et opérations de forme : valeurs calculées à la main.
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/ref/half.h"
#include "test_util.h"

using ajp::Refusal;
using ajp::RefusalCode;
using namespace ajp::gpu::ir;
using ajp::kernels::ref::round_to_half;
using namespace ajp::gpu::ir::test;

namespace {

constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();

Value ints(Dims dims, std::vector<std::int64_t> values) {
  return Value{.type = DataType::I64, .dims = std::move(dims), .i = std::move(values)};
}

// Graphe d'un seul nœud : x en entrée, les autres opérandes en constantes déjà posées dans m.
Value run_node(ModelBuilder& m, const Value& x, std::string_view op,
               std::initializer_list<std::string_view> inputs,
               std::initializer_list<ProtoWriter> attrs = {}) {
  m.input("x", x.dims, x.type == DataType::I64 ? onnx::kInt64 : onnx::kFloat)
      .node(op, inputs, {"y"}, attrs)
      .output("y");
  return execute1(build(m), {{.name = "x", .value = x}});
}

std::expected<std::vector<NamedValue>, ajp::Refusal> try_run(ModelBuilder& m, const Value& x,
                                                             std::string_view op,
                                                             std::initializer_list<std::string_view> inputs,
                                                             std::initializer_list<ProtoWriter> attrs = {}) {
  m.input("x", x.dims, x.type == DataType::I64 ? onnx::kInt64 : onnx::kFloat)
      .node(op, inputs, {"y"}, attrs)
      .output("y");
  const Graph graph = build(m);
  return run(graph, std::vector<NamedValue>{{.name = "x", .value = x}});
}

}  // namespace

TEST_CASE("Add : diffusion de rangs différents", "[gpu_ir][ops][elementwise]") {
  const Value a = floats({2, 3, 1}, {1, 2, 3, 4, 5, 6});
  ModelBuilder m;
  m.weight("b", {1, 4}, {10, 20, 30, 40});
  const Value y = run_node(m, a, "Add", {"x", "b"});
  REQUIRE(y.dims == Dims{2, 3, 4});
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t k = 0; k < 4; ++k)
      CHECK(y.f[(i * 4) + k] == a.f[i] + (10.0F * static_cast<float>(k + 1)));
  }
  ModelBuilder bad;
  bad.weight("b", {3}, {1, 2, 3});
  bad.input("x", {2, 2}).node("Add", {"x", "b"}, {"y"}).output("y");
  CHECK(refusal(bad).code == RefusalCode::ShapeInvalid);
}

TEST_CASE("Sub, Mul, Div par un scalaire", "[gpu_ir][ops][elementwise]") {
  const Value x = floats({4}, {1, -2, 3, 8});
  for (const auto& [op, expected] : std::vector<std::pair<std::string_view, std::vector<float>>>{
           {"Sub", {-1, -4, 1, 6}}, {"Mul", {2, -4, 6, 16}}, {"Div", {0.5F, -1, 1.5F, 4}}}) {
    ModelBuilder m;
    m.weight("two", {}, {2});
    CHECK(run_node(m, x, op, {"x", "two"}).f == expected);
  }
}

TEST_CASE("Activations : Relu, LeakyRelu, PRelu, Sigmoid, Tanh, Clip", "[gpu_ir][ops][elementwise]") {
  const Value x = floats({1, 3, 1, 2}, {-2, 1, -0.5F, 0, 3, -4});
  ModelBuilder relu;
  CHECK(run_node(relu, x, "Relu", {"x"}).f == std::vector<float>{0, 1, 0, 0, 3, 0});
  ModelBuilder leaky;
  CHECK(run_node(leaky, x, "LeakyRelu", {"x"}, {attr_float("alpha", 0.25F)}).f ==
        std::vector<float>{-0.5F, 1, -0.125F, 0, 3, -1});
  ModelBuilder leaky_default;
  CHECK(run_node(leaky_default, x, "LeakyRelu", {"x"}).f[0] == -2 * 0.01F);
  ModelBuilder prelu;
  prelu.weight("slope", {3, 1, 1}, {0.5F, 2, -1});
  CHECK(run_node(prelu, x, "PRelu", {"x", "slope"}).f == std::vector<float>{-1, 1, -1, 0, 3, 4});
  ModelBuilder sigmoid;
  const Value s = run_node(sigmoid, x, "Sigmoid", {"x"});
  ModelBuilder tanh_model;
  const Value t = run_node(tanh_model, x, "Tanh", {"x"});
  for (std::size_t k = 0; k < x.f.size(); ++k) {
    CHECK(std::abs(s.f[k] - (1.0F / (1.0F + std::exp(-x.f[k])))) < 1e-7F);
    CHECK(std::abs(t.f[k] - std::tanh(x.f[k])) < 1e-7F);
  }
  ModelBuilder clip;
  clip.weight("lo", {}, {-1}).weight("hi", {}, {2});
  CHECK(run_node(clip, x, "Clip", {"x", "lo", "hi"}).f == std::vector<float>{-1, 1, -0.5F, 0, 2, -1});
  ModelBuilder clip_hi;
  clip_hi.weight("hi", {}, {0.5F});
  CHECK(run_node(clip_hi, x, "Clip", {"x", "", "hi"}).f == std::vector<float>{-2, 0.5F, -0.5F, 0, 0.5F, -4});
}

TEST_CASE("Cast : F16 arrondit, entiers tronqués, non-fini refusé", "[gpu_ir][ops][elementwise]") {
  const Value x = floats({3}, {1.0F + std::ldexp(1.0F, -12), 2.7F, -2.7F});
  ModelBuilder half;
  const Value h = run_node(half, x, "Cast", {"x"}, {attr_int("to", onnx::kFloat16)});
  CHECK(h.type == DataType::F16);
  CHECK(h.f == std::vector<float>{1.0F, round_to_half(2.7F), round_to_half(-2.7F)});
  ModelBuilder to_int;
  const Value i = run_node(to_int, x, "Cast", {"x"}, {attr_int("to", onnx::kInt64)});
  CHECK(i.i == std::vector<std::int64_t>{1, 2, -2});
  ModelBuilder nan;
  auto refused = try_run(nan, floats({1}, {std::numeric_limits<float>::infinity()}), "Cast", {"x"},
                         {attr_int("to", onnx::kInt64)});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::GraphInvalid);
  ModelBuilder dbl;
  dbl.input("x", {1}).node("Cast", {"x"}, {"y"}, {attr_int("to", 11)}).output("y");
  CHECK(refusal(dbl).code == RefusalCode::OnnxUnsupported);
}

TEST_CASE("Entiers : Div et Mod refusent la division par zéro et le débordement",
          "[gpu_ir][ops][elementwise]") {
  ModelBuilder div;
  div.ints("d", {2, 0});
  auto by_zero = try_run(div, ints({2}, {7, 7}), "Div", {"x", "d"});
  REQUIRE_FALSE(by_zero);
  CHECK(by_zero.error().code == RefusalCode::GraphInvalid);
  ModelBuilder overflow;
  overflow.ints("d", {-1});
  CHECK_FALSE(try_run(overflow, ints({1}, {kMin}), "Div", {"x", "d"}));

  ModelBuilder floor_mod;
  floor_mod.ints("d", {3, -3, 3});
  CHECK(run_node(floor_mod, ints({3}, {-7, 7, 7}), "Mod", {"x", "d"}).i ==
        std::vector<std::int64_t>{2, -2, 1});
  ModelBuilder c_mod;
  c_mod.ints("d", {3, -3});
  CHECK(run_node(c_mod, ints({2}, {-7, 7}), "Mod", {"x", "d"}, {attr_int("fmod", 1)}).i ==
        std::vector<std::int64_t>{-1, 1});
}

TEST_CASE("Entiers : Add, Sub et Mul refusent le débordement au lieu de le replier",
          "[gpu_ir][ops][elementwise]") {
  struct Case {
    std::string_view op;
    std::int64_t a;
    std::int64_t b;
    bool overflows;
  };
  const std::vector<Case> cases{
      {"Add", kMax, 1, true},
      {"Add", kMin, -1, true},
      {"Add", kMax, -1, false},
      {"Sub", kMin, 1, true},
      {"Sub", kMax, -1, true},
      {"Sub", -1, kMax, false},
      {"Mul", kMin, -1, true},
      {"Mul", -1, kMin, true},
      {"Mul", std::int64_t{1} << 32, std::int64_t{1} << 30, false},
      {"Mul", std::int64_t{1} << 32, std::int64_t{1} << 31, true},  // 2^63
      {"Mul", 3037000500, 3037000500, true},                        // juste au-delà de la racine de 2^63
      {"Mul", -3037000499, 3037000499, false},
  };
  for (const Case& c : cases) {
    INFO(c.op << " " << c.a << " " << c.b);
    ModelBuilder m;
    m.ints("d", {c.b});
    auto out = try_run(m, ints({1}, {c.a}), c.op, {"x", "d"});
    CHECK(out.has_value() != c.overflows);
    if (!c.overflows && out) {
      const std::int64_t expected = c.op == "Add" ? c.a + c.b : c.op == "Sub" ? c.a - c.b : c.a * c.b;
      CHECK((*out)[0].value.i == std::vector<std::int64_t>{expected});
    }
  }
}

TEST_CASE("Concat, Split et anciens attributs split", "[gpu_ir][ops][shape]") {
  const Value x = random_floats({1, 2, 2, 3}, 1);
  ModelBuilder concat;
  concat.weight("c", {1, 1, 2, 3}, {9, 9, 9, 9, 9, 9});
  const Value joined = run_node(concat, x, "Concat", {"x", "c", "x"}, {attr_int("axis", -3)});
  REQUIRE(joined.dims == Dims{1, 5, 2, 3});
  CHECK(std::vector<float>(joined.f.begin(), joined.f.begin() + 12) == x.f);
  CHECK(joined.f[12] == 9);

  ModelBuilder split(13);
  split.ints("sizes", {1, 2})
      .input("x", x.dims)
      .node("Split", {"x", "sizes"}, {"a", "b"}, {attr_int("axis", 3)});
  split.output("a").output("b");
  std::vector<NamedValue> parts = execute(build(split), {{.name = "x", .value = x}});
  REQUIRE(parts.size() == 2);
  CHECK(parts[0].value.dims == Dims{1, 2, 2, 1});
  CHECK(parts[1].value.dims == Dims{1, 2, 2, 2});
  CHECK(parts[0].value.f[1] == x.f[3]);
  CHECK(parts[1].value.f[0] == x.f[1]);

  // Opset 11 : la répartition est un attribut, et l'import la passe en entrée.
  ModelBuilder old(11);
  old.input("x", x.dims).node("Split", {"x"}, {"a", "b"}, {attr_int("axis", 3), attr_ints("split", {2, 1})});
  old.output("a").output("b");
  parts = execute(build(old), {{.name = "x", .value = x}});
  CHECK(parts[0].value.dims == Dims{1, 2, 2, 2});

  // Opset 18 : parts égales, la dernière plus petite.
  ModelBuilder equal(18);
  equal.input("x", {1, 7})
      .node("Split", {"x"}, {"a", "b"}, {attr_int("axis", 1), attr_int("num_outputs", 2)});
  equal.output("a").output("b");
  parts = execute(build(equal), {{.name = "x", .value = random_floats({1, 7})}});
  CHECK(parts[0].value.dims == Dims{1, 4});
  CHECK(parts[1].value.dims == Dims{1, 3});
}

TEST_CASE("Slice : pas négatif, bornes extrêmes, pas de 2", "[gpu_ir][ops][shape]") {
  const Value x = floats({1, 1, 2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  ModelBuilder reverse;
  reverse.ints("s", {-1}).ints("e", {kMin}).ints("a", {3}).ints("st", {-1});
  CHECK(run_node(reverse, x, "Slice", {"x", "s", "e", "a", "st"}).f ==
        std::vector<float>{4, 3, 2, 1, 0, 9, 8, 7, 6, 5});
  ModelBuilder stride;
  stride.ints("s", {1, 0}).ints("e", {kMax, kMax}).ints("a", {-2, -1}).ints("st", {1, 2});
  const Value strided = run_node(stride, x, "Slice", {"x", "s", "e", "a", "st"});
  CHECK(strided.dims == Dims{1, 1, 1, 3});
  CHECK(strided.f == std::vector<float>{5, 7, 9});
  ModelBuilder empty;
  empty.ints("s", {3}).ints("e", {1}).ints("a", {3});
  CHECK(run_node(empty, x, "Slice", {"x", "s", "e", "a"}).dims == Dims{1, 1, 2, 0});

  ModelBuilder zero_step;
  zero_step.ints("s", {0}).ints("e", {2}).ints("a", {3}).ints("st", {0});
  zero_step.input("x", x.dims).node("Slice", {"x", "s", "e", "a", "st"}, {"y"}).output("y");
  CHECK(refusal(zero_step).code == RefusalCode::GraphInvalid);
  ModelBuilder twice;
  twice.ints("s", {0, 0}).ints("e", {1, 1}).ints("a", {3, -1});
  twice.input("x", x.dims).node("Slice", {"x", "s", "e", "a"}, {"y"}).output("y");
  CHECK(refusal(twice).code == RefusalCode::GraphInvalid);
}

TEST_CASE("Pad : constante, reflect, edge et marges négatives", "[gpu_ir][ops][shape]") {
  const Value x = floats({1, 1, 1, 3}, {1, 2, 3});
  ModelBuilder constant;
  constant.ints("p", {0, 0, 0, 2, 0, 0, 0, 1}).weight("v", {}, {-1});
  CHECK(run_node(constant, x, "Pad", {"x", "p", "v"}).f == std::vector<float>{-1, -1, 1, 2, 3, -1});
  ModelBuilder reflect;
  reflect.ints("p", {0, 0, 0, 2, 0, 0, 0, 2});
  CHECK(run_node(reflect, x, "Pad", {"x", "p"}, {attr_string("mode", "reflect")}).f ==
        std::vector<float>{3, 2, 1, 2, 3, 2, 1});
  ModelBuilder edge;
  edge.ints("p", {0, 0, 1, 1, 0, 0, 0, 0});
  const Value e = run_node(edge, x, "Pad", {"x", "p"}, {attr_string("mode", "edge")});
  CHECK(e.dims == Dims{1, 1, 2, 4});
  CHECK(e.f == std::vector<float>{1, 1, 2, 3, 1, 1, 2, 3});
  ModelBuilder crop;
  crop.ints("p", {0, 0, 0, -1, 0, 0, 0, 1});
  CHECK(run_node(crop, x, "Pad", {"x", "p"}).f == std::vector<float>{2, 3, 0});

  ModelBuilder too_far;
  too_far.ints("p", {0, 0, 0, 3, 0, 0, 0, 0});
  too_far.input("x", x.dims).node("Pad", {"x", "p"}, {"y"}, {attr_string("mode", "reflect")}).output("y");
  CHECK(refusal(too_far).code == RefusalCode::ShapeInvalid);
  ModelBuilder wrap(19);
  wrap.ints("p", {0, 0, 0, 1, 0, 0, 0, 1});
  wrap.input("x", x.dims).node("Pad", {"x", "p"}, {"y"}, {attr_string("mode", "wrap")}).output("y");
  CHECK(refusal(wrap).code == RefusalCode::OnnxUnsupported);
}

TEST_CASE("Reshape, Transpose, Squeeze, Unsqueeze, Gather, Shape", "[gpu_ir][ops][shape]") {
  const Value x = random_floats({2, 3, 4}, 2);
  ModelBuilder reshape;
  reshape.ints("shape", {0, -1, 2});
  CHECK(run_node(reshape, x, "Reshape", {"x", "shape"}).dims == Dims{2, 6, 2});
  ModelBuilder two_free;
  two_free.ints("shape", {-1, -1});
  two_free.input("x", x.dims).node("Reshape", {"x", "shape"}, {"y"}).output("y");
  CHECK(refusal(two_free).code == RefusalCode::ShapeInvalid);

  ModelBuilder transpose;
  const Value t = run_node(transpose, x, "Transpose", {"x"}, {attr_ints("perm", {2, 0, 1})});
  CHECK(t.dims == Dims{4, 2, 3});
  CHECK(t.f[1] == x.f[4]);  // t[0, 0, 1] = x[0, 1, 0]
  ModelBuilder reversed;
  CHECK(run_node(reversed, x, "Transpose", {"x"}).dims == Dims{4, 3, 2});

  ModelBuilder unsqueeze;
  unsqueeze.ints("axes", {0, -1});
  CHECK(run_node(unsqueeze, x, "Unsqueeze", {"x", "axes"}).dims == Dims{1, 2, 3, 4, 1});
  ModelBuilder squeeze(11);
  squeeze.input("x", {1, 3, 1}).node("Squeeze", {"x"}, {"y"}, {attr_ints("axes", {0})}).output("y");
  CHECK(execute1(build(squeeze), {{.name = "x", .value = random_floats({1, 3, 1})}}).dims == Dims{3, 1});

  ModelBuilder gather;
  gather.ints("idx", {-1, 0});
  const Value g = run_node(gather, x, "Gather", {"x", "idx"}, {attr_int("axis", 1)});
  CHECK(g.dims == Dims{2, 2, 4});
  CHECK(g.f[0] == x.f[8]);
  ModelBuilder out_of_range;
  out_of_range.ints("idx", {3});
  out_of_range.input("x", x.dims).node("Gather", {"x", "idx"}, {"y"}, {attr_int("axis", 1)}).output("y");
  CHECK(refusal(out_of_range).code == RefusalCode::GraphInvalid);

  ModelBuilder shape(15);
  const Value s = run_node(shape, x, "Shape", {"x"}, {attr_int("start", -2)});
  CHECK(s.type == DataType::I64);
  CHECK(s.i == std::vector<std::int64_t>{3, 4});
}

TEST_CASE("ReduceMean : axes, keepdims et noop_with_empty_axes", "[gpu_ir][ops][shape]") {
  const Value x = floats({1, 2, 2, 2}, {1, 2, 3, 4, 10, 20, 30, 40});
  ModelBuilder spatial(18);
  spatial.ints("axes", {2, 3});
  const Value m = run_node(spatial, x, "ReduceMean", {"x", "axes"});
  CHECK(m.dims == Dims{1, 2, 1, 1});
  CHECK(m.f == std::vector<float>{2.5F, 25.0F});
  ModelBuilder flat(13);
  const Value all = run_node(flat, x, "ReduceMean", {"x"}, {attr_int("keepdims", 0)});
  CHECK(all.dims.empty());
  CHECK(all.f == std::vector<float>{13.75F});
  ModelBuilder old(13);
  CHECK(run_node(old, x, "ReduceMean", {"x"}, {attr_ints("axes", {1})}).f ==
        std::vector<float>{5.5F, 11, 16.5F, 22});
  ModelBuilder noop(18);
  CHECK(run_node(noop, x, "ReduceMean", {"x"}, {attr_int("noop_with_empty_axes", 1)}).f == x.f);
}
