// promote_grid_chains (passes.h) : la grille d'un GridSample quitte le float16. Le graphe FP16 de DistilDRBA
// livré jusqu'en novembre 2026 y calcule base + flot × pas : à 1024 de large, le warp se pose à plus d'un
// dixième de pixel du flot (A03-18, A44-01).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/interpreter.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "kernels/ref/half.h"
#include "test_util.h"

using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

constexpr int kWidth = 1024;
constexpr int kHeight = 2;
constexpr float kFlowX = 3.25F;  // exact en float16

// Le warp de l'export de DistilDRBA tel que convert_rife_fp16.py le laissait : base (float, venue des
// formes) arrondie par un Cast, flot float16 × pas arrondi, somme, Concat, Transpose, Cast float16 (le
// grid.to(image.dtype) de l'export), puis le Cast float devant GridSample. `grid_f32` : la même chaîne
// telle que le convertisseur corrigé la laisse, sans aucun float16 entre la base et GridSample.
ModelBuilder drba_warp(bool grid_f32) {
  std::vector<float> base_x(kWidth);
  for (int x = 0; x < kWidth; ++x)
    base_x[static_cast<std::size_t>(x)] = 2.0F * static_cast<float>(x) / (kWidth - 1) - 1.0F;
  const std::int64_t f16 = onnx::kFloat16;
  const std::int64_t chain = grid_f32 ? onnx::kFloat : onnx::kFloat16;
  ModelBuilder m(17);
  m.input("image", {1, 1, kHeight, kWidth})
      .input("flow", {1, 2, kHeight, kWidth}, onnx::kFloat16)
      .weight("base_x", {1, 1, 1, kWidth}, base_x)
      .weight("base_y", {1, 1, kHeight, 1}, {-1.0F, 1.0F})
      .weight("step_x", {}, {2.0F / (kWidth - 1)})
      .weight("step_y", {}, {2.0F / (kHeight - 1)})
      .ints("split", {1, 1});
  m.node("Split", {"flow", "split"}, {"fx", "fy"}, {attr_int("axis", 1)})
      .node("Cast", {"base_x"}, {"bx"}, {attr_int("to", chain)})
      .node("Cast", {"base_y"}, {"by"}, {attr_int("to", chain)})
      .node("Cast", {"step_x"}, {"sx"}, {attr_int("to", chain)})
      .node("Cast", {"step_y"}, {"sy"}, {attr_int("to", chain)});
  if (grid_f32) {
    m.node("Cast", {"fx"}, {"fx32"}, {attr_int("to", onnx::kFloat)})
        .node("Cast", {"fy"}, {"fy32"}, {attr_int("to", onnx::kFloat)});
  }
  m.node("Mul", {grid_f32 ? "fx32" : "fx", "sx"}, {"mx"})
      .node("Mul", {grid_f32 ? "fy32" : "fy", "sy"}, {"my"})
      .node("Add", {"bx", "mx"}, {"gx"})
      .node("Add", {"by", "my"}, {"gy"})
      .node("Concat", {"gx", "gy"}, {"g"}, {attr_int("axis", 1)})
      .node("Transpose", {"g"}, {"gt"}, {attr_ints("perm", {0, 2, 3, 1})})
      .node("Cast", {"gt"}, {"gc"}, {attr_int("to", grid_f32 ? onnx::kFloat : f16)})
      .node("Cast", {"gc"}, {"grid"}, {attr_int("to", onnx::kFloat)})
      .node("GridSample", {"image", "grid"}, {"out"},
            {attr_int("align_corners", 1), attr_string("padding_mode", "border")})
      .output("out");
  return m;
}

// Une rampe : la sortie du warp, × (largeur − 1), est la position échantillonnée. Rend l'écart au flot, en
// pixels, au pire des colonnes.
float worst_position_error(const Graph& graph) {
  std::vector<float> ramp(static_cast<std::size_t>(kHeight) * kWidth);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x)
      ramp[static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x)] =
          static_cast<float>(x) / (kWidth - 1);
  }
  std::vector<float> flow(static_cast<std::size_t>(2) * kHeight * kWidth, 0.0F);
  std::fill_n(flow.begin(), static_cast<std::size_t>(kHeight) * kWidth, kFlowX);
  const Value out =
      execute1(graph, {{.name = "image", .value = floats({1, 1, kHeight, kWidth}, ramp)},
                       {.name = "flow", .value = floats({1, 2, kHeight, kWidth}, flow, DataType::F16)}});
  float worst = 0.0F;
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const float want = std::min(static_cast<float>(x) + kFlowX, static_cast<float>(kWidth - 1));
      const float got =
          out.f[static_cast<std::size_t>(y) * kWidth + static_cast<std::size_t>(x)] * (kWidth - 1);
      worst = std::max(worst, std::abs(got - want));
    }
  }
  return worst;
}

DataType type_of(const Graph& graph, std::string_view name) {
  const auto id = find_tensor(graph, name);
  REQUIRE(id.has_value());
  return graph.tensors[*id].type;
}

}  // namespace

TEST_CASE("passes : la grille float16 d'un GridSample passe en float", "[gpu_ir][passes][grid]") {
  Graph graph = build(drba_warp(false));
  // Telle quelle, la somme float16 pose le warp à plus d'un dixième de pixel du flot.
  CHECK(worst_position_error(graph) > 0.1F);
  CHECK(type_of(graph, "gx") == DataType::F16);

  CHECK(promote_grid_chains(graph) == 11);
  for (const char* name : {"bx", "by", "sx", "sy", "mx", "my", "gx", "gy", "g", "gt", "gc"})
    CHECK(type_of(graph, name) == DataType::F32);
  CHECK(type_of(graph, "fx") == DataType::F16);  // le flot reste une donnée float16
  CHECK(worst_position_error(graph) < 2e-3F);
  CHECK(promote_grid_chains(graph) == 0);  // rien à refaire

  // Avant optimize : sinon le pliage arrondit la base à travers son Cast, et la passe arrive trop tard.
  auto stats = optimize(graph);
  REQUIRE(stats.has_value());
  CHECK(worst_position_error(graph) < 2e-3F);
  // Les Cast de la chaîne, devenus de float vers float, sont retirés ; restent ceux du flot, exacts.
  CHECK(count_ops(graph, OpKind::Cast) == 2);
}

TEST_CASE("passes : une grille déjà en float n'est pas touchée", "[gpu_ir][passes][grid]") {
  Graph graph = build(drba_warp(true));
  const float before = worst_position_error(graph);
  CHECK(before < 2e-3F);
  CHECK(promote_grid_chains(graph) == 0);
  CHECK(worst_position_error(graph) == before);
}

TEST_CASE("passes : la grille n'entraîne ni le flot ni l'image en float", "[gpu_ir][passes][grid]") {
  // Le flot vient d'un Mul float16 (flot × échelle) : il n'est pas le produit flot × pas de la somme, la
  // passe s'arrête à lui. L'image float16 de GridSample n'est pas sa grille.
  std::vector<float> base(static_cast<std::size_t>(4) * 4 * 2, 0.0F);
  ModelBuilder m(17);
  m.input("image", {1, 1, 4, 4}, onnx::kFloat16)
      .input("flow", {1, 4, 4, 2}, onnx::kFloat16)
      .initializer(tensor_f16("base", {1, 4, 4, 2}, base))
      .initializer(tensor_f16("two", {}, std::vector<float>{2.0F}))
      .initializer(tensor_f16("step", {}, std::vector<float>{0.5F}))
      .node("Mul", {"flow", "two"}, {"scaled"})
      .node("Mul", {"scaled", "step"}, {"moved"})
      .node("Add", {"base", "moved"}, {"grid16"})
      .node("Cast", {"grid16"}, {"grid"}, {attr_int("to", onnx::kFloat)})
      .node("Cast", {"image"}, {"image32"}, {attr_int("to", onnx::kFloat)})
      .node("GridSample", {"image32", "grid"}, {"out"}, {attr_int("align_corners", 1)})
      .output("out");
  Graph graph = build(m);
  CHECK(promote_grid_chains(graph) == 2);
  CHECK(type_of(graph, "grid16") == DataType::F32);
  CHECK(type_of(graph, "moved") == DataType::F32);
  CHECK(type_of(graph, "scaled") == DataType::F16);
  CHECK(type_of(graph, "image") == DataType::F16);
}

// C44-02 : la grille passée en float ne rend pas le graphe inexécutable (ONNX : image et grille ont chacune
// leur type). Le graphe promu passe l'inférence de formes et l'interpréteur.
TEST_CASE("passes : promote_grid_chains laisse un graphe exécutable", "[gpu_ir][passes][grid]") {
  ModelBuilder m;
  m.input("x", {1, 3, 6, 8}, onnx::kFloat16).input("flow", {1, 2, 6, 8}, onnx::kFloat16);
  m.weight("base32", {1, 2, 6, 8}, sequence(96, -1.0F, 1.0F, 1))
      .initializer(tensor_f16("step", {1, 2, 1, 1}, std::vector<float>{0.25F, 0.3333F}))
      .node("Cast", {"base32"}, {"base16"}, {attr_int("to", onnx::kFloat16)})
      .node("Mul", {"flow", "step"}, {"term"})
      .node("Add", {"base16", "term"}, {"sum"})
      .node("Transpose", {"sum"}, {"grid"}, {attr_ints("perm", {0, 2, 3, 1})})
      .node("GridSample", {"x", "grid"}, {"y"}, {attr_string("padding_mode", "border")})
      .output("y", onnx::kFloat16);
  Graph graph = build(m);
  REQUIRE(promote_grid_chains(graph) > 0);
  REQUIRE(optimize(graph));
  REQUIRE(infer_shapes(graph));  // un nouvel appel ne refuse plus la grille float sous une image float16
  Value x = random_floats({1, 3, 6, 8}, 1);
  Value f = random_floats({1, 2, 6, 8}, 2);
  x.type = f.type = DataType::F16;
  for (float& v : x.f) v = ajp::kernels::ref::round_to_half(v);
  for (float& v : f.f) v = ajp::kernels::ref::round_to_half(v);
  CHECK(run(graph, std::vector<NamedValue>{{.name = "x", .value = x}, {.name = "flow", .value = f}}));
}
