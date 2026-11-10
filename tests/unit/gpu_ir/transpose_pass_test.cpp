// Réécritures de Transpose : ConvTranspose en Conv puis DepthToSpace (transposes_as_convs),
// Transpose de grille replié dans GridSample ; mêmes sorties de l'interpréteur.
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/passes.h"
#include "test_util.h"

using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

// x [1, 8, 5, 6] → ConvTranspose (8 → cout, noyau k, pas s, marges pad) → DepthToSpace facultatif.
ModelBuilder transpose_model(std::int64_t cout, std::int64_t k, std::int64_t s, std::int64_t pad,
                             const std::string& shuffle_mode) {
  ModelBuilder m(17);
  m.input("x", {1, 8, 5, 6})
      .weight("w", {8, cout, k, k}, sequence(static_cast<std::size_t>(8 * cout * k * k), -0.5F, 0.5F, 41))
      .weight("b", {cout}, sequence(static_cast<std::size_t>(cout), -0.2F, 0.2F, 42));
  m.node("ConvTranspose", {"x", "w", "b"}, {shuffle_mode.empty() ? "y" : "t"},
         {attr_ints("strides", {s, s}), attr_ints("pads", {pad, pad, pad, pad})});
  if (!shuffle_mode.empty()) {
    m.node("DepthToSpace", {"t"}, {"y"}, {attr_int("blocksize", 2), attr_string("mode", shuffle_mode)});
  }
  m.output("y");
  return m;
}

PassStats check(const ModelBuilder& model, std::int64_t max_channels, Graph& after, bool phases = false) {
  const Graph before = build(model);
  after = before;
  auto stats = optimize(after, {.transpose_conv_channels = max_channels, .transpose_phases = phases});
  if (!stats) FAIL("refus " << ajp::name(stats.error().code) << " : " << stats.error().detail);
  const std::vector<NamedValue> inputs{{.name = "x", .value = random_floats({1, 8, 5, 6}, 43)}};
  const std::vector<NamedValue> expected = execute(before, inputs);
  const std::vector<NamedValue> got = execute(after, inputs);
  REQUIRE(got.size() == 1);
  REQUIRE(got[0].value.dims == expected[0].value.dims);
  CHECK(max_abs_diff(got[0].value.f, expected[0].value.f) < 1e-5F);
  return *stats;
}

}  // namespace

TEST_CASE("ConvTranspose 4 × 4 de pas 2 : Conv 3 × 3 aux phases, DepthToSpace fondu", "[gpu_ir][passes]") {
  Graph after;
  const PassStats stats = check(transpose_model(3, 4, 2, 1, ""), 256, after);
  CHECK(stats.transposes == 1);
  CHECK(stats.spaces == 1);
  CHECK(count_ops(after, OpKind::ConvTranspose) == 0);
  CHECK(count_ops(after, OpKind::Conv) == 1);
  CHECK(count_ops(after, OpKind::DepthToSpace) == 0);
  const auto& conv = std::get<ConvAttrs>(after.nodes[0].attrs);
  CHECK(conv.kernel == Pair{3, 3});
  CHECK(conv.epilogue.space == 2);
  const Dims& out = after.tensors[after.nodes[0].outputs[0]].dims;
  CHECK(out == Dims{1, 3, 10, 12});
  CHECK(conv_result_dims(out, conv.epilogue) == Dims{1, 12, 5, 6});
}

TEST_CASE("ConvTranspose suivie d'un pixel_shuffle : un seul réarrangement de bloc 4, fondu",
          "[gpu_ir][passes]") {
  for (const std::string mode : {"CRD", "DCR"}) {
    Graph after;
    const PassStats stats = check(transpose_model(8, 4, 2, 1, mode), 256, after);
    CHECK(stats.transposes == 1);
    CHECK(count_ops(after, OpKind::DepthToSpace) == 0);
    REQUIRE(count_ops(after, OpKind::Conv) == 1);
    for (const Node& node : after.nodes) {
      if (node.op == OpKind::Conv) CHECK(std::get<ConvAttrs>(node.attrs).epilogue.space == 4);
    }
  }
}

TEST_CASE("ConvTranspose 2 × 2 de pas 2 : Conv 1 × 1", "[gpu_ir][passes]") {
  Graph after;
  const PassStats stats = check(transpose_model(4, 2, 2, 0, ""), 256, after);
  CHECK(stats.transposes == 1);
  CHECK(std::get<ConvAttrs>(after.nodes[0].attrs).kernel == Pair{1, 1});
}

// V3 : une Conv 2 × 2 par phase au lieu de la Conv 3 × 3 aux prises nulles (4 sur 9 utiles).
TEST_CASE("ConvTranspose 4 × 4 de pas 2 : une Conv sans zéros par phase", "[gpu_ir][passes]") {
  for (const std::string mode : {"", "DCR", "CRD"}) {
    INFO("pixel_shuffle " << mode);
    Graph after;
    const PassStats stats = check(transpose_model(8, 4, 2, 1, mode), 256, after, true);
    CHECK(stats.transposes == 1);
    // A44-12 : le pixel_shuffle qui suit se fond, chaque phase en deux Conv (une par ligne du bloc), et
    // il ne reste qu'un réarrangement, de bloc 4.
    CHECK(count_ops(after, OpKind::Conv) == (mode.empty() ? 4U : 8U));
    CHECK(count_ops(after, OpKind::Concat) == 1);
    CHECK(count_ops(after, OpKind::DepthToSpace) == 1);
    for (const Node& node : after.nodes) {
      if (node.op == OpKind::Conv) CHECK(std::get<ConvAttrs>(node.attrs).kernel == Pair{2, 2});
      if (node.op == OpKind::DepthToSpace)
        CHECK(std::get<BlockAttrs>(node.attrs).blocksize == (mode.empty() ? 2 : 4));
    }
  }
  // Noyau 2 × 2 de pas 2 : une prise par phase, autant que la Conv 1 × 1 unique ; elle est gardée.
  Graph after;
  CHECK(check(transpose_model(8, 2, 2, 0, ""), 256, after, true).transposes == 1);
  CHECK(count_ops(after, OpKind::Conv) == 1);
}

// A44-12 : fondre les deux réarrangements ne change aucune somme. Le même modèle dont la sortie de la
// ConvTranspose est aussi une sortie du graphe garde les deux DepthToSpace : les deux graphes rendent y au
// bit près.
TEST_CASE("ConvTranspose par phases : le pixel_shuffle fondu rend les mêmes bits que les deux réarrangements",
          "[gpu_ir][passes]") {
  for (const std::string mode : {"DCR", "CRD"}) {
    INFO("pixel_shuffle " << mode);
    ModelBuilder kept = transpose_model(8, 4, 2, 1, mode);
    kept.output("t");
    Graph merged = build(transpose_model(8, 4, 2, 1, mode));
    Graph apart = build(kept);
    REQUIRE(transposes_as_convs(merged, 256, true) == 1);
    REQUIRE(transposes_as_convs(apart, 256, true) == 1);
    CHECK(count_ops(merged, OpKind::DepthToSpace) == 1);
    CHECK(count_ops(apart, OpKind::DepthToSpace) == 2);
    const std::vector<NamedValue> inputs{{.name = "x", .value = random_floats({1, 8, 5, 6}, 43)}};
    const std::vector<NamedValue> one = execute(merged, inputs);
    const std::vector<NamedValue> two = execute(apart, inputs);
    REQUIRE(one.size() == 1);
    REQUIRE(two.size() == 2);
    const auto y = std::ranges::find(two, "y", &NamedValue::name);
    REQUIRE(y != two.end());
    REQUIRE(one[0].value.dims == y->value.dims);
    CHECK(one[0].value.f == y->value.f);
  }
}

TEST_CASE("ConvTranspose gardée : trop de canaux, ou passe éteinte", "[gpu_ir][passes]") {
  Graph after;
  CHECK(check(transpose_model(8, 4, 2, 1, ""), 16, after).transposes == 0);
  CHECK(count_ops(after, OpKind::ConvTranspose) == 1);
  CHECK(check(transpose_model(8, 4, 2, 1, ""), 0, after).transposes == 0);
}

TEST_CASE("GridSample : le Transpose de sa grille replié, mêmes sorties", "[gpu_ir][passes]") {
  ModelBuilder m(17);
  m.input("x", {1, 3, 4, 5}).input("flow", {1, 2, 4, 5});
  m.node("Transpose", {"flow"}, {"grid"}, {attr_ints("perm", {0, 2, 3, 1})})
      .node("GridSample", {"x", "grid"}, {"y"},
            {attr_int("align_corners", 1), attr_string("padding_mode", "border")})
      .output("y");
  const Graph before = build(m);
  Graph after = before;
  auto stats = optimize(after);
  REQUIRE(stats);
  CHECK(stats->grid_transposes == 1);
  CHECK(count_ops(after, OpKind::Transpose) == 0);
  const std::vector<NamedValue> inputs{
      {.name = "x", .value = random_floats({1, 3, 4, 5}, 44)},
      {.name = "flow", .value = random_floats({1, 2, 4, 5}, 45, -1.2F, 1.2F)}};
  const std::vector<NamedValue> expected = execute(before, inputs);
  const std::vector<NamedValue> got = execute(after, inputs);
  REQUIRE(got.size() == 1);
  CHECK(max_abs_diff(got[0].value.f, expected[0].value.f) < 1e-6F);
}
