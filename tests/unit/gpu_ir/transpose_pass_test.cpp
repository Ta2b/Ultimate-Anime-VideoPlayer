// Réécritures de Transpose : ConvTranspose en Conv puis DepthToSpace (transposes_as_convs),
// Transpose de grille replié dans GridSample ; mêmes sorties de l'interpréteur.
#include <cstdint>
#include <string>
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

PassStats check(const ModelBuilder& model, std::int64_t max_channels, Graph& after) {
  const Graph before = build(model);
  after = before;
  auto stats = optimize(after, {.transpose_conv_channels = max_channels});
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

TEST_CASE("ConvTranspose suivie d'un pixel_shuffle : un seul réarrangement de bloc 4, fondu", "[gpu_ir][passes]") {
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
