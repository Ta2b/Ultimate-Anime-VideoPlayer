// Resize qui réduit remonté devant Concat, Reshape et Split (net_hoist.h) : le graphe réécrit rend
// les mêmes valeurs que l'original à l'interpréteur, bit à bit, et l'appareil rend les mêmes
// valeurs que l'interpréteur sur le graphe réécrit ; les cas où rien ne se gagne restent intacts.
#include "gpu/vulkan/backend/net_hoist.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "net_rig.h"
#include "test_models.h"

using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
using ir::test::attr_int;
using ir::test::attr_string;
using ir::test::ModelBuilder;

namespace {

enum class Form { Scales, Sizes };

// x [2,7,H,W] → Split (3, 4) → Reshape [1,6] et [1,8] ; y [1,9] → Relu et Sigmoid ; Concat → Resize.
// extra : le Concat est aussi lu par un Relu (sa sortie pleine taille existe de toute façon) ;
// upscale : le Resize agrandit ; all_in_place : le Concat ne reçoit que les sorties de Relu et de Sigmoid.
ModelBuilder warped(Form form, bool extra, bool upscale, bool all_in_place = false) {
  ModelBuilder m;
  const std::int64_t h = upscale ? 4 : 16;
  const std::int64_t out = 8;
  m.input("x", {2, 7, h, h}).input("y", {1, 9, h, h});
  m.ints("split", {3, 4}).ints("shape_a", {1, 6, h, h}).ints("shape_b", {1, 8, h, h});
  m.node("Split", {"x", "split"}, {"s1", "s2"}, {attr_int("axis", 1)});
  m.node("Reshape", {"s1", "shape_a"}, {"a"}).node("Reshape", {"s2", "shape_b"}, {"b"});
  m.node("Relu", {"y"}, {"r"}).node("Sigmoid", {"y"}, {"q"});
  std::int64_t channels = 6 + 8 + 9 + 9;
  if (all_in_place) {
    m.node("Concat", {"r", "q"}, {"cat"}, {attr_int("axis", 1)});
    channels = 18;
  } else {
    m.node("Concat", {"a", "b", "r", "q"}, {"cat"}, {attr_int("axis", 1)});
  }
  if (form == Form::Scales) {
    const float scale = upscale ? 2.0F : 0.5F;
    m.weight("scales", {4}, {1.0F, 1.0F, scale, scale});
    m.node("Resize", {"cat", "", "scales"}, {"z"}, {attr_string("mode", "linear")});
  } else {
    m.ints("sizes", {1, channels, out, out});
    m.node("Resize", {"cat", "", "", "sizes"}, {"z"}, {attr_string("mode", "nearest")});
  }
  m.output("z");
  if (extra) {
    m.node("Relu", {"cat"}, {"w"});
    m.output("w");
  }
  return m;
}

ir::Graph build(const ModelBuilder& model) {
  ir::Graph graph = require_value(ir::load_onnx(model.bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  return graph;
}

std::vector<ir::NamedValue> inputs_for(const ir::Graph& graph) {
  std::vector<ir::NamedValue> in;
  std::size_t seed = 3;
  for (const ir::TensorId id : graph.inputs) {
    const ir::Tensor& t = graph.tensors[id];
    in.push_back({t.name, test_values(ir::DataType::F32, t.dims, seed++, -2.0F, 2.0F)});
  }
  return in;
}

std::size_t count_op(const ir::Graph& graph, ir::OpKind op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(graph.nodes, [&](const ir::Node& n) { return n.op == op; }));
}

// Plus grand nombre d'éléments écrit par un Concat, un Split ou un Reshape.
std::size_t largest_moved(const ir::Graph& graph) {
  std::size_t most = 0;
  for (const ir::Node& n : graph.nodes) {
    if (n.op != ir::OpKind::Concat && n.op != ir::OpKind::Split && n.op != ir::OpKind::Reshape) continue;
    for (const ir::TensorId id : n.outputs) most = std::max(most, ir::element_count(graph.tensors[id].dims));
  }
  return most;
}

void check_hoisted(Form form) {
  NetRig rig = make_net_rig();
  const ir::Graph before = build(warped(form, false, false));
  ir::Graph after = before;
  CHECK(vk::net::hoist_downscales(after) == 4);  // Concat, deux Reshape, puis le Split
  (void)ir::eliminate_dead_code(after);
  require_ok(ir::validate(after));
  // Plus de Concat, de Split ni de Reshape à la taille d'origine : tout est à celle de la sortie.
  CHECK(largest_moved(before) == 32U * 16 * 16);
  CHECK(largest_moved(after) == 32U * 8 * 8);
  CHECK(count_op(after, ir::OpKind::Resize) == 3);
  const std::vector<ir::NamedValue> inputs = inputs_for(before);
  const std::vector<ir::Value> want = run_reference(before, inputs);
  const std::vector<ir::Value> got = run_reference(after, inputs);
  REQUIRE(got.size() == want.size());
  CHECK(got[0].dims == want[0].dims);
  CHECK(got[0].f == want[0].f);  // identiques bit à bit
  const std::vector<ir::Value> device = run_on_device(rig, after, inputs);
  REQUIRE(device.size() == want.size());
  require_close(device[0], want[0], kFp32, "z");
}

}  // namespace

TEST_CASE("Réseau : Resize remonté devant Concat, Reshape et Split (échelles)", "[vk_backend][net][hoist]") {
  check_hoisted(Form::Scales);
}

TEST_CASE("Réseau : Resize remonté devant Concat, Reshape et Split (tailles)", "[vk_backend][net][hoist]") {
  check_hoisted(Form::Sizes);
}

TEST_CASE("Réseau : Resize remonté, cas laissés intacts", "[vk_backend][net][hoist]") {
  for (const Form form : {Form::Scales, Form::Sizes}) {
    // Le Concat est aussi lu ailleurs : sa sortie pleine taille existe de toute façon.
    ir::Graph shared = build(warped(form, true, false));
    CHECK(vk::net::hoist_downscales(shared) == 0);
    // Un Resize qui agrandit : le remonter ferait copier davantage.
    ir::Graph bigger = build(warped(form, false, true));
    CHECK(vk::net::hoist_downscales(bigger) == 0);
    // Toutes les entrées écrites en place par leur producteur : le Concat ne copie rien.
    ir::Graph in_place = build(warped(form, false, false, true));
    CHECK(vk::net::hoist_downscales(in_place) == 0);
  }
}
