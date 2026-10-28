// Copies et conversions que le moteur de réseaux évite. Tenseurs sans copie (net_alias.h) :
// Concat écrit en place par ses producteurs, Split et Reshape lus dans leur source, sortie du
// graphe comme racine ; chaque cas est comparé à l'interpréteur, puis refait avec un alignement
// que rien ne tient, où les copies reviennent et le résultat ne change pas. Cast retirés
// (fold_float_casts) : l'appareil sur le graphe sans eux, contre l'interpréteur sur le graphe
// d'origine.
#include "gpu/vulkan/backend/net_alias.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "../gpu_ir/onnx_builder.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "net_rig.h"

using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;
using ir::test::attr_float;
using ir::test::attr_int;
using ir::test::attr_string;
using ir::test::ModelBuilder;

namespace {

// x → Relu, Sigmoid → Concat → Split (2, 4) → Tanh et Reshape → LeakyRelu ; concat_out : le
// Concat est aussi une sortie du graphe, racine de ses entrées et des sorties du Split.
ModelBuilder copies(std::int32_t type, bool concat_out) {
  ModelBuilder m;
  m.input("x", {1, 3, 16, 16}, type);
  m.ints("split", {2, 4}).ints("shape", {1, 4, 256});
  m.node("Relu", {"x"}, {"r1"}).node("Sigmoid", {"x"}, {"r2"});
  m.node("Concat", {"r1", "r2"}, {"cat"}, {attr_int("axis", 1)});
  m.node("Split", {"cat", "split"}, {"s1", "s2"}, {attr_int("axis", 1)});
  m.node("Tanh", {"s1"}, {"y1"});
  m.node("Reshape", {"s2", "shape"}, {"flat"});
  m.node("LeakyRelu", {"flat"}, {"y2"}, {attr_float("alpha", 0.5F)});
  m.output("y1", type).output("y2", type);
  if (concat_out) {
    // Le Transpose garde r2 en Nchw, et le Concat y couple r1 : sans quoi, la sortie du graphe
    // étant seule en Nchw, ses entrées passeraient en Nhwc16 et devraient être recopiées.
    m.node("Transpose", {"r2"}, {"t"}, {ir::test::attr_ints("perm", {0, 1, 3, 2})});
    m.output("cat", type).output("t", type);
  }
  return m;
}

ir::Graph build(const ModelBuilder& model) {
  ir::Graph graph = require_value(ir::load_onnx(model.bytes()));
  require_ok(ir::infer_shapes(graph));
  (void)require_value(ir::optimize(graph));
  return graph;
}

std::size_t copy_dispatches(const ir::Graph& graph, const vk::net::Target& target) {
  const auto choices = require_value(vk::net::default_choices(graph, target));
  const vk::net::Program program = require_value(vk::net::lower(graph, choices, target));
  return static_cast<std::size_t>(std::ranges::count_if(program.dispatches, [&](const vk::net::Dispatch& d) {
    const ir::OpKind op = graph.nodes[d.node].op;
    return op == ir::OpKind::Concat || op == ir::OpKind::Split || op == ir::OpKind::Reshape;
  }));
}

void check_copies(NetRig& rig, ir::DataType type, bool concat_out) {
  const std::int32_t onnx = type == ir::DataType::F16 ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  const ir::Graph graph = build(copies(onnx, concat_out));
  const std::vector<ir::NamedValue> inputs{{"x", test_values(type, {1, 3, 16, 16}, 5, -3.0F, 3.0F)}};
  const Tolerance tolerance = type == ir::DataType::F16 ? kFp16 : kFp32;
  INFO((concat_out ? "Concat en sortie" : "Concat interne")
       << ", alignement " << rig.target.storage_alignment);
  // Un canal en float16 fait 512 octets : chaque tranche tombe sur un décalage que l'appareil lie.
  if (rig.target.storage_alignment <= 512) CHECK(copy_dispatches(graph, rig.target) == 0);
  check_graph(rig, graph, inputs, tolerance);

  // Seules les tranches au décalage 0 restent des vues : r2 et s2 sont recopiés.
  const std::uint32_t kept = rig.target.storage_alignment;
  rig.target.storage_alignment = 1U << 20;
  CHECK(copy_dispatches(graph, rig.target) == 2);
  check_graph(rig, graph, inputs, tolerance);
  rig.target.storage_alignment = kept;
}

}  // namespace

TEST_CASE("Réseau : Concat, Split et Reshape sans copie", "[vk_backend][net][alias]") {
  NetRig rig = make_net_rig();
  for (const bool concat_out : {false, true}) check_copies(rig, ir::DataType::F32, concat_out);
  require_f16(rig);
  for (const bool concat_out : {false, true}) check_copies(rig, ir::DataType::F16, concat_out);
}

TEST_CASE("Réseau : plan des alias", "[vk_backend][net][alias]") {
  const ir::Graph graph = build(copies(ir::onnx::kFloat, false));
  const auto id = [&](std::string_view name) {
    return static_cast<ir::TensorId>(std::ranges::find(graph.tensors, name, &ir::Tensor::name) -
                                     graph.tensors.begin());
  };
  const std::vector<ir::Alias> aliases = vk::net::plan_aliases(graph, 16);
  const std::uint64_t channel = 16 * 16 * 4;
  CHECK(vk::net::aliased(aliases, id("r1"), id("cat"), 0));
  CHECK(vk::net::aliased(aliases, id("r2"), id("cat"), 3 * channel));
  CHECK(vk::net::aliased(aliases, id("s1"), id("cat"), 0));
  CHECK(vk::net::aliased(aliases, id("s2"), id("cat"), 2 * channel));
  CHECK(vk::net::aliased(aliases, id("flat"), id("s2"), 0));
  // Les sorties et l'entrée du graphe gardent leur propre tampon.
  for (const ir::TensorId k : {id("x"), id("y1"), id("y2"), id("cat")})
    CHECK(aliases[k].parent == ir::kNoTensor);

  // La famille entière a une place, réservée du premier producteur au dernier lecteur.
  const ir::MemoryPlan plan = require_value(ir::plan_memory(graph, 256, aliases));
  REQUIRE(plan.allocations.size() == 1);
  CHECK(plan.allocations[0].tensor == id("cat"));
  CHECK(plan.allocations[0].first == 0);
  CHECK(plan.allocations[0].last == graph.nodes.size() - 1);
}

TEST_CASE("Réseau : Cast autour d'un Resize retirés", "[vk_backend][net][alias]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  ModelBuilder m;
  m.input("x", {1, 3, 8, 8}, ir::onnx::kFloat16).weight("scales", {4}, {1.0F, 1.0F, 2.0F, 2.0F});
  m.node("Relu", {"x"}, {"h"});
  m.node("Cast", {"h"}, {"a"}, {attr_int("to", ir::onnx::kFloat)});
  m.node("Resize", {"a", "", "scales"}, {"r"}, {attr_string("mode", "linear")});
  m.node("Cast", {"r"}, {"rh"}, {attr_int("to", ir::onnx::kFloat16)});
  m.node("Sigmoid", {"rh"}, {"y"});
  m.output("y", ir::onnx::kFloat16);
  const ir::Graph before = build(m);
  ir::Graph after = before;
  CHECK(ir::fold_float_casts(after) == 2);
  CHECK(std::ranges::none_of(after.nodes, [](const ir::Node& n) { return n.op == ir::OpKind::Cast; }));
  const std::vector<ir::NamedValue> inputs{
      {"x", test_values(ir::DataType::F16, {1, 3, 8, 8}, 6, -2.0F, 2.0F)}};
  const std::vector<ir::Value> want = run_reference(before, inputs);
  const std::vector<ir::Value> got = run_on_device(rig, after, inputs);
  REQUIRE(got.size() == 1);
  require_close(got[0], want[0], kFp16, "y");
}
