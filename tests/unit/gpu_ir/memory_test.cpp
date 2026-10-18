// Disposition NHWC16 et plan mémoire.
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/memory_plan.h"
#include "gpu/ir/passes.h"
#include "test_util.h"

using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

// x -> Conv -> Conv -> ... -> y, toutes les activations de forme [1, 8, 8, 8].
ModelBuilder conv_chain(int count) {
  ModelBuilder m;
  m.input("x", {1, 8, 8, 8}).weight("w", {8, 8, 1, 1}, sequence(64, -0.5F, 0.5F, 1));
  std::string previous = "x";
  for (int k = 0; k < count; ++k) {
    const std::string out = k + 1 == count ? "y" : "t" + std::to_string(k);
    m.node("Conv", {previous, "w"}, {out});
    previous = out;
  }
  m.output("y");
  return m;
}

// Aucun couple de tampons vivants en même temps ne se recouvre dans l'arène.
void check_disjoint(const MemoryPlan& plan, std::uint64_t alignment = kBufferAlignment) {
  for (std::size_t a = 0; a < plan.allocations.size(); ++a) {
    const Allocation& p = plan.allocations[a];
    CHECK(p.offset % alignment == 0);
    CHECK(p.offset + p.bytes <= plan.arena_bytes);
    for (std::size_t b = a + 1; b < plan.allocations.size(); ++b) {
      const Allocation& q = plan.allocations[b];
      const bool alive_together = p.first <= q.last && q.first <= p.last;
      const bool apart = p.offset + p.bytes <= q.offset || q.offset + q.bytes <= p.offset;
      if (alive_together && !apart) FAIL("tampons " << p.tensor << " et " << q.tensor << " superposés");
    }
  }
  CHECK(plan.arena_bytes <= plan.unshared_bytes);
}

}  // namespace

TEST_CASE("mémoire : une chaîne de Conv alterne entre deux tampons", "[gpu_ir][memory]") {
  Graph graph = build(conv_chain(4));
  REQUIRE(optimize(graph));
  const MemoryPlan plan = plan_memory(graph);
  REQUIRE(plan.allocations.size() == 3);
  constexpr std::uint64_t kBuffer = 8 * 8 * 16 * 4;  // 8 canaux rangés sur 16, en F32
  for (const Allocation& a : plan.allocations) {
    CHECK(a.bytes == kBuffer);
    CHECK(a.last == a.first + 1);
  }
  CHECK(plan.unshared_bytes == 3 * kBuffer);
  CHECK(plan.arena_bytes == 2 * kBuffer);
  CHECK(plan.io_bytes == 2 * 8 * 8 * 8 * 4);
  CHECK(plan.constant_bytes == 64 * 4);
  check_disjoint(plan);
}

TEST_CASE("mémoire : un tampon lu plus loin reste réservé", "[gpu_ir][memory]") {
  // a est relu par le dernier Add : il vit pendant toute la chaîne.
  ModelBuilder m;
  m.input("x", {1, 4, 4, 4})
      .node("Relu", {"x"}, {"a"})
      .node("Sigmoid", {"a"}, {"b"})
      .node("Tanh", {"b"}, {"c"})
      .node("Relu", {"c"}, {"d"})
      .node("Add", {"a", "d"}, {"y"})
      .output("y");
  Graph graph = build(m);
  const MemoryPlan plan = plan_memory(graph);  // sans passes : disposition Nchw
  REQUIRE(plan.allocations.size() == 4);
  constexpr std::uint64_t kBuffer = 4 * 4 * 4 * 4;
  CHECK(plan.allocations[0].first == 0);
  CHECK(plan.allocations[0].last == 4);
  CHECK(plan.arena_bytes == 3 * std::max<std::uint64_t>(kBuffer, kBufferAlignment));
  check_disjoint(plan);
}

TEST_CASE("mémoire : tailles diverses, alignement choisi", "[gpu_ir][memory]") {
  ModelBuilder m;
  m.input("x", {1, 3, 9, 7})
      .weight("w1", {5, 3, 3, 3}, sequence(135, -1, 1, 2))
      .weight("w2", {17, 5, 1, 1}, sequence(85, -1, 1, 3))
      .weight("w3", {2, 17, 3, 3}, sequence(306, -1, 1, 4))
      .node("Conv", {"x", "w1"}, {"a"}, {attr_ints("pads", {1, 1, 1, 1})})
      .node("Conv", {"a", "w2"}, {"b"})
      .node("Conv", {"b", "w3"}, {"c"}, {attr_ints("strides", {2, 2})})
      .node("Relu", {"c"}, {"d"})
      .node("Concat", {"a", "a"}, {"e"}, {attr_int("axis", 1)})
      .node("ReduceMean", {"e"}, {"f"}, {attr_ints("axes", {1})})
      .output("d")
      .output("f");
  Graph graph = build(m);
  REQUIRE(optimize(graph));
  for (const std::uint64_t alignment : {std::uint64_t{1}, std::uint64_t{64}, kBufferAlignment}) {
    const MemoryPlan plan = plan_memory(graph, alignment);
    for (const Allocation& a : plan.allocations) CHECK(a.bytes % alignment == 0);
    check_disjoint(plan, alignment);
  }
  // b compte 17 canaux : 32 en NHWC16.
  const auto b = find_tensor(graph, "b");
  REQUIRE(b);
  CHECK(graph.tensors[*b].layout == Layout::Nhwc16);
  CHECK(storage_bytes(graph.tensors[*b]) == 9 * 7 * 32 * 4);
}

TEST_CASE("disposition : les nœuds qui réinterprètent les axes gardent Nchw", "[gpu_ir][memory]") {
  ModelBuilder m;
  m.input("x", {1, 4, 4, 4})
      .weight("w", {8, 4, 1, 1}, sequence(32, -1, 1, 5))
      .weight("v", {2, 8, 1, 1}, sequence(16, -1, 1, 6))
      .ints("shape", {1, 8, 2, 8})
      .node("Conv", {"x", "w"}, {"c"})
      .node("Relu", {"c"}, {"r"})
      .node("Reshape", {"r", "shape"}, {"s"})
      .node("Conv", {"s", "v"}, {"y"})
      .output("y");
  Graph graph = build(m);
  REQUIRE(optimize(graph));
  // Relu est fondue dans la Conv ; sa sortie est lue par Reshape, qui ne sait pas lire NHWC.
  for (const Tensor& t : graph.tensors) {
    if (t.kind == TensorKind::Activation) CHECK(t.layout == Layout::Nchw);
  }
  const Value got = execute1(graph, {{.name = "x", .value = random_floats({1, 4, 4, 4}, 7)}});
  CHECK(got.dims == Dims{1, 2, 2, 8});
}
