// Dépliage de pixels 6-D réécrit en 5-D (onnx_unshuffle.h) : un Slice devant le premier Reshape,
// la permutation [0,2,4,1,3], les value_info des tenseurs changés de rang retirées ; la sortie
// est celle de pixel_unshuffle par l'interpréteur ; rien n'est réécrit hors du motif.
#include "gpu/ir/onnx_unshuffle.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/onnx_model.h"
#include "gpu/ir/protobuf.h"
#include "test_util.h"

using namespace ajp::gpu::ir;
using test::ModelBuilder;

namespace {

using Names = std::vector<std::string>;

// x [1, 2, 4, 6] déplié par 2 : [1, 8, 2, 3], comme l'exporte PyTorch.
ModelBuilder unshuffle(std::vector<std::int64_t> perm = {0, 1, 3, 5, 2, 4}) {
  ModelBuilder m;
  m.input("x", {1, 2, 4, 6})
      .ints("shape6", {1, 2, 2, 2, 3, 2})
      .ints("shape4", {1, 8, 2, 3})
      .node("Reshape", {"x", "shape6"}, {"a"})
      .node("Transpose", {"a"}, {"b"}, {test::attr_ints("perm", std::move(perm))})
      .node("Reshape", {"b", "shape4"}, {"y"});
  m.graph().message(13, test::value_info("a", onnx::kFloat, {1, 2, 2, 2, 3, 2}));
  m.graph().message(13, test::value_info("b", onnx::kFloat, {1, 2, 2, 2, 2, 3}));
  m.graph().message(13, test::value_info("y", onnx::kFloat, {1, 8, 2, 3}));
  return m;
}

// Les noms des value_info (champ 13 de GraphProto), que le décodeur n'expose pas.
Names value_info_names(std::span<const std::byte> model) {
  Names names;
  proto::Reader reader(model);
  for (auto field = reader.next(); field && *field; field = reader.next()) {
    if ((*field)->number != 7) continue;
    proto::Reader graph((*field)->bytes);
    for (auto g = graph.next(); g && *g; g = graph.next()) {
      if ((*g)->number != 13) continue;
      proto::Reader info((*g)->bytes);
      for (auto i = info.next(); i && *i; i = info.next()) {
        if ((*i)->number == 1) names.push_back(*proto::as_string(**i));
      }
    }
  }
  return names;
}

const onnx::Tensor& initializer(const onnx::Graph& g, std::string_view name) {
  for (const onnx::Tensor& t : g.initializers) {
    if (t.name == name) return t;
  }
  FAIL("poids absent : " << name);
  return g.initializers.front();
}

}  // namespace

TEST_CASE("onnx : le dépliage 6-D passe en 5-D, sortie de pixel_unshuffle", "[gpu_ir][onnx]") {
  ModelBuilder m = unshuffle();
  m.output("y");
  auto rewritten = onnx::with_unshuffle_5d(m.bytes());
  REQUIRE(rewritten);
  REQUIRE_FALSE(rewritten->empty());

  auto model = onnx::read_onnx(*rewritten);
  REQUIRE(model);
  const onnx::Graph& g = model->graph;
  REQUIRE(g.nodes.size() == 4);
  CHECK(g.nodes[0].op_type == "Slice");
  CHECK(g.nodes[0].inputs == Names{"shape6", "ajp.unbatch.starts", "ajp.unbatch.ends"});
  CHECK(g.nodes[0].outputs == Names{"shape6.5d"});
  CHECK(g.nodes[1].op_type == "Reshape");
  CHECK(g.nodes[1].inputs == Names{"x", "shape6.5d"});
  CHECK(g.nodes[2].op_type == "Transpose");
  REQUIRE(g.nodes[2].attributes.size() == 1);
  CHECK(g.nodes[2].attributes[0].name == "perm");
  CHECK(g.nodes[2].attributes[0].type == onnx::AttributeType::Ints);
  CHECK(g.nodes[2].attributes[0].ints == std::vector<std::int64_t>{0, 2, 4, 1, 3});
  CHECK(g.nodes[3].inputs == Names{"b", "shape4"});
  CHECK(initializer(g, "ajp.unbatch.starts").ints == std::vector<std::int64_t>{1});
  CHECK(initializer(g, "ajp.unbatch.ends").ints == std::vector<std::int64_t>{6});
  CHECK(initializer(g, "shape6").ints == std::vector<std::int64_t>{1, 2, 2, 2, 3, 2});  // intact
  CHECK(value_info_names(*rewritten) == Names{"y"});

  auto graph = load_onnx(*rewritten);
  REQUIRE(graph);
  REQUIRE(infer_shapes(*graph, {}));
  const std::vector<float> values = test::sequence(48, -1.0F, 1.0F);
  const Value got = test::execute1(*graph, {{"x", test::floats({1, 2, 4, 6}, values)}});
  REQUIRE(got.f.size() == 48);
  // y[0, c·4 + i·2 + j, h, w] = x[0, c, 2h + i, 2w + j]
  std::vector<float> expected(48);
  for (int c = 0; c < 2; ++c) {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int h = 0; h < 2; ++h) {
          for (int w = 0; w < 3; ++w) {
            const int out = (((c * 4) + (i * 2) + j) * 2 + h) * 3 + w;
            const int in = ((c * 4) + (2 * h) + i) * 6 + (2 * w) + j;
            expected[static_cast<std::size_t>(out)] = values[static_cast<std::size_t>(in)];
          }
        }
      }
    }
  }
  CHECK(got.f == expected);
}

TEST_CASE("onnx : rien à réécrire hors du dépliage 6-D", "[gpu_ir][onnx]") {
  SECTION("une autre permutation") {
    ModelBuilder m = unshuffle({0, 1, 2, 3, 4, 5});
    m.output("y");
    auto same = onnx::with_unshuffle_5d(m.bytes());
    REQUIRE(same);
    CHECK(same->empty());
  }
  SECTION("la forme 6-D relue ailleurs") {
    ModelBuilder m = unshuffle();
    m.node("Relu", {"a"}, {"z"}).output("y").output("z");
    auto same = onnx::with_unshuffle_5d(m.bytes());
    REQUIRE(same);
    CHECK(same->empty());
  }
  SECTION("la sortie permutée est une sortie du graphe") {
    ModelBuilder m = unshuffle();
    m.output("y").output("b");
    auto same = onnx::with_unshuffle_5d(m.bytes());
    REQUIRE(same);
    CHECK(same->empty());
  }
  SECTION("un opset sans Slice à bornes en entrées") {
    ModelBuilder m(9);
    m.input("x", {1, 2, 4, 6})
        .ints("shape6", {1, 2, 2, 2, 3, 2})
        .node("Reshape", {"x", "shape6"}, {"a"})
        .node("Transpose", {"a"}, {"y"}, {test::attr_ints("perm", {0, 1, 3, 5, 2, 4})})
        .output("y");
    auto same = onnx::with_unshuffle_5d(m.bytes());
    REQUIRE(same);
    CHECK(same->empty());
  }
}
