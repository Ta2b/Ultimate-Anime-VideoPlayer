// L'entrée empilée d'un graphe ONNX coupée en images (TemporalFix), sur un modèle écrit à la main.
#include "prepare/onnx_frames.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/onnx_outputs.h"
#include "prepare/onnx_scan.h"

namespace ajp::prepare {
namespace {

using Bytes = std::vector<std::byte>;

void varint(Bytes& out, std::uint64_t v) {
  while (v >= 0x80U) {
    out.push_back(static_cast<std::byte>((v & 0x7fU) | 0x80U));
    v >>= 7U;
  }
  out.push_back(static_cast<std::byte>(v));
}
void delimited(Bytes& out, std::uint32_t number, const Bytes& payload) {
  varint(out, (std::uint64_t{number} << 3U) | 2U);
  varint(out, payload.size());
  out.insert(out.end(), payload.begin(), payload.end());
}
Bytes text(std::string_view s) {
  return {reinterpret_cast<const std::byte*>(s.data()),
          reinterpret_cast<const std::byte*>(s.data()) + s.size()};
}

// Une dimension fixée, ou nommée (value 0).
Bytes value(std::string_view name, std::initializer_list<std::int64_t> dims, std::string_view param = {}) {
  Bytes shape;
  for (const std::int64_t d : dims) {
    Bytes dim;
    if (d > 0) {
      varint(dim, (1U << 3U) | 0U);
      varint(dim, static_cast<std::uint64_t>(d));
    } else {
      delimited(dim, 2, text(param));
    }
    delimited(shape, 1, dim);
  }
  Bytes tensor;
  varint(tensor, (1U << 3U) | 0U);
  varint(tensor, 10);
  delimited(tensor, 2, shape);
  Bytes type;
  delimited(type, 1, tensor);
  Bytes info;
  delimited(info, 1, text(name));
  delimited(info, 2, type);
  return info;
}

Bytes node(std::string_view input, std::string_view output, std::string_view op) {
  Bytes n;
  delimited(n, 1, text(input));
  delimited(n, 2, text(output));
  delimited(n, 4, text(op));
  return n;
}

// input (1 × 21 × 8 × 12) → Relu → output.
Bytes model(std::int64_t height = 8) {
  Bytes graph;
  delimited(graph, 1, node("input", "a", "Relu"));
  delimited(graph, 1, node("a", "output", "Identity"));
  delimited(graph, 2, text("pile"));
  delimited(graph, 11, value("input", {1, 21, height, 12}, "height"));
  delimited(graph, 12, value("output", {1, 3, height, 12}, "height"));
  Bytes out;
  varint(out, (1U << 3U) | 0U);
  varint(out, 9);
  delimited(out, 7, graph);
  return out;
}

TEST_CASE("L'entrée empilée devient sept images, que le Concat de tête refait", "[prepare][onnx]") {
  const auto split = with_frame_inputs(model(), "input", 7);
  REQUIRE(split);
  const auto scanned = scan_onnx(*split);
  REQUIRE(scanned);
  REQUIRE(scanned->inputs.size() == 7);
  for (int k = 0; k < 7; ++k) {
    CHECK(scanned->inputs[static_cast<std::size_t>(k)].name == frame_input_name("input", k));
    CHECK(scanned->inputs[static_cast<std::size_t>(k)].elem_type == 10);
    CHECK(scanned->inputs[static_cast<std::size_t>(k)].dims == std::vector<std::int64_t>{1, 3, 8, 12});
  }
  const auto nodes = onnx_nodes(*split);
  REQUIRE(nodes);
  REQUIRE(nodes->size() == 3);
  CHECK((*nodes)[0].op_type == "Concat");
  CHECK((*nodes)[0].inputs == std::vector<std::string>{"input_0", "input_1", "input_2", "input_3", "input_4",
                                                       "input_5", "input_6"});
  CHECK((*nodes)[0].outputs == std::vector<std::string>{"input"});
  CHECK((*nodes)[1].op_type == "Relu");
  CHECK(scanned->nodes == 3);
}

TEST_CASE("Une entrée absente, dynamique ou indivisible est refusée", "[prepare][onnx]") {
  CHECK(with_frame_inputs(model(), "x", 7).error().code == RefusalCode::FieldMissing);
  CHECK(with_frame_inputs(model(0), "input", 7).error().code == RefusalCode::FormatUnsupported);
  CHECK(with_frame_inputs(model(), "input", 5).error().code == RefusalCode::FormatUnsupported);
  CHECK(with_frame_inputs(model(), "input", 1).error().code == RefusalCode::FieldOutOfRange);
  const Bytes broken{std::byte{0x3a}, std::byte{0xff}};
  CHECK_FALSE(with_frame_inputs(broken, "input", 7));
  // Coupé une fois, il n'a plus d'entrée « input » : une seconde coupe est refusée.
  const auto split = with_frame_inputs(model(), "input", 7);
  REQUIRE(split);
  CHECK(with_frame_inputs(*split, "input", 7).error().code == RefusalCode::FieldMissing);
}

}  // namespace
}  // namespace ajp::prepare
