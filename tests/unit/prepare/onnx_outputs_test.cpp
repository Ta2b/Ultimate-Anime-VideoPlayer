// Sorties d'un graphe ONNX changées au niveau protobuf, sur un modèle écrit à la main : le reste
// du modèle est recopié tel quel.
#include "prepare/onnx_outputs.h"

#include <cstddef>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

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
Bytes node(std::string_view input, std::string_view output, std::string_view op) {
  Bytes n;
  delimited(n, 1, text(input));
  delimited(n, 2, text(output));
  delimited(n, 4, text(op));
  return n;
}
Bytes value(std::string_view name) {
  Bytes tensor;
  varint(tensor, (1U << 3U) | 0U);
  varint(tensor, 10);
  Bytes type;
  delimited(type, 1, tensor);
  Bytes info;
  delimited(info, 1, text(name));
  delimited(info, 2, type);
  return info;
}

// ir_version 9 ; graphe x → Relu → a → Sigmoid → y, entrée x, sortie y ; opset 17.
Bytes model() {
  Bytes graph;
  delimited(graph, 1, node("x", "a", "Relu"));
  delimited(graph, 1, node("a", "y", "Sigmoid"));
  delimited(graph, 2, text("tiny"));
  delimited(graph, 11, value("x"));
  delimited(graph, 12, value("y"));
  Bytes out;
  varint(out, (1U << 3U) | 0U);
  varint(out, 9);
  delimited(out, 7, graph);
  Bytes opset;
  varint(opset, (2U << 3U) | 0U);
  varint(opset, 17);
  delimited(out, 8, opset);
  return out;
}

TEST_CASE("Les sorties du graphe sont remplacées, le reste du modèle recopié", "[prepare][onnx]") {
  const Bytes original = model();
  const auto produced = onnx_node_outputs(original);
  REQUIRE(produced);
  CHECK(*produced == std::vector<std::string>{"a", "y"});

  const std::vector<OnnxOutput> wanted{{.name = "a"}, {.name = "y", .elem_type = 1}};
  const auto derived = with_outputs(original, wanted);
  REQUIRE(derived);
  // Les nœuds et l'entrée restent : scan_onnx lit la même entrée.
  const auto scanned = scan_onnx(*derived);
  REQUIRE(scanned);
  REQUIRE(scanned->inputs.size() == 1);
  CHECK(scanned->inputs[0].name == "x");
  CHECK(*onnx_node_outputs(*derived) == std::vector<std::string>{"a", "y"});
  // Recommencer avec les mêmes sorties ne change plus rien.
  const auto again = with_outputs(*derived, wanted);
  REQUIRE(again);
  CHECK(*again == *derived);
}

TEST_CASE("Un tenseur que personne ne produit est refusé par son nom", "[prepare][onnx]") {
  const std::vector<OnnxOutput> wanted{{.name = "/Add_7_output_0"}};
  const auto derived = with_outputs(model(), wanted);
  REQUIRE_FALSE(derived);
  CHECK(derived.error().code == RefusalCode::FieldMissing);
  CHECK(derived.error().detail.find("/Add_7_output_0") != std::string::npos);
  const Bytes broken{std::byte{0x3a}, std::byte{0xff}};
  CHECK(with_outputs(broken, wanted).error().code == RefusalCode::ModelUnreadable);
}

}  // namespace
}  // namespace ajp::prepare
