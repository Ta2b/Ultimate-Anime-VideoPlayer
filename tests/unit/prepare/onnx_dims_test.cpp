// Dimensions nommées d'un graphe ONNX fixées au niveau protobuf, sur un modèle écrit à la main.
#include "prepare/onnx_dims.h"

#include <cstddef>
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

// Une dimension : un nombre, ou un nom.
struct Dim {
  std::int64_t value = 0;
  std::string_view param;
};

Bytes value(std::string_view name, std::initializer_list<Dim> dims) {
  Bytes shape;
  for (const Dim& d : dims) {
    Bytes dim;
    if (d.param.empty()) {
      varint(dim, (1U << 3U) | 0U);
      varint(dim, static_cast<std::uint64_t>(d.value));
    } else {
      delimited(dim, 2, text(d.param));
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

// x (1 × 6 × H × W) → AveragePool → y (1 × 3 × H/2 × W/2), une value_info au nom inconnu.
Bytes model() {
  Bytes graph;
  delimited(graph, 1, node("x", "a", "Relu"));
  delimited(graph, 1, node("a", "y", "AveragePool"));
  delimited(graph, 2, text("tiny"));
  delimited(graph, 11, value("x", {{1}, {6}, {0, "H"}, {0, "W"}}));
  delimited(graph, 12, value("y", {{1}, {3}, {0, "H/2"}, {0, "W/2"}}));
  delimited(graph, 13, value("a", {{1}, {6}, {0, "unk__7"}, {0, "W"}}));
  Bytes out;
  varint(out, (1U << 3U) | 0U);
  varint(out, 9);
  delimited(out, 7, graph);
  return out;
}

const std::vector<NamedDim> kSize{{"H", 192}, {"W", 256}, {"H/2", 96}, {"W/2", 128}};

TEST_CASE("Les dimensions nommées prennent leur valeur, le reste du modèle est recopié", "[prepare][onnx]") {
  const Bytes original = model();
  const auto fixed = with_fixed_dims(original, kSize);
  REQUIRE(fixed);
  const auto scanned = scan_onnx(*fixed);
  REQUIRE(scanned);
  REQUIRE(scanned->inputs.size() == 1);
  CHECK(scanned->inputs[0].name == "x");
  CHECK(scanned->inputs[0].dims == std::vector<std::int64_t>{1, 6, 192, 256});
  CHECK(*onnx_node_outputs(*fixed) == std::vector<std::string>{"a", "y"});
  // Recommencer ne change plus rien ; le modèle d'origine avait bien des dimensions nommées.
  CHECK(*with_fixed_dims(*fixed, kSize) == *fixed);
  CHECK(scan_onnx(original)->inputs[0].dims == std::vector<std::int64_t>{1, 6, -1, -1});
}

TEST_CASE("Une entrée ou une sortie qui garde un nom sans valeur est refusée", "[prepare][onnx]") {
  const std::vector<NamedDim> partial{{"H", 192}, {"W", 256}};
  const auto fixed = with_fixed_dims(model(), partial);
  REQUIRE_FALSE(fixed);
  CHECK(fixed.error().code == RefusalCode::FieldMissing);
  CHECK(fixed.error().detail.find("H/2") != std::string::npos);
  const std::vector<NamedDim> zero{{"H", 0}};
  CHECK(with_fixed_dims(model(), zero).error().code == RefusalCode::FieldOutOfRange);
  const Bytes broken{std::byte{0x3a}, std::byte{0xff}};
  CHECK(with_fixed_dims(broken, kSize).error().code == RefusalCode::ModelUnreadable);
}

}  // namespace
}  // namespace ajp::prepare
