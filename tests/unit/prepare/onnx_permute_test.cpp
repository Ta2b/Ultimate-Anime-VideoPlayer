// permute_input_channels (prepare/onnx_split.h) : les poids d'une convolution permutés sur leur axe d'entrée,
// ce que la découpe de RIFE 4.25 lite fait à la première convolution de chaque bloc (rife_split.h).
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/onnx_split.h"
#include "support.h"

namespace ajp::prepare {
namespace {

using test::field;

// Un modèle d'un seul nœud Conv dont le poids k (2 × 3 × 1 × 1, FP32 en raw_data) vaut values.
std::string conv_model(std::initializer_list<float> values) {
  std::string raw;
  for (const float v : values) {
    const auto bits = std::bit_cast<std::uint32_t>(v);
    raw.append(reinterpret_cast<const char*>(&bits), 4);
  }
  const std::string tensor =
      field(1, 2) + field(1, 3) + field(1, 1) + field(1, 1) + field(2, 1) + field(8, "k") + field(9, raw);
  const std::string node = field(1, "x") + field(1, "k") + field(2, "y") + field(3, "c") + field(4, "Conv");
  return field(1, 8) + field(7, field(1, node) + field(5, tensor));
}

}  // namespace

TEST_CASE("Les poids d'une convolution se permutent sur leur axe d'entrée", "[prepare][onnx][rife_fused]") {
  const std::string text = conv_model({0, 1, 2, 10, 11, 12});
  const std::span<const std::byte> model = test::as_bytes(text);
  const std::array<int, 3> source{2, 0, 1};
  const auto moved = permute_input_channels(model, "k", source);
  REQUIRE(moved);
  const auto parsed = read_onnx_graph(*moved);
  REQUIRE(parsed);
  CHECK(parsed->constants.at("k") == std::vector<double>{2, 0, 1, 12, 10, 11});
  CHECK(parsed->nodes.size() == 1);
  CHECK(permute_input_channels(model, "absent", source).error().code == RefusalCode::FieldMissing);
  const std::array<int, 3> twice{0, 0, 1};
  CHECK(permute_input_channels(model, "k", twice).error().code == RefusalCode::FormatUnsupported);
  const std::array<int, 2> short_one{1, 0};
  CHECK(permute_input_channels(model, "k", short_one).error().code == RefusalCode::FormatUnsupported);
}

}  // namespace ajp::prepare
