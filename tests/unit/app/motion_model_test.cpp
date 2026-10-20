// La variante mouvement de RIFE sur des graphes écrits à la main : le flot final est le candidat
// que seul le Reshape du warp final consomme, quel que soit son nom ; un ensemble n'en a pas.
#include "app/play/motion_model.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/onnx_outputs.h"

namespace ajp::app::play {
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
Bytes node(std::initializer_list<std::string_view> inputs, std::string_view output, std::string_view op) {
  Bytes n;
  for (const std::string_view input : inputs) delimited(n, 1, text(input));
  delimited(n, 2, text(output));
  delimited(n, 4, text(op));
  return n;
}
// Entrée « input » FP16 1 × 7 × 64 × 64.
Bytes input_info() {
  Bytes shape;
  for (const std::uint64_t d : {1U, 7U, 64U, 64U}) {
    Bytes dim;
    varint(dim, (1U << 3U) | 0U);
    varint(dim, d);
    delimited(shape, 1, dim);
  }
  Bytes tensor;
  varint(tensor, (1U << 3U) | 0U);
  varint(tensor, 10);
  delimited(tensor, 2, shape);
  Bytes type;
  delimited(type, 1, tensor);
  Bytes info;
  delimited(info, 1, text("input"));
  delimited(info, 2, type);
  return info;
}

// Comme les v2 de 4.18 et 4.22 lite : /Add_5 est le flot final, /Add_7 une image qui part vers un
// Pad. Le warp final suit les exports FP16 : Reshape, Cast vers fp32, Mul par le pas, Add du
// maillage, Transpose, GridSample. ensemble : /Add_5 part aussi vers un autre bloc.
std::filesystem::path write_rife(const std::filesystem::path& dir, bool ensemble) {
  Bytes graph;
  delimited(graph, 1, node({"input", "input"}, "/Add_5_output_0", "Add"));
  delimited(graph, 1, node({"/Add_5_output_0", "shape"}, "/Reshape_9_output_0", "Reshape"));
  delimited(graph, 1, node({"/Reshape_9_output_0"}, "/Reshape_9_output_0/cast32", "Cast"));
  delimited(graph, 1, node({"/Reshape_9_output_0/cast32", "step"}, "/Mul_6_output_0", "Mul"));
  delimited(graph, 1, node({"mesh", "/Mul_6_output_0"}, "/Add_8_output_0", "Add"));
  delimited(graph, 1, node({"/Add_8_output_0"}, "/Transpose_3_output_0", "Transpose"));
  delimited(graph, 1, node({"input", "/Transpose_3_output_0"}, "/GridSample_3_output_0", "GridSample"));
  if (ensemble) delimited(graph, 1, node({"/Add_5_output_0", "input"}, "/Concat_2_output_0", "Concat"));
  delimited(graph, 1, node({"input", "input"}, "/Add_7_output_0", "Add"));
  delimited(graph, 1, node({"/Add_7_output_0"}, "/Pad_1_output_0", "Pad"));
  delimited(graph, 1, node({"input"}, "/Sigmoid_output_0", "Sigmoid"));
  delimited(graph, 11, input_info());
  Bytes model;
  varint(model, (1U << 3U) | 0U);
  varint(model, 9);
  delimited(model, 7, graph);
  std::filesystem::create_directories(dir);
  const std::filesystem::path path = dir / std::format("rife_v4.18{}.onnx", ensemble ? "_ensemble" : "");
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(model.data()), static_cast<std::streamsize>(model.size()));
  return path;
}

Bytes read(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  const std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto bytes = std::as_bytes(std::span(raw));
  return {bytes.begin(), bytes.end()};
}

}  // namespace

TEST_CASE("RIFE mouvement : le flot final reconnu à son warp, pas à son nom", "[app][motion_model]") {
  const auto dir = std::filesystem::temp_directory_path() / "ajp-motion-model";
  std::filesystem::remove_all(dir);
  const auto derived = motion_model(write_rife(dir, false), dir / "modeles");
  REQUIRE(derived);
  // Ses sorties sont exactement le flot /Add_5 et le masque, en FP16 comme l'entrée.
  const Bytes model = read(*derived);
  const std::vector<prepare::OnnxOutput> wanted{{.name = "/Add_5_output_0"}, {.name = "/Sigmoid_output_0"}};
  CHECK(*prepare::with_outputs(model, wanted) == model);
  CHECK(rife_input_planes(dir / "rife_v4.18.onnx") == 7);

  const auto ensemble = motion_model(write_rife(dir, true), dir / "modeles");
  REQUIRE_FALSE(ensemble);
  CHECK(ensemble.error().code == RefusalCode::FieldMissing);
  std::filesystem::remove_all(dir);
}

}  // namespace ajp::app::play
