// Une métadonnée d'un ONNX lue dans le fichier sans charger le graphe, sur des modèles écrits à la main.
#include "prepare/onnx_metadata.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

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
Bytes entry(std::string_view key, std::string_view value) {
  Bytes e;
  delimited(e, 1, text(key));
  delimited(e, 2, text(value));
  return e;
}

// ir_version 9, un graphe de 300 Kio d'octets quelconques (ses poids), l'opset 17, puis les métadonnées.
Bytes model(const std::vector<std::pair<std::string, std::string>>& metadata) {
  Bytes out;
  varint(out, (1U << 3U) | 0U);
  varint(out, 9);
  Bytes graph(300 * 1024, std::byte{0x5a});
  delimited(out, 7, graph);
  Bytes opset;
  varint(opset, (2U << 3U) | 0U);
  varint(opset, 17);
  delimited(out, 8, opset);
  for (const auto& [key, value] : metadata) delimited(out, 14, entry(key, value));
  return out;
}

std::filesystem::path write(const std::string& name, const Bytes& bytes) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / ("ajp_onnx_metadata_" + name);
  std::ofstream(path, std::ios::binary)
      .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return path;
}

}  // namespace

TEST_CASE("La métadonnée d'un ONNX se lit sans passer par le graphe", "[prepare][onnx]") {
  const auto path = write("known.onnx", model({{"ajp.gmfss", "base"}, {"ajp.gmfss.export", "2"}}));
  const auto found = onnx_metadata(path, "ajp.gmfss.export");
  REQUIRE(found);
  REQUIRE(*found);
  CHECK(**found == "2");
  CHECK(**onnx_metadata(path, "ajp.gmfss") == "base");
  // Une clé absente n'est pas une erreur.
  const auto missing = onnx_metadata(path, "ajp.autre");
  REQUIRE(missing);
  CHECK_FALSE(*missing);
  // Répétée : la dernière valeur.
  const auto twice = write("twice.onnx", model({{"k", "1"}, {"k", "3"}}));
  CHECK(**onnx_metadata(twice, "k") == "3");
  // Sans aucune métadonnée.
  const auto none = write("none.onnx", model({}));
  const auto empty = onnx_metadata(none, "k");
  REQUIRE(empty);
  CHECK_FALSE(*empty);
  for (const auto& file : {path, twice, none}) std::filesystem::remove(file);
}

TEST_CASE("Un ONNX tronqué, malformé ou absent est refusé", "[prepare][onnx]") {
  Bytes truncated = model({{"k", "1"}});
  truncated.resize(truncated.size() - 3);
  const auto cut = write("cut.onnx", truncated);
  const auto refused = onnx_metadata(cut, "k");
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::ModelUnreadable);
  const auto garbage = write("garbage.onnx", text("x"));
  CHECK_FALSE(onnx_metadata(garbage, "k"));
  CHECK_FALSE(onnx_metadata(std::filesystem::temp_directory_path() / "ajp_onnx_metadata_absent.onnx", "k"));
  const auto zero = write("zero.onnx", {});
  CHECK_FALSE(onnx_metadata(zero, "k"));
  for (const auto& file : {cut, garbage, zero}) std::filesystem::remove(file);
}

}  // namespace ajp::prepare
