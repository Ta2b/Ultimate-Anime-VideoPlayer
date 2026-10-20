// Lecture d'ONNX synthétiques écrits au format protobuf par support.h.
#include "prepare/onnx_scan.h"

#include <catch2/catch_test_macros.hpp>

#include "support.h"

using namespace ajp::prepare;
using ajp::RefusalCode;

TEST_CASE("scan_onnx : entrées, types, dimensions et GridSample", "[prepare][onnx]") {
  const std::string model =
      test::onnx({.inputs = {{.name = "input", .elem_type = 10, .dims = {1, 11, -1, -1}}},
                  .ops = {"Conv", "GridSample", "Add", "GridSample"},
                  .initializers = {"weight"}});
  const auto summary = scan_onnx(test::as_bytes(model));
  REQUIRE(summary);
  REQUIRE(summary->inputs.size() == 1);  // l'initialiseur déclaré en entrée n'est pas une entrée
  CHECK(summary->inputs[0].name == "input");
  CHECK(summary->inputs[0].elem_type == static_cast<int>(OnnxType::Float16));
  CHECK(summary->inputs[0].dims == std::vector<std::int64_t>{1, 11, -1, -1});
  CHECK_FALSE(summary->inputs[0].is_static());
  CHECK(summary->grid_samples == 2);
  CHECK(summary->nodes == 4);
  CHECK(summary->external_locations.empty());
}

TEST_CASE("scan_onnx : une entrée entièrement fixée est statique", "[prepare][onnx]") {
  const std::string model = test::onnx({.inputs = {{.elem_type = 1, .dims = {1, 3, 720, 1280}}}});
  const auto summary = scan_onnx(test::as_bytes(model));
  REQUIRE(summary);
  CHECK(summary->inputs[0].is_static());
  CHECK(summary->inputs[0].elem_type == static_cast<int>(OnnxType::Float));
}

TEST_CASE("onnx_external_locations : triés, dédoublonnés", "[prepare][onnx]") {
  const std::string model = test::onnx({.external = {"poids/b.bin", "a.bin", "poids/b.bin"}});
  const auto locations = onnx_external_locations(test::as_bytes(model));
  REQUIRE(locations);
  CHECK(*locations == std::vector<std::string>{"a.bin", "poids/b.bin"});
}

TEST_CASE("onnx_external_locations : poids d'un sous-graphe d'attribut", "[prepare][onnx]") {
  // NodeProto.attribute (5) → AttributeProto.g (6) → GraphProto.initializer (5).
  const std::string subgraph = test::field(5, test::initializer("w", "branche.bin"));
  const std::string node = test::field(4, std::string_view("If")) + test::field(5, test::field(6, subgraph));
  const std::string model = test::field(7, test::field(1, node));
  const auto locations = onnx_external_locations(test::as_bytes(model));
  REQUIRE(locations);
  CHECK(*locations == std::vector<std::string>{"branche.bin"});
}

TEST_CASE("scan_onnx : refus d'un ONNX malformé", "[prepare][onnx]") {
  const std::string model = test::onnx({});
  CHECK(test::refusal(scan_onnx(test::as_bytes(std::string_view(model).substr(0, model.size() - 3)))) ==
        RefusalCode::ModelUnreadable);
  CHECK(test::refusal(scan_onnx(test::as_bytes(""))) == RefusalCode::ModelUnreadable);
  CHECK(test::refusal(scan_onnx(test::as_bytes(test::field(1, std::uint64_t{8})))) ==
        RefusalCode::ModelUnreadable);
  // data_location = EXTERNAL sans emplacement.
  const std::string no_location = test::field(
      7, test::field(5, test::field(8, std::string_view("w")) + test::field(14, std::uint64_t{1})));
  CHECK(test::refusal(scan_onnx(test::as_bytes(no_location))) == RefusalCode::ModelUnreadable);
}

TEST_CASE("is_safe_location : un poids reste sous le dossier du modèle", "[prepare][onnx]") {
  CHECK(is_safe_location("poids/a.bin"));
  CHECK(is_safe_location("a.bin"));
  CHECK_FALSE(is_safe_location(""));
  CHECK_FALSE(is_safe_location("/etc/passwd"));
  CHECK_FALSE(is_safe_location("../voisin.bin"));
  CHECK_FALSE(is_safe_location("poids/../../voisin.bin"));
  CHECK_FALSE(is_safe_location("C:poids.bin"));
  CHECK_FALSE(is_safe_location("poids\\a.bin"));
}
