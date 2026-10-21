#include "kernels/glsl.h"

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/charconv.h"
#include "kernels/spec.h"

using namespace ajp::kernels;

namespace {

const GlslConstant& find(const std::vector<GlslConstant>& constants, std::string_view name) {
  for (const GlslConstant& c : constants) {
    if (c.name == name) return c;
  }
  FAIL("constante absente : " << name);
  return constants.front();
}

// Relu comme le ferait un compilateur GLSL : directement en float, ou en double puis en float.
void check_reads_back(float value) {
  const auto literal = glsl_literal(value);
  REQUIRE(literal);
  const std::string& text = *literal;
  CHECK(text.find_first_of(".e") != std::string::npos);
  float direct = 0.0F;
  const auto [end, error] = ajp::core::from_chars(text.data(), text.data() + text.size(), direct);
  REQUIRE(error == std::errc{});
  REQUIRE(end == text.data() + text.size());
  CHECK(std::bit_cast<std::uint32_t>(direct) == std::bit_cast<std::uint32_t>(value));
  const auto via_double = static_cast<float>(std::strtod(text.c_str(), nullptr));
  CHECK(std::bit_cast<std::uint32_t>(via_double) == std::bit_cast<std::uint32_t>(value));
}

}  // namespace

TEST_CASE("GLSL : littéraux float relus au bit près", "[glsl]") {
  CHECK(glsl_literal(1.0F) == "1.0");
  CHECK(glsl_literal(65535.0F) == "65535.0");
  CHECK(glsl_literal(0.25F) == "0.25");
  CHECK(glsl_literal(-0.5F) == "-0.5");
  for (const float v :
       {0.2126F, 0.1593017578125F, 1e-4F, 1.0F / 255.0F, 1.0F / 65472.0F, 0.17883277F, 3e-7F}) {
    check_reads_back(v);
  }
}

TEST_CASE("GLSL : chaque constante se relit à l'identique", "[glsl]") {
  const std::vector<GlslConstant> constants = glsl_constants();
  REQUIRE(constants.size() > 100);
  std::set<std::string> names;
  for (const GlslConstant& c : constants) {
    CAPTURE(c.name);
    CHECK(names.insert(c.name).second);  // noms uniques
    CHECK(c.name.starts_with("AJP_"));
    CHECK(c.name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::string::npos);
    for (const float v : c.values) check_reads_back(v);
  }
}

TEST_CASE("GLSL : les nombres sont ceux de spec.h", "[glsl]") {
  const std::vector<GlslConstant> constants = glsl_constants();
  CHECK(find(constants, "AJP_KR_BT709").values.at(0) == coefficients(Matrix::Bt709).kr);
  CHECK(find(constants, "AJP_KB_BT2020").values.at(0) == coefficients(Matrix::Bt2020Ncl).kb);
  CHECK(find(constants, "AJP_YUV_SCALE_P010_LIMITED").values ==
        std::vector<float>{4096, 56064, 32768, 57344});
  CHECK(find(constants, "AJP_QUANT_P010").values == std::vector<float>{64, 1023});
  CHECK(find(constants, "AJP_PQ_M1").values.at(0) == pq::kM1);
  CHECK(find(constants, "AJP_BLEND_LAW_B").values.at(0) == kBlendLawB);
  CHECK(find(constants, "AJP_SCENE_NORM_NV12").values.at(0) == scene_norm(Format::Nv12));
  CHECK(find(constants, "AJP_CHROMA_SHIFT_UP_TOPLEFT").values == std::vector<float>{0.25F, 0.25F});
  const GlslConstant& held = find(constants, "AJP_HELD_THRESHOLD");
  CHECK(held.type == GlslType::Uint);
  CHECK(held.integer == kHeldThreshold);
  CHECK(find(constants, "AJP_RIFE_ALIGN").integer == kRifeAlign);
  CHECK(find(constants, "AJP_RIFE_PAD_LUMA_P010").integer == 4096);
  // mat3 colonne par colonne : la deuxième valeur est la ligne 1 de la colonne 0.
  const GlslConstant& m = find(constants, "AJP_BT709_TO_BT2020");
  REQUIRE(m.values.size() == 9);
  CHECK(m.values[1] == static_cast<float>(kBt709ToBt2020.at(1, 0)));
  CHECK(m.values[3] == static_cast<float>(kBt709ToBt2020.at(0, 1)));
}

TEST_CASE("GLSL : déclarations et en-tête", "[glsl]") {
  CHECK(glsl_declaration({.name = "AJP_X", .type = GlslType::Uint, .integer = 70}) ==
        "const uint AJP_X = 70u;");
  CHECK(glsl_declaration({.name = "AJP_Y", .type = GlslType::Int, .integer = -3}) == "const int AJP_Y = -3;");
  CHECK(glsl_declaration({.name = "AJP_Z", .type = GlslType::Vec2, .values = {0.5F, 1.0F}}) ==
        "const vec2 AJP_Z = vec2(0.5, 1.0);");
  CHECK(glsl_declaration({.name = "AJP_W", .type = GlslType::Float, .values = {0.2126F}}) ==
        "const float AJP_W = 0.2126;");

  const auto generated = glsl_spec_header();
  REQUIRE(generated);
  const std::string& header = *generated;
  CHECK(header.starts_with("// Généré depuis kernels/spec.h"));
  CHECK(header.find("#ifndef AJP_KERNELS_SPEC_GLSL\n#define AJP_KERNELS_SPEC_GLSL\n") != std::string::npos);
  CHECK(header.find(std::format("#define AJP_KERNELS_SPEC_VERSION {}\n", kSpecVersion)) != std::string::npos);
  CHECK(header.ends_with("#endif  // AJP_KERNELS_SPEC_GLSL\n"));
  for (const GlslConstant& c : glsl_constants()) {
    const auto declaration = glsl_declaration(c);
    REQUIRE(declaration);
    const std::string line = *declaration + "\n";
    CAPTURE(line);
    const std::size_t at = header.find(line);
    REQUIRE(at != std::string::npos);
    CHECK(header.find(line, at + 1) == std::string::npos);
  }
  CHECK(header.find("const float AJP_PQ_M2 = 78.84375;") != std::string::npos);
  CHECK(header.find("const vec4 AJP_YUV_SCALE_NV12_FULL = vec4(0.0, 255.0, 128.0, 255.0);") !=
        std::string::npos);
}

TEST_CASE("GLSL : une constante mal formée est refusée, jamais écrite", "[glsl]") {
  const auto refused = [](const GlslConstant& c) {
    const auto line = glsl_declaration(c);
    REQUIRE_FALSE(line);
    CHECK(line.error().code == ErrorCode::InvalidConstant);
  };
  refused({.name = "AJP_X", .type = GlslType::Float});                         // aucune valeur
  refused({.name = "AJP_X", .type = GlslType::Vec4, .values = {1.0F, 2.0F}});  // deux sur quatre
  refused({.name = "AJP_X", .type = GlslType::Int, .values = {1.0F}});         // entier porté par integer
  refused({.name = "AJP_X", .type = GlslType::Mat3, .values = std::vector(8, 0.0F)});  // 8 sur 9
  refused({.name = "AJP_X", .type = GlslType::Float, .values = {std::numeric_limits<float>::infinity()}});
  refused(
      {.name = "AJP_X", .type = GlslType::Vec2, .values = {0.0F, std::numeric_limits<float>::quiet_NaN()}});
  refused({.name = "AJP_X", .type = GlslType::Uint, .integer = -1});
  refused({.name = "AJP_X", .type = GlslType::Int, .integer = std::int64_t{1} << 40});
  refused({.name = "kr", .type = GlslType::Float, .values = {0.5F}});
  refused({.name = "AJP_", .type = GlslType::Float, .values = {0.5F}});
  refused({.name = "AJP_A B", .type = GlslType::Float, .values = {0.5F}});
  CHECK(glsl_literal(std::numeric_limits<float>::infinity()).error().code == ErrorCode::InvalidConstant);
  CHECK(glsl_literal(-std::numeric_limits<float>::infinity()).error().code == ErrorCode::InvalidConstant);
  CHECK(name(ErrorCode::InvalidConstant) == "InvalidConstant");
  // Les bornes d'un uint et d'un int passent.
  CHECK(glsl_declaration({.name = "AJP_U", .type = GlslType::Uint, .integer = 4294967295}) ==
        "const uint AJP_U = 4294967295u;");
  CHECK(glsl_declaration({.name = "AJP_I", .type = GlslType::Int, .integer = -2147483647}) ==
        "const int AJP_I = -2147483647;");
  // −2³¹ s'écrirait « -2147483648 » : le moins s'applique à 2147483648, trop grand pour un int.
  refused({.name = "AJP_I", .type = GlslType::Int, .integer = -2147483648});
}
