// Générateur de l'en-tête GLSL des constantes de spec.h : les shaders du moteur Vulkan
// reprennent exactement les mêmes nombres que la référence et que les ports CUDA.
// Le programme ajp_kernels_glsl l'écrit dans le dossier de construction.
// Tests : tests/unit/kernels/glsl_test.cpp.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ajp::kernels {

enum class GlslType : std::uint8_t { Float, Int, Uint, Vec2, Vec4, Mat3 };

// Une constante de la spécification telle qu'elle part en GLSL.
struct GlslConstant {
  std::string name;  // AJP_<SUJET>_<NOM>
  GlslType type = GlslType::Float;
  std::vector<float> values;  // Float, Vec2, Vec4 ; Mat3 colonne par colonne, ordre de mat3()
  std::int64_t integer = 0;   // Int, Uint
  std::string comment;
};

inline constexpr std::string_view kGlslHeaderName = "ajp_kernels_spec.glsl";

// Toutes les constantes, dans l'ordre de l'en-tête.
std::vector<GlslConstant> glsl_constants();

// Littéral float qui se relit au bit près : la plus courte écriture décimale qui redonne le
// même float, toujours avec un point ou un exposant.
std::string glsl_literal(float value);

// « const <type> <nom> = <valeur>; »
std::string glsl_declaration(const GlslConstant& constant);

// L'en-tête complet, avec sa garde et AJP_KERNELS_SPEC_VERSION.
std::string glsl_spec_header();

}  // namespace ajp::kernels
