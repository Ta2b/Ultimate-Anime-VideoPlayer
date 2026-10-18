// Générateur de l'en-tête GLSL des constantes de spec.h : les shaders du moteur Vulkan
// reprennent exactement les mêmes nombres que la référence et que les ports CUDA.
// Le programme ajp_kernels_glsl l'écrit dans le dossier de construction.
// Tests : tests/unit/kernels/glsl_test.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "kernels/error.h"

namespace ajp::kernels {

enum class GlslType : std::uint8_t { Float, Int, Uint, Vec2, Vec4, Mat3 };

// Nombre de valeurs float d'un type : 1, 2, 4 ou 9 ; 0 pour les entiers, portés par integer.
constexpr std::size_t glsl_float_count(GlslType type) {
  switch (type) {
    case GlslType::Float:
      return 1;
    case GlslType::Vec2:
      return 2;
    case GlslType::Vec4:
      return 4;
    case GlslType::Mat3:
      return 9;
    case GlslType::Int:
    case GlslType::Uint:
      return 0;
  }
  return 0;
}

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
// même float, toujours avec un point ou un exposant. Refus : infini ou NaN, sans littéral GLSL.
Result<std::string> glsl_literal(float value);

// « const <type> <nom> = <valeur>; ». Refus (InvalidConstant) : nom hors de AJP_[A-Z0-9_]+,
// nombre de valeurs différent de glsl_float_count, valeur non finie, Uint hors de
// [0, 2³² − 1], Int hors de [−(2³¹ − 1), 2³¹ − 1].
Result<std::string> glsl_declaration(const GlslConstant& constant);

// L'en-tête complet, avec sa garde et AJP_KERNELS_SPEC_VERSION. Refus : la première constante
// refusée par glsl_declaration.
Result<std::string> glsl_spec_header();

}  // namespace ajp::kernels
