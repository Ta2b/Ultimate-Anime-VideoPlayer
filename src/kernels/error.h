// Refus du module kernels. Aucune dépendance hors bibliothèque standard : kernels est une
// feuille du graphe, gpu traduit Error en Refusal.
#pragma once

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>

namespace ajp::kernels {

// Un code s'ajoute à la fin, avec son nom dans name().
enum class ErrorCode : std::uint8_t {
  DimensionOutOfRange,  // largeur ou hauteur hors de sa plage
  OddDimension,         // image 4:2:0 ou de la chaîne de dimensions impaires
  SizeMismatch,         // plans ou images qui devraient avoir la même taille
  SampleOutOfRange,     // échantillon hors de son conteneur (P010 : bits bas non nuls)
  FilterDirection,      // Area ne sert qu'à réduire
  ParameterOutOfRange,  // décalage, instant t ou luminance hors de sa plage
  InvalidConstant,      // constante GLSL mal formée : nom, nombre de valeurs, valeur non finie
};

constexpr std::string_view name(ErrorCode code) {
  switch (code) {
    case ErrorCode::DimensionOutOfRange:
      return "DimensionOutOfRange";
    case ErrorCode::OddDimension:
      return "OddDimension";
    case ErrorCode::SizeMismatch:
      return "SizeMismatch";
    case ErrorCode::SampleOutOfRange:
      return "SampleOutOfRange";
    case ErrorCode::FilterDirection:
      return "FilterDirection";
    case ErrorCode::ParameterOutOfRange:
      return "ParameterOutOfRange";
    case ErrorCode::InvalidConstant:
      return "InvalidConstant";
  }
  return "Unknown";
}

struct Error {
  ErrorCode code = ErrorCode::DimensionOutOfRange;
  std::string detail;  // quoi et pourquoi, par exemple « largeur 0 hors de [2, 16384] »
};

template <class T>
using Result = std::expected<T, Error>;

inline std::unexpected<Error> fail(ErrorCode code, std::string detail) {
  return std::unexpected(Error{.code = code, .detail = std::move(detail)});
}

}  // namespace ajp::kernels
