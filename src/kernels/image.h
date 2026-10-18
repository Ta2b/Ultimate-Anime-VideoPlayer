// Images planaires de la référence CPU, et ses refus. Aucune dépendance hors bibliothèque
// standard : kernels est une feuille du graphe, gpu traduit Error en Refusal.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernels/spec/color.h"

namespace ajp::kernels {

enum class ErrorCode : std::uint8_t {
  DimensionOutOfRange,  // largeur ou hauteur hors de sa plage
  OddDimension,         // image 4:2:0 ou de la chaîne de dimensions impaires
  SizeMismatch,         // plans ou images qui devraient avoir la même taille
  SampleOutOfRange,     // échantillon hors de son conteneur (P010 : bits bas non nuls)
  FilterDirection,      // Area ne sert qu'à réduire
  ParameterOutOfRange,  // décalage, instant t ou luminance hors de sa plage
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

// Un plan : largeur × hauteur échantillons, ligne par ligne, sans pas de ligne.
template <class T>
struct Plane {
  int width = 0;
  int height = 0;
  std::vector<T> data;

  Plane() = default;
  Plane(int w, int h, T fill = T{})
      : width(w), height(h), data(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), fill) {}

  std::size_t index(int x, int y) const {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
  }
  const T& at(int x, int y) const { return data[index(x, y)]; }
  T& at(int x, int y) { return data[index(x, y)]; }

  friend bool operator==(const Plane&, const Plane&) = default;
};

using PlaneF = Plane<float>;
using PlaneU16 = Plane<std::uint16_t>;

// RGB planaire float, l'entrée et la sortie des réseaux (NCHW sans lot).
struct RgbImage {
  std::array<PlaneF, 3> planes;  // R, G, B

  RgbImage() = default;
  RgbImage(int w, int h, float fill = 0.0F)
      : planes{PlaneF(w, h, fill), PlaneF(w, h, fill), PlaneF(w, h, fill)} {}

  int width() const { return planes[0].width; }
  int height() const { return planes[0].height; }
  Rgb at(int x, int y) const {
    return {.r = planes[0].at(x, y), .g = planes[1].at(x, y), .b = planes[2].at(x, y)};
  }

  friend bool operator==(const RgbImage&, const RgbImage&) = default;
};

// YUV en codes du conteneur (P010 garde son décalage de 6 bits), chroma toujours planaire :
// l'entrelacement de NV12 et P0xx est une affaire de mémoire, pas de valeurs.
struct YuvImage {
  Format format = Format::Nv12;
  PlaneU16 y;
  PlaneU16 u;  // Cb
  PlaneU16 v;  // Cr

  YuvImage() = default;
  // Plans aux dimensions du format, remplis d'une couleur ; width et height pairs en 4:2:0.
  YuvImage(Format f, int w, int h, std::uint16_t luma, std::uint16_t cb, std::uint16_t cr)
      : format(f),
        y(w, h, luma),
        u(info(f).subsampled ? w / 2 : w, info(f).subsampled ? h / 2 : h, cb),
        v(info(f).subsampled ? w / 2 : w, info(f).subsampled ? h / 2 : h, cr) {}

  int width() const { return y.width; }
  int height() const { return y.height; }

  friend bool operator==(const YuvImage&, const YuvImage&) = default;
};

}  // namespace ajp::kernels
