// Images planaires de la référence CPU. Aucune dépendance hors bibliothèque standard :
// kernels est une feuille du graphe.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kernels/error.h"  // IWYU pragma: export
#include "kernels/spec/color.h"
#include "kernels/spec/resample.h"

namespace ajp::kernels {

// Un plan : largeur × hauteur échantillons, ligne par ligne, sans pas de ligne.
// Des dimensions hors de [0, kMaxDimension] n'allouent rien : le plan garde ses dimensions,
// et chaque fonction de la référence le refuse (check_plane), au lieu d'une allocation géante.
template <class T>
struct Plane {
  int width = 0;
  int height = 0;
  std::vector<T> data;

  Plane() = default;
  Plane(int w, int h, T fill = T{}) : width(w), height(h), data(allocatable() ? area() : 0, fill) {}

  // Échantillons attendus, width × height, pour des dimensions positives.
  std::size_t area() const { return static_cast<std::size_t>(width) * static_cast<std::size_t>(height); }

  std::size_t index(int x, int y) const {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
  }
  const T& at(int x, int y) const { return data[index(x, y)]; }
  T& at(int x, int y) { return data[index(x, y)]; }

  friend bool operator==(const Plane&, const Plane&) = default;

 private:
  bool allocatable() const {
    return width >= 0 && width <= kMaxDimension && height >= 0 && height <= kMaxDimension;
  }
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
