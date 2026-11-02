// Lecture des points d'une image décodée en mémoire hôte (Nv12, Yuv420p, P010, Yuv420p10) : les images fixes
// du SPAN FP8 (stills.cpp) et les mesures de la pré-analyse des plans (shot_probe.cpp). En-tête interne.
#pragma once

#include <cstddef>
#include <optional>
#include <span>

#include "core/frame.h"
#include "core/types.h"

namespace ajp::media::yuv {

inline bool is_hdr(const ColorInfo& color) {
  return color.transfer == Transfer::Pq || color.transfer == Transfer::Hlg ||
         color.matrix == ColorMatrix::Bt2020Ncl || color.primaries == Primaries::Bt2020 ||
         color.hdr.has_value();
}

struct Layout {
  int bits = 8;
  bool semi_planar = false;  // chroma entrelacée (Nv12, P010)
  bool high = false;         // échantillons de 16 bits
  int shift = 0;             // décalage vers le bas des bits utiles (P010 : 6)
};

inline std::optional<Layout> layout_of(PixelFormat format) {
  switch (format) {
    case PixelFormat::Nv12:
      return Layout{.bits = 8, .semi_planar = true};
    case PixelFormat::Yuv420p:
      return Layout{.bits = 8};
    case PixelFormat::P010:
      return Layout{.bits = 10, .semi_planar = true, .high = true, .shift = 6};
    case PixelFormat::Yuv420p10:
      return Layout{.bits = 10, .high = true};
    default:
      return std::nullopt;
  }
}

inline int sample(std::span<const std::byte> plane, std::size_t stride, int x, int y, const Layout& layout,
                  int channel = 0, int channels = 1) {
  const std::size_t unit = layout.high ? 2U : 1U;
  const std::size_t at =
      static_cast<std::size_t>(y) * stride +
      (static_cast<std::size_t>(x) * static_cast<std::size_t>(channels) + static_cast<std::size_t>(channel)) *
          unit;
  if (!layout.high) return static_cast<int>(plane[at]);
  const unsigned raw = static_cast<unsigned>(plane[at]) | (static_cast<unsigned>(plane[at + 1]) << 8U);
  return static_cast<int>(raw >> static_cast<unsigned>(layout.shift));
}

}  // namespace ajp::media::yuv
