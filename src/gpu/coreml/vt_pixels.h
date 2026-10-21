// Les images de l'interpolation d'Apple (coreml_vt.h) : les plans RGB d'un tenseur de la chaîne
// (NCHW, F16 ou F32, de 0 à 1) écrits dans un format de pixels de VideoToolbox, et relus. Portable :
// rien de propre à Apple, les tests le vérifient partout.
//
// RGBA demi-flottant d'abord : sans perte ni matrice. En YCbCr 4:2:0, la matrice BT.709 de RIFE
// (kernels/spec/rife.h), la chroma moyennée sur chaque bloc 2 × 2, donc centrée, et relue en
// bilinéaire au même centre.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "gpu/tensor.h"

namespace ajp::gpu::coreml_rt::vt {

enum class Pixels : std::uint8_t {
  RgbaHalf,       // kCVPixelFormatType_64RGBAHalf, « RGhA »
  Bgra8,          // kCVPixelFormatType_32BGRA
  Yuv420Full10,   // kCVPixelFormatType_420YpCbCr10BiPlanarFullRange, « xf20 »
  Yuv420Video10,  // kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange, « x420 »
  Yuv420Full8,    // kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, « 420f »
  Yuv420Video8,   // kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, « 420v »
};

// Le code de format de Core Video (OSType), et son nom de quatre lettres.
std::uint32_t four_cc(Pixels format);
std::string_view name(Pixels format);
// Le premier de nos formats parmi ceux qu'offre l'interpolateur, dans l'ordre de préférence.
std::optional<Pixels> preferred(std::span<const std::uint32_t> offered);
bool is_yuv(Pixels format);

// Les plans d'un tampon de pixels verrouillé : un pour RGB, luma et chroma entrelacée pour YCbCr.
struct Planes {
  std::array<std::byte*, 2> data{};
  std::array<std::size_t, 2> stride{};  // octets par rangée
};

// Les rangées [first, last) de l'image ; first pair (la chroma 4:2:0 va par paire de rangées).
struct Rows {
  int first = 0;
  int last = 0;
};

// Écrit les rangées rows des 3 plans width × height de rgb (type F16 ou F32) dans out.
void pack(Pixels format, const std::byte* rgb, TensorType type, int width, int height, const Planes& out,
          Rows rows);
// Relit les rangées rows de in dans les 3 plans de rgb.
void unpack(Pixels format, const Planes& in, std::byte* rgb, TensorType type, int width, int height,
            Rows rows);

}  // namespace ajp::gpu::coreml_rt::vt
