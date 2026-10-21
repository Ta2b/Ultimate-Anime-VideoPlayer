// Référence des conversions YUV ↔ RGB, du prétraitement des réseaux (RGB planaire float) et
// du post-traitement (quantification YUV, RGBA16 UNORM). Contrat : spec/color.h et
// spec/resample.h. Repris de animejanai-inference-rtx@118c272 src/kernels.cu:99-313
// (k_uv_h, k_v_f32, k_pre_combine, k_post_matrix, k_h_f32, k_uv_v_store, k_post444, k_pre444).
// Tests : tests/unit/kernels/color_test.cpp.
#pragma once

#include <cstdint>
#include <vector>

#include "kernels/image.h"
#include "kernels/spec/resample.h"

namespace ajp::kernels::ref {

// Description de couleur d'une image YUV, déjà résolue (effective_matrix, effective_range,
// effective_siting).
struct ColorSpec {
  Matrix matrix = Matrix::Bt709;
  Range range = Range::Limited;
  Siting siting = Siting::Left;
};

// Une image YUV conforme : dimensions dans la plage (paires en 4:2:0), plans de chroma aux
// dimensions du format, chaque code ≤ max_code et, en P010, ses 6 bits bas à zéro.
Result<void> check_yuv(const YuvImage& image);

// Ingestion : YUV → RGB planaire float, nominalement [0, 1] et jamais borné.
Result<RgbImage> yuv_to_rgb(const YuvImage& in, const ColorSpec& color, Filter chroma = kChainChromaFilter);

// Sortie : RGB planaire float → YUV quantifié du format, sans tramage par défaut. La chaîne passe
// kOutputSiting ; RIFE passe kRifeSiting ; l'export par puits CUDA, Bayer8 et le siting de son flux.
Result<YuvImage> rgb_to_yuv(const RgbImage& in, Format format, const ColorSpec& color,
                            Filter chroma = kChainChromaFilter, Dither dither = Dither::None);

// Sortie du ring : RGBA16 UNORM entrelacé, alpha opaque (unorm16).
Result<std::vector<std::uint16_t>> to_rgba16(const RgbImage& in);

}  // namespace ajp::kernels::ref
