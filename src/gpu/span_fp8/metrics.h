// Les mesures de qualité de la dérivation du SPAN FP8 : celles de tools/span/quantize_span.py (psnr et ssim),
// pour que le reçu d'une dérivation native et celui de l'outil Python se lisent de la même façon.
#pragma once

#include <span>

namespace ajp::gpu::span_fp8 {

// PSNR en dB (pic 1) de deux images planaires 3×H×W de valeurs dans [0, 1], plafonné à 99 dB.
double psnr(std::span<const float> a, std::span<const float> b);

// SSIM de la luminance BT.709 (0,2126 R + 0,7152 G + 0,0722 B) de deux images planaires 3×H×W de valeurs
// dans [0, 1] : fenêtre gaussienne de 11 pixels, σ = 1,5, constantes de Wang et al. (2004), moyenne sur la
// zone où la fenêtre tient entièrement. 0 si l'image est plus petite que la fenêtre.
double ssim(int width, int height, std::span<const float> a, std::span<const float> b);

}  // namespace ajp::gpu::span_fp8
