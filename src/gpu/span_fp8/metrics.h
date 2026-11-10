// Les mesures de qualité de la dérivation du SPAN FP8 : celles de tools/span/quantize_span.py (psnr et ssim),
// pour que le reçu d'une dérivation native et celui de l'outil Python se lisent de la même façon.
#pragma once

#include <span>

namespace ajp::gpu::span_fp8 {

// Un échantillon qui n'est pas fini (NaN, infini) dans l'une des deux images donne à chaque mesure sa pire
// note : 0 dB, un SSIM de 0, une erreur de kNotFiniteError (l'écart entier), quel que soit le quantile
// demandé. Sans cela une image avec des NaN comptait 99 dB et passait la porte (A02-01).
inline constexpr double kNotFiniteError = 1.0;

// PSNR en dB (pic 1) de deux images planaires 3×H×W de valeurs dans [0, 1], plafonné à 99 dB.
double psnr(std::span<const float> a, std::span<const float> b);

// SSIM de la luminance BT.709 (0,2126 R + 0,7152 G + 0,0722 B) de deux images planaires 3×H×W de valeurs
// dans [0, 1] : fenêtre gaussienne de 11 pixels, σ = 1,5, constantes de Wang et al. (2004), moyenne sur la
// zone où la fenêtre tient entièrement. 0 si l'image est plus petite que la fenêtre.
double ssim(int width, int height, std::span<const float> a, std::span<const float> b);

// Étape 3.7a. Le quantile q (0 à 1) de |a − b| sur tous les échantillons ; 0 si les tailles diffèrent ou
// sont nulles. Quantile par le rang ⌊q·(n − 1)⌋, sans interpolation.
double abs_error_quantile(std::span<const float> a, std::span<const float> b, double q);
// Le même, sur les seuls pixels sombres de la référence a (luminance BT.709 < dark) : c'est là qu'un palier
// de 8 bits se voit. 0 sans pixel sombre.
inline constexpr double kDarkLuma = 32.0 / 255.0;
double dark_abs_error_quantile(int width, int height, std::span<const float> a, std::span<const float> b,
                               double q, double dark = kDarkLuma);

}  // namespace ajp::gpu::span_fp8
