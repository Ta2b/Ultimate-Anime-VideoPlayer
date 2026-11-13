// Le SPAN FP16 natif jugé sur la carte au chargement (docs/span-fp8-cutlass.md, SPAN FP16 natif) : les noyaux
// CUTLASS sont aussi compilés pour les RTX 50 sans y avoir jamais tourné devant nous. Une image d'essai passe
// par le réseau chargé ; le coin haut-gauche de sa sortie se compare au SPAN f32 du processeur
// (span_fp8::forward), un calcul écrit à part, sur la même découpe. Sous kNativeCheckDb, le chargement est
// refusé et la chaîne reprend TensorRT FP16.
#pragma once

#include <expected>
#include <optional>

#include "core/types.h"
#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/span_fp8/package.h"

namespace ajp::gpu::cuda {

// La référence FP16 des noyaux colle au FP32 à 78,8 dB (docs/span-fp8-cutlass.md) ; le réseau FP8 se tient
// vers 50 dB. Des noyaux faux sur une architecture tombent bien plus bas.
inline constexpr double kNativeCheckDb = 50.0;
// Côté de la découpe d'entrée jugée : 64 pixels, un demi-plan de 32 dont un b5f48 garde 15 lignes exactes.
inline constexpr int kNativeCheckSide = 64;

// L'image d'essai au pixel (x, y) du plan c, avant l'arrondi FP16 : rampes, ondes et marches, dans [0, 1].
float native_check_pattern(int c, int y, int x);

// net est créé pour height × width, sa zone liée. Rend le PSNR (crête 1) de la sortie de la carte contre le
// calcul du processeur, nullopt si l'image est trop petite pour qu'une découpe en garde un pixel exact ;
// refus VendorCallFailed si un appel CUDA échoue, ou si la sortie s'écarte (sous kNativeCheckDb, ou non
// finie).
std::expected<std::optional<double>, Refusal> check_native(SpanFp8Net& net, const span_fp8::Package& package,
                                                           int height, int width);

}  // namespace ajp::gpu::cuda
