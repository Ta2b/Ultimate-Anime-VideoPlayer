// Un paquet « .span8 » de la forme du vrai, rempli de valeurs régulières : pour les tests sans modèle (C02-09
// : il vivait dans le module du produit, gpu/span_fp8/package.h). En-tête seul, inclus par chemin relatif
// depuis tests/unit/<module>/ et tests/fuzz/.
#pragma once

#include <cstddef>
#include <cstdint>

#include "gpu/span_fp8/package.h"

namespace ajp::gpu::span_fp8::testing {

// Poids en FP16 tous égaux (0x211E : 0,01 en demi-précision ; la valeur exacte n'importe pas aux tests,
// seulement qu'elle soit finie) quand fill est positif, nuls sinon ; biais à fill.
inline Conv filled(std::size_t weights, std::size_t bias, float fill) {
  Conv conv;
  conv.weights.assign(weights, fill > 0.0F ? std::uint16_t{0x211E} : std::uint16_t{0});
  conv.bias.assign(bias, fill);
  return conv;
}

inline Package synthetic(float fill = 0.01F, int blocks = kBlocks) {
  constexpr std::size_t channels = kChannels;
  Package package;
  package.blocks.resize(static_cast<std::size_t>(blocks));
  package.scales.fill(0.05F);
  package.first = filled(channels * 3 * 3 * kUnshuffledPadded, channels, fill);
  for (Block& block : package.blocks) {
    for (Conv& conv : block.conv) conv = filled(channels * 3 * 3 * channels, channels, fill);
    block.attention = filled(channels * channels, channels, fill);
  }
  package.concat = filled(channels * kConcatChannels, channels, fill);
  package.pixel = filled(std::size_t{kPixelChannels} * 3 * 3 * channels, kPixelChannels, fill);
  return package;
}

}  // namespace ajp::gpu::span_fp8::testing
