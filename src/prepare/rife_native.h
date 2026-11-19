// Les poids d'un morceau de RIFE fusionné (rife_split.h : l'encodeur ou un bloc) lus pour les noyaux natifs
// du backend TensorRT (--rife-native, docs/rife-natif.md). Un morceau est une chaîne de convolutions : depuis
// son entrée unique jusqu'à sa sortie unique, des Conv 3 × 3 (pas 1 ou 2, bord 1) suivies ou non d'une
// activation (LeakyRelu, PRelu, Relu), d'une échelle par canal (Mul) et d'une addition résiduelle de l'entrée
// de la convolution (Add), puis une ConvTranspose 4 × 4 de pas 2 à la fin. C'est l'encodeur de RIFE 4.2x
// (cnn0 à cnn3) et son IFBlock (conv0, convblock de ResConv à bêta, lastconv). Tout le reste est refusé : le
// morceau reste alors aux moteurs TensorRT.
#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <vector>

#include "core/types.h"

namespace kirari::prepare {

struct RifeNativeLayer {
  bool transposed = false;  // ConvTranspose 4 × 4 de pas 2, bord 1 ; sinon Conv 3 × 3, bord 1
  int cin = 0;
  int cout = 0;
  int kernel = 3;  // 3, ou 4 pour la transposée
  int stride = 1;  // 1 ou 2 (2 pour la transposée)
  // [cout][kernel × kernel][cin], en flottants (les FP16 du modèle, exacts). Transposée : le poids ONNX
  // [cin][cout][ky][kx] rangé de même, [cout][ky × 4 + kx][cin].
  std::vector<float> weights;
  std::vector<float> bias;   // cout valeurs (zéros si la convolution n'en a pas)
  std::vector<float> slope;  // pente négative de l'activation par canal ; vide : pas d'activation
  // Résiduel : sortie = act(scale · (conv + biais) + entrée de la convolution) ; vide : pas de résiduel.
  std::vector<float> scale;
};

// Les couches du morceau, dans l'ordre. Refus : ModelUnreadable (ONNX illisible), FieldMissing (structure
// autre qu'une chaîne de convolutions reconnue, avec le nœud qui l'arrête).
std::expected<std::vector<RifeNativeLayer>, Refusal> rife_native_layers(std::span<const std::byte> piece);

// Les couches forment-elles ce que les noyaux savent faire ? Une ou plusieurs convolutions 3 × 3 (la
// première à in_channels canaux), puis une seule transposée à out_channels canaux, sans activation ni
// résiduel ; chaque couche lit les canaux de la précédente, un résiduel garde ses canaux. Refus FieldMissing.
std::expected<void, Refusal> rife_native_check(std::span<const RifeNativeLayer> layers, int in_channels,
                                               int out_channels);

}  // namespace kirari::prepare
