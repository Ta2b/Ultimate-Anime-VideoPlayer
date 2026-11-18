// Couper l'entrée d'un graphe ONNX en plusieurs images, au niveau du format protobuf, sans dépendance onnx :
// TemporalFix (app/play/temporal_fix.h) lit 7 images RGB empilées en une entrée 1 × 21 × H × W ; le lecteur
// garde ses images à part, dans 7 tenseurs 1 × 3 × H × W qui tournent d'une image à l'autre. Le modèle dérivé
// prend ces 7 tenseurs, et un Concat ajouté en tête du graphe refait l'entrée d'origine : ni copie côté
// lecteur, ni noyau de plus dans les backends. Les poids sont recopiés octet pour octet.
//
// Les dimensions H et W restent libres : figées (with_fixed_dims), les grilles que TemporalFix calcule depuis
// la forme (ConstantOfShape, CumSum, Expand) deviennent des constantes que TensorRT range comme des poids,
// 6,5 Go en 1080p, et le moteur passe de 28 ms à 295 ms par image (RTX 4070 Ti).
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::prepare {

// Le nom de l'entrée k (0 ≤ k < frames) du modèle coupé : « <input>_<k> ».
std::string frame_input_name(std::string_view input, int k);

// La coupe : l'entrée input (N × frames·C × H × W ; H et W fixes ou libres) remplacée par frames entrées
// N × C × H × W de même type et de mêmes dimensions, nommées frame_input_name(input, k) et concaténées sur
// l'axe 1, dans l'ordre, par un Concat ajouté en tête des nœuds.
//
// half : le réseau tourne à demi-taille, sur les images moyennées 2 × 2 (AveragePool) ; sa sortie, moins
// l'image center moyennée de même, est agrandie (Resize bilinéaire) et ajoutée à l'image center à pleine
// taille : la correction basse fréquence du réseau, le détail de l'image gardé. Coût du réseau : un quart.
// Le graphe doit alors rendre output, N × C × H × W ; ses value_info, aux formes d'avant, sont retirées.
struct FrameSplit {
  std::string_view input = "input";
  int frames = 7;
  bool half = false;
  int center = 3;
  std::string_view output = "output";
};

// Refus : ModelUnreadable (protobuf malformé, pas de graphe), FieldMissing (pas d'entrée input, pas de
// sortie output en demi-taille), FormatUnsupported (entrée de rang autre que 4, aux canaux libres ou qui ne
// se divisent pas en frames, nom déjà pris), FieldOutOfRange (frames < 2, center hors des images).
std::expected<std::vector<std::byte>, Refusal> with_frame_inputs(std::span<const std::byte> model,
                                                                 const FrameSplit& split);
inline std::expected<std::vector<std::byte>, Refusal> with_frame_inputs(std::span<const std::byte> model,
                                                                        std::string_view input, int frames) {
  return with_frame_inputs(model, FrameSplit{.input = input, .frames = frames});
}

}  // namespace ajp::prepare
