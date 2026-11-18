// Couper l'entrée d'un graphe ONNX en plusieurs images, au niveau du format protobuf, sans dépendance onnx :
// TemporalFix (app/play/temporal_fix.h) lit 7 images RGB empilées en une entrée 1 × 21 × H × W ; le lecteur
// garde ses images à part, dans 7 tenseurs 1 × 3 × H × W qui tournent d'une image à l'autre. Le modèle dérivé
// prend ces 7 tenseurs, et un Concat ajouté en tête du graphe refait l'entrée d'origine : ni copie côté
// lecteur, ni noyau de plus dans les backends. Les nœuds et les poids sont recopiés octet pour octet.
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

// Le modèle dont l'entrée input (statique, N × frames·C × H × W, dims fixées d'abord par with_fixed_dims) est
// remplacée par frames entrées N × C × H × W de même type, nommées frame_input_name(input, k) et concaténées
// sur l'axe 1, dans l'ordre, par un Concat ajouté en tête des nœuds. Le reste du graphe lit toujours input.
// Refus : ModelUnreadable (protobuf malformé, pas de graphe), FieldMissing (pas d'entrée input),
// FormatUnsupported (entrée non statique, de rang autre que 4, ou dont les canaux ne se divisent pas en
// frames), FieldOutOfRange (frames < 2).
std::expected<std::vector<std::byte>, Refusal> with_frame_inputs(std::span<const std::byte> model,
                                                                 std::string_view input, int frames);

}  // namespace ajp::prepare
