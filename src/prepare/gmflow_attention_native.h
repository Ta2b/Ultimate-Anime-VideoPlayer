// Les couches d'attention du transformeur de GMFlow lues pour les noyaux natifs (docs/gmflow-natif.md, PR 2)
// : les projections q, k, v et la fusion par les noms de leurs MatMul («
// /flownet/transformer/layers.<b>/<couche>/ q_proj/MatMul », poids sans nom propre dans l'export), les
// LayerNorm par leurs initialiseurs (« gmfss.flownet.transformer.layers.<b>.<couche>.norm1.* »), et tout ce
// qui les relie dans le graphe : le facteur 1/√d, le Softmax, les fenêtres (Reshape et Transpose de
// split_feature), le décalage de Swin (Concat de deux Slice) et son masque, la cible de la cross-attention
// (l'autre moitié du lot à l'entrée du bloc), le résiduel. Le graphe est lu aux deux échelles (nœuds suffixés
// « _1 » au quart) : mêmes poids, fenêtres lues.
#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "core/types.h"
#include "prepare/gmflow_attention.h"

namespace kirari::prepare {

// Les douze couches et le plan des six blocs. La paire peut être en FP16 ou en FP32. Refus : ModelUnreadable
// (ONNX illisible), FieldMissing (un nom, une forme, un attribut ou un lien que les noyaux ne font pas, avec
// son nom).
std::expected<GmflowAttention, Refusal> gmflow_attention(std::span<const std::byte> pair);

// Le plan se tient-il pour les noyaux ? Des poids de 128 canaux, une self puis une cross par bloc, chaque
// opération lit des tenseurs déjà écrits et écrit un tenseur neuf, la self garde le résiduel et lit q, k, v
// dans le même tenseur, la cross lit k et v à l'entrée du bloc dans l'autre moitié du lot, le MLP ferme le
// bloc. Refus FieldMissing.
std::expected<void, Refusal> gmflow_attention_check(const GmflowAttention& attention);

}  // namespace kirari::prepare
