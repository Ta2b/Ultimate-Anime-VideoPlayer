// Le MLP des couches de cross-attention du transformeur de GMFlow lu pour le noyau natif du backend TensorRT
// (docs/gmflow-natif.md, « MLP natif (PR 3) »). Les nœuds se trouvent par leurs noms dans la paire ONNX
// qu'exporte tools/gmfss/export_gmfss.py (« /flownet/transformer/layers.<i>/cross_attn_ffn/mlp/mlp.0/MatMul
// », … ; suffixe « _1 » au second appel du transformeur, le quart), l'échelle et le décalage de la LayerNorm
// par les leurs (« gmfss.flownet.transformer.layers.<i>.cross_attn_ffn.norm2.weight ») ; les poids des deux
// MatMul, que l'export renomme (« onnx::MatMul_… », un Identity au second appel), par l'entrée de leur nœud.
// Le reste du chemin (Concat, GELU, résiduel) se suit d'arête en arête : un autre graphe se trahit par un
// nom, une forme, un attribut, une constante ou un lecteur de trop.
#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "core/types.h"
#include "prepare/gmflow_mlp_plan.h"

namespace kirari::prepare {

// Les six couches et leurs douze emplois (huitième puis quart). Refus : ModelUnreadable (ONNX illisible),
// FieldMissing (un nœud, une forme, un attribut ou une constante que le noyau ne fait pas, avec son nom).
std::expected<GmflowMlp, Refusal> gmflow_mlp(std::span<const std::byte> pair);

// Le MLP se tient-il pour le noyau ? 128 canaux, une couche cachée par tranches de 64 (4096 au plus), des
// poids finis et représentables en FP16, des LayerNorm d'ε positif, chaque emploi sur une couche qui existe,
// une fois par (échelle, couche). Refus FieldMissing.
std::expected<void, Refusal> gmflow_mlp_check(const GmflowMlp& mlp);

}  // namespace kirari::prepare
