// Q/DQ FP8 d'ONNX (modèles quantifiés par Model Optimizer, tools/fp8) : chaque paire
// QuantizeLinear → DequantizeLinear à zéro FP8 E4M3 et même échelle scalaire devient un seul nœud
// FakeQuantE4M3(x, échelle), qui arrondit x / échelle à la grille E4M3 puis remultiplie. Aucun
// tenseur FP8 n'entre dans l'IR : le noyau des matrices coopératives FP8 relit le motif à la
// fusion avec la Conv qui consomme la paire.
#pragma once

#include <cstddef>

#include "gpu/ir/onnx_model.h"

namespace ajp::gpu::ir::onnx {

// Remplace en place les paires fusionnables ; renvoie leur nombre. Les Q/DQ restants (échelles
// différentes, par axe, zéro absent ou d'un autre type) gardent leur nom d'opérateur, que
// l'import refuse (OperatorUnknown). Les zéros FP8 devenus inutiles quittent les initialiseurs.
std::size_t fuse_fp8_quant_pairs(Graph& graph);

}  // namespace ajp::gpu::ir::onnx
