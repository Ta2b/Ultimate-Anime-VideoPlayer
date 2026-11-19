// Q/DQ FP8 et INT8 d'ONNX (modèles quantifiés par Model Optimizer, tools/fp8, tools/span/quantize_span.py) :
// chaque paire QuantizeLinear → DequantizeLinear à zéro FP8 E4M3 (ou INT8) nul et même échelle scalaire
// devient un seul nœud FakeQuantE4M3(x, échelle) (ou FakeQuantInt8), qui arrondit x / échelle à la grille
// puis remultiplie ; par canal, sur des poids constants, un initialiseur arrondi ; un DequantizeLinear INT8
// seul de poids entiers, l'initialiseur q × échelle. Aucun
// tenseur FP8 n'entre dans l'IR : le noyau des matrices coopératives FP8 relit le motif à la
// fusion avec la Conv qui consomme la paire.
#pragma once

#include <cstddef>

#include "gpu/ir/onnx_model.h"

namespace kirari::gpu::ir::onnx {

// Remplace en place les paires fusionnables ; renvoie leur nombre. Les Q/DQ restants (échelles
// différentes, par axe, zéro absent ou d'un autre type) gardent leur nom d'opérateur, que
// l'import refuse (OperatorUnknown). Les zéros FP8 devenus inutiles quittent les initialiseurs.
std::size_t fuse_fp8_quant_pairs(Graph& graph);

}  // namespace kirari::gpu::ir::onnx
