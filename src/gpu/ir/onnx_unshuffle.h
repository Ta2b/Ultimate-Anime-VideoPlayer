// Réécriture du dépliage de pixels en 6-D (Reshape, Transpose [0,1,3,5,2,4], Reshape : le
// pixel_unshuffle de PyTorch, à l'entrée des SPAN « unshuffle ») en 5-D, pour un lot de 1 : la
// forme cible du premier Reshape perd sa première dimension (un Slice ajouté), la permutation
// devient [0,2,4,1,3]. Les éléments restent dans le même ordre, et le Reshape suivant rend la
// même sortie.
//
// Pour Core ML par ONNX Runtime : le fournisseur CoreML refuse les tenseurs de rang supérieur à
// 5, et ces trois nœuds tournaient sur le CPU entre deux morceaux du réseau.
#pragma once

#include <cstddef>
#include <expected>
#include <span>
#include <vector>

#include "core/types.h"

namespace ajp::gpu::ir::onnx {

// Le fichier réécrit, ou vide quand il n'a aucun dépliage 6-D. À n'appeler que pour un lot de 1
// (la première dimension de la forme cible est retirée sans être lue). Un dépliage est réécrit
// quand le Transpose lit la sortie d'un Reshape qu'il est seul à lire, et que sa propre sortie
// ne va qu'à des Reshape, ni l'une ni l'autre n'étant une sortie du graphe. Le reste du fichier
// est recopié octet par octet, poids compris ; les value_info des deux tenseurs changés de rang
// sont retirées. Refus OnnxMalformed, OnnxUnsupported : ceux du décodeur.
std::expected<std::vector<std::byte>, Refusal> with_unshuffle_5d(std::span<const std::byte> model);

}  // namespace ajp::gpu::ir::onnx
