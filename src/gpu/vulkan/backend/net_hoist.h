// Resize qui réduit remonté devant les copies qui le précèdent (passe propre au moteur Vulkan).
//
// Un Resize traite chaque (lot, canal) à part. Réduire une image avant de la recopier donne le
// même résultat, bit à bit, avec des copies de la taille réduite :
//   Resize(Concat(a, b, ...))              → Concat(Resize(a), Resize(b), ...)
//   Resize(Reshape(y))     [y sortie d'un Split de lot > 1]  → Reshape(Resize(y))
//   Resize(o_0), Resize(o_1), ... [toutes les sorties d'un Split de lot > 1]
//                                          → Split(Resize(z))
// Dans RIFE, chaque bloc concatène des cartes pleine taille (images, flots, masque) que seul un
// Resize lit ensuite : les copies de Concat et de Split, la moitié du temps des dispatchs de copie,
// disparaissent ; chaque entrée est réduite là où elle se trouve (entrée du graphe, vue d'un autre
// tenseur, sortie d'un noyau), sans être déplacée.
#pragma once

#include "gpu/ir/ir.h"

namespace ajp::gpu::vk::net {

// Réécrit le graphe (formes connues, tenseurs en Nchw) ; renvoie le nombre de réécritures. Les
// tenseurs orphelins ne sont pas retirés : eliminate_dead_code ensuite.
int hoist_downscales(ir::Graph& graph);

}  // namespace ajp::gpu::vk::net
