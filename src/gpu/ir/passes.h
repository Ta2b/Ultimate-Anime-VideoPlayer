// Passes sur l'IR, après l'inférence des formes. Chacune préserve les sorties
// de l'interpréteur (à l'arrondi FP32 près) et l'ordre topologique des nœuds.
#pragma once

#include <expected>

#include "core/types.h"
#include "gpu/ir/ir.h"

namespace ajp::gpu::ir {

// Pliage des constantes : un nœud dont toutes les entrées sont constantes,
// ou un Shape d'une forme connue, devient une constante. Renvoie le nombre de nœuds pliés.
std::expected<int, Refusal> fold_constants(Graph& graph);

// Retire les nœuds dont aucune sortie n'atteint une sortie du graphe, puis les tenseurs orphelins.
int eliminate_dead_code(Graph& graph);

// Retire Identity et les nœuds sans effet : Cast vers le même type (hors F16,
// où il arrondit), Reshape, Unsqueeze ou Squeeze qui gardent la forme, Transpose
// identité, Slice entier, Pad nul, Resize d'échelle 1, Concat d'une entrée, Split d'une sortie.
int eliminate_identities(Graph& graph);

// Reconnaît Reshape, Transpose, Reshape de rang 6 : pixel_shuffle et pixel_unshuffle
// de PyTorch (CRD), ou leurs équivalents ONNX (DCR), en DepthToSpace et SpaceToDepth.
int fuse_pixel_shuffles(Graph& graph);

// Conv suivie de Add, Sub, Mul ou Div par une constante par canal : fondue dans
// les poids et le biais de la Conv.
int fuse_conv_bias(Graph& graph);

// Conv suivie d'un Add avec une activation de même forme : le résiduel entre dans l'épilogue.
int fuse_residuals(Graph& graph);

// Conv suivie de Relu, LeakyRelu, PRelu, Sigmoid, Tanh ou Clip : l'activation entre dans l'épilogue.
int fuse_activations(Graph& graph);

// Range en Nhwc16 les activations internes de rang 4 que tous leurs nœuds savent
// lire et écrire ainsi ; un nœud garde une seule disposition pour ses activations
// internes. Entrées et sorties du graphe restent en Nchw. Renvoie le nombre de tenseurs Nhwc16.
int assign_layout(Graph& graph);

struct PassStats {
  int folded = 0;
  int dead = 0;
  int identities = 0;
  int pixel_shuffles = 0;
  int bias = 0;
  int residuals = 0;
  int activations = 0;
  int nhwc = 0;
};

// Toutes les passes, dans l'ordre. Exige des formes connues (infer_shapes).
std::expected<PassStats, Refusal> optimize(Graph& graph);

}  // namespace ajp::gpu::ir
