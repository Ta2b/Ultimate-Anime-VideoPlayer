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

// GridSample dont la grille sort d'un Transpose (0, 2, 3, 1), celui de warp dans RIFE : la grille
// est lue dans le tenseur [N, 2, H, W] d'avant (GridSampleAttrs::grid_nchw), sans copie.
int fold_grid_transposes(Graph& graph);

// Conv suivie de Add, Sub, Mul ou Div par une constante par canal : fondue dans
// les poids et le biais de la Conv.
int fuse_conv_bias(Graph& graph);

// Conv suivie d'un Add avec une activation de même forme : le résiduel entre dans l'épilogue.
int fuse_residuals(Graph& graph);

// Conv suivie de Relu, LeakyRelu, PRelu, Sigmoid, Tanh ou Clip : l'activation entre dans l'épilogue.
int fuse_activations(Graph& graph);

// Porte des blocs de SPAN, Mul(Add(x, r), Conv(x)) où la Conv garde la forme de x : le Mul et
// l'Add entrent dans l'épilogue de la Conv (Epilogue::gate). Après fuse_activations : l'activation
// de la Conv (Tanh dans SPAN) y est déjà. Deux dispatchs et deux tenseurs en moins par bloc.
int fuse_gates(Graph& graph);

// DepthToSpace lecteur unique d'une Conv (la fin de SPAN, une ConvTranspose réécrite) : la Conv
// écrit chaque valeur à sa place dans la sortie réarrangée (Epilogue::space). Un dispatch et une
// lecture et écriture de la plus grande image en moins.
int fuse_spaces(Graph& graph);

// Range en Nhwc16 les activations internes de rang 4 que tous leurs nœuds savent
// lire et écrire ainsi ; un nœud garde une seule disposition pour ses activations
// internes. Entrées et sorties du graphe restent en Nchw. Renvoie le nombre de tenseurs Nhwc16.
int assign_layout(Graph& graph);

// Retire les Cast entre float et float16 qu'un moteur aux noyaux typés fait de lui-même, quand
// chaque noyau lit et écrit chaque tenseur dans son propre type et calcule en FP32 (Vulkan).
// Vers float, la conversion est exacte : les lecteurs autres que les convolutions lisent la
// source. Vers float16, le producteur du résultat, s'il en est le seul lecteur et n'est pas une
// convolution, l'écrit lui-même en float16, avec le même arrondi. Un Cast entre types égaux
// disparaît. Hors de optimize() : l'interpréteur n'arrondit qu'aux Cast, ses sorties sur le
// graphe obtenu peuvent s'écarter d'un arrondi float16. Renvoie le nombre de Cast retirés.
int fold_float_casts(Graph& graph);

// ConvTranspose de pas s ≥ 2 sans groupe, dont la sortie fait s fois l'entrée : Conv de pas 1 aux
// s² × cout canaux (au plus max_channels) suivie d'un DepthToSpace, où se fond un DepthToSpace
// lecteur unique de la sortie. Pour un moteur dont seules les Conv passent par les matrices
// coopératives (OptimizeOptions). Renvoie le nombre de ConvTranspose réécrites.
int transposes_as_convs(Graph& graph, std::int64_t max_channels);

struct PassStats {
  int folded = 0;
  int dead = 0;
  int identities = 0;
  int pixel_shuffles = 0;
  int bias = 0;
  int residuals = 0;
  int activations = 0;
  int gates = 0;
  int spaces = 0;
  int grid_transposes = 0;
  int transposes = 0;
  int nhwc = 0;
};

struct OptimizeOptions {
  std::int64_t transpose_conv_channels = 0;  // > 0 : transposes_as_convs, avant les fusions
};

// Toutes les passes, dans l'ordre. Exige des formes connues (infer_shapes).
std::expected<PassStats, Refusal> optimize(Graph& graph, const OptimizeOptions& options = {});

}  // namespace ajp::gpu::ir
