// Représentation interne (IR) d'un réseau : tenseurs nommés et opérations.
//
// Base commune du moteur Vulkan maison et du futur générateur Metal. Le jeu
// d'opérations se limite à SPAN, RIFE 4.26 et aux modèles AnimeJaNai ; tout
// autre opérateur est refusé dès l'import. Les paramètres qu'ONNX passe en
// entrée (formes de Reshape, bornes de Slice, échelles de Resize…) restent des
// entrées : le pliage des constantes les rend constants.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

namespace ajp::gpu::ir {

enum class DataType : std::uint8_t { F32, F16, I64 };

using Dims = std::vector<std::int64_t>;

// Valeurs d'un tenseur en ordre ligne, sur la forme logique ONNX (NCHW).
// F32 et F16 vivent dans f, I64 dans i. Les constantes F16 importées sont
// représentables en F16 ; les valeurs calculées ne le sont pas forcément :
// l'interpréteur calcule toujours en FP32 et seul un Cast vers F16 arrondit.
struct Value {
  DataType type = DataType::F32;
  Dims dims;
  std::vector<float> f;
  std::vector<std::int64_t> i;
};

enum class TensorKind : std::uint8_t { Input, Constant, Activation };

// Rangement en mémoire d'un tenseur. Nchw : ordre logique ONNX, sans bourrage.
// Nhwc16 : canaux en dernier, leur nombre arrondi au multiple de 16 supérieur.
enum class Layout : std::uint8_t { Nchw, Nhwc16 };

struct Tensor {
  std::string name;
  TensorKind kind = TensorKind::Activation;
  DataType type = DataType::F32;
  Dims dims;           // -1 : dimension encore inconnue
  bool known = false;  // toutes les dimensions sont connues
  Value value;         // constantes seulement
  Layout layout = Layout::Nchw;
};

using TensorId = std::uint32_t;
inline constexpr TensorId kNoTensor = std::numeric_limits<TensorId>::max();

enum class OpKind : std::uint8_t {
  Conv,
  ConvTranspose,
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  Relu,
  LeakyRelu,
  PRelu,
  Sigmoid,
  Tanh,
  Clip,
  Cast,
  Identity,
  Concat,
  Split,
  Slice,
  Reshape,
  Transpose,
  Unsqueeze,
  Squeeze,
  Shape,
  Gather,
  DepthToSpace,
  SpaceToDepth,
  Pad,
  Resize,
  GridSample,
  ReduceMean,
  AveragePool,
  Reciprocal,
  Range,  // plié : ses bornes viennent des formes, connues avant la construction
  // Paire Q/DQ FP8 E4M3 fusionnée à l'import (onnx_fp8.h) : x → arrondi(x / échelle) × échelle.
  FakeQuantE4M3,
};

enum class AutoPad : std::uint8_t { NotSet, SameUpper, SameLower, Valid };

// Activation appliquée en fin de convolution, après le biais et le résiduel.
enum class Activation : std::uint8_t { None, Relu, LeakyRelu, PRelu, Sigmoid, Tanh, Clip };

// DCR : profondeur, colonne, ligne (ONNX par défaut). CRD : ordre de
// pixel_shuffle et pixel_unshuffle de PyTorch.
enum class BlockMode : std::uint8_t { Dcr, Crd };

struct Epilogue {
  bool residual = false;  // ajoute l'entrée kConvResidual
  Activation act = Activation::None;
  float alpha = 0.0F;  // LeakyRelu
  float lo = 0.0F;     // Clip
  float hi = 0.0F;     // Clip
  // Porte de SPAN : la sortie activée multiplie (x + r), x l'entrée de la Conv au même élément
  // (sortie de la forme de x) et r l'entrée kConvResidual, qui n'est alors pas ajoutée avant
  // l'activation (residual reste faux).
  bool gate = false;
  // DepthToSpace fondu : la Conv écrit directement la sortie du DepthToSpace qui la lisait, en
  // blocs de space × space pixels rangés selon space_mode ; 0 : pas de réarrangement. Le résiduel
  // et la porte gardent la forme calculée par la Conv (conv_result_dims).
  std::int64_t space = 0;
  BlockMode space_mode = BlockMode::Dcr;
};

// Forme calculée par une Conv avant son DepthToSpace fondu : celle de son tenseur de sortie
// sans réarrangement.
inline Dims conv_result_dims(Dims out, const Epilogue& e) {
  if (e.space > 1 && out.size() == 4) {
    out[1] *= e.space * e.space;
    out[2] /= e.space;
    out[3] /= e.space;
  }
  return out;
}

// Entrées d'une Conv : les trois d'ONNX, puis celles qu'ajoutent les fusions.
inline constexpr std::size_t kConvX = 0;
inline constexpr std::size_t kConvW = 1;
inline constexpr std::size_t kConvB = 2;
inline constexpr std::size_t kConvResidual = 3;
inline constexpr std::size_t kConvSlope = 4;  // pente PRelu, une par canal de sortie

using Pair = std::array<std::int64_t, 2>;  // (hauteur, largeur)
using Quad = std::array<std::int64_t, 4>;  // (haut, gauche, bas, droite)

struct NoAttrs {};

struct ConvAttrs {
  Pair kernel{0, 0};  // 0 : déduit des poids
  Pair strides{1, 1};
  Pair dilations{1, 1};
  Quad pads{0, 0, 0, 0};
  AutoPad auto_pad = AutoPad::NotSet;
  std::int64_t group = 1;
  Epilogue epilogue;
  // FP8 E4M3 (FakeQuantE4M3 fusionnés par fuse_fp8_quant) : in_quant > 0 lit x arrondi à la grille
  // de cette échelle (la porte de SPAN garde le x brut) ; out_quant > 0 arrondit de même le
  // résultat, après le biais et avant résiduel, activation et porte.
  float in_quant = 0.0F;
  float out_quant = 0.0F;
};

struct ConvTransposeAttrs {
  Pair kernel{0, 0};
  Pair strides{1, 1};
  Pair dilations{1, 1};
  Quad pads{0, 0, 0, 0};
  Pair output_padding{0, 0};
  Dims output_shape;  // vide : déduite des pads
  AutoPad auto_pad = AutoPad::NotSet;
  std::int64_t group = 1;
};

struct LeakyReluAttrs {
  float alpha = 0.01F;
};

struct AxisAttrs {  // Concat, Gather
  std::int64_t axis = 0;
};

struct SplitAttrs {
  std::int64_t axis = 0;
  std::int64_t num_outputs = 0;  // 0 : absent
};

struct ReshapeAttrs {
  bool allowzero = false;
};

struct TransposeAttrs {
  Dims perm;  // vide : ordre inversé
};

struct ShapeAttrs {
  std::int64_t start = 0;
  std::int64_t end = std::numeric_limits<std::int64_t>::max();
};

struct CastAttrs {
  DataType to = DataType::F32;
};

struct ModAttrs {
  bool fmod = false;
};

struct BlockAttrs {  // DepthToSpace, SpaceToDepth
  std::int64_t blocksize = 0;
  BlockMode mode = BlockMode::Dcr;
};

enum class PadMode : std::uint8_t { Constant, Reflect, Edge };

struct PadAttrs {
  PadMode mode = PadMode::Constant;
};

enum class ResizeMode : std::uint8_t { Nearest, Linear };
enum class CoordMode : std::uint8_t { HalfPixel, PytorchHalfPixel, AlignCorners, Asymmetric };
enum class NearestMode : std::uint8_t { RoundPreferFloor, RoundPreferCeil, Floor, Ceil };

struct ResizeAttrs {
  ResizeMode mode = ResizeMode::Nearest;
  CoordMode coord = CoordMode::HalfPixel;
  NearestMode nearest = NearestMode::RoundPreferFloor;
  Dims axes;  // vide : tous les axes
};

enum class GridMode : std::uint8_t { Bilinear, Nearest };
enum class GridPadding : std::uint8_t { Zeros, Border };

struct GridSampleAttrs {
  GridMode mode = GridMode::Bilinear;
  GridPadding padding = GridPadding::Zeros;
  bool align_corners = false;
  bool grid_nchw = false;  // grille [N, 2, H, W] : le Transpose qui la rangeait en [N, H, W, 2], replié
};

struct ReduceAttrs {
  bool keepdims = true;
  bool noop_with_empty_axes = false;
};

struct PoolAttrs {
  Pair kernel{0, 0};
  Pair strides{1, 1};
  Quad pads{0, 0, 0, 0};
  AutoPad auto_pad = AutoPad::NotSet;
  bool count_include_pad = false;
};

using Attrs = std::variant<NoAttrs, ConvAttrs, ConvTransposeAttrs, LeakyReluAttrs, AxisAttrs, SplitAttrs,
                           ReshapeAttrs, TransposeAttrs, ShapeAttrs, CastAttrs, ModAttrs, BlockAttrs,
                           PadAttrs, ResizeAttrs, GridSampleAttrs, ReduceAttrs, PoolAttrs>;

struct Node {
  std::string name;
  OpKind op = OpKind::Identity;
  Attrs attrs;
  std::vector<TensorId> inputs;  // kNoTensor : entrée facultative absente
  std::vector<TensorId> outputs;
};

// Un graphe dont les nœuds sont toujours rangés en ordre topologique.
struct Graph {
  std::vector<Tensor> tensors;
  std::vector<Node> nodes;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  std::int64_t opset = 0;
};

}  // namespace ajp::gpu::ir
