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
};

enum class AutoPad : std::uint8_t { NotSet, SameUpper, SameLower, Valid };

// Activation appliquée en fin de convolution, après le biais et le résiduel.
enum class Activation : std::uint8_t { None, Relu, LeakyRelu, PRelu, Sigmoid, Tanh, Clip };

struct Epilogue {
  bool residual = false;  // ajoute l'entrée kConvResidual
  Activation act = Activation::None;
  float alpha = 0.0F;  // LeakyRelu
  float lo = 0.0F;     // Clip
  float hi = 0.0F;     // Clip
};

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

// DCR : profondeur, colonne, ligne (ONNX par défaut). CRD : ordre de
// pixel_shuffle et pixel_unshuffle de PyTorch.
enum class BlockMode : std::uint8_t { Dcr, Crd };

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
