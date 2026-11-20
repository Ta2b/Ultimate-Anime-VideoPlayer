// Le squelette CNN de GMFlow (le flot du graphe « paire » de GMFSS Fortuna union) tel que les noyaux natifs
// le font (docs/gmflow-natif.md) : les poids, lus par leurs noms dans l'ONNX de la paire par
// prepare/gmflow_native.h, et le plan qui les enchaîne, des tenseurs et une liste d'opérations, que le
// backend exécute à la lettre et que tools/gmfss/gmflow_squelette_emule.py rejoue. Sans std::expected, car
// nvcc lit cet en-tête en C++20.
#pragma once

#include <array>
#include <vector>

namespace kirari::prepare {

// Des poids de convolution carrée, de bord kernel / 2, sans groupes ni dilatation : [cout][kernel²][cin],
// rangés depuis l'ONNX [cout][cin][ky][kx] (prise ky · kernel + kx). Le biais a cout valeurs, nulles si la
// convolution n'en a pas.
struct GmflowConv {
  int kernel = 3;
  int cin = 0;
  int cout = 0;
  std::vector<float> weights;
  std::vector<float> bias;
};

// Un tenseur du plan, deux images (le lot de GMFlow) rangées l'une après l'autre. level 0 : les images
// d'entrée, trois plans FP16 chacune ; 1, 2, 3 : NHWC FP16 à la moitié, au quart et au huitième.
struct GmflowTensor {
  int level = 0;
  int channels = 0;
};

// Une lecture : le tenseur, et s'il est la sortie brute d'une convolution suivie d'une InstanceNorm, les
// statistiques de celle-ci (stats ≥ 0 : (x - moyenne) / √(variance + ε) par image et par canal, puis la Relu
// si relu).
struct GmflowRead {
  int tensor = -1;
  int stats = -1;
  bool relu = false;
};

// Une opération du plan.
//  - kConv : la convolution conv au pas stride lit in et écrit out ; stats ≥ 0 : elle ajoute aux statistiques
//    stats les sommes Σx et Σx² par image et par canal de ce qu'elle écrit (après l'arrondi FP16).
//  - kBlockEnd : la fin d'un bloc résiduel, élément par élément : out = relu(skip + in), les deux lus avec
//    leurs normalisations.
enum class GmflowOpKind : int { kConv, kBlockEnd };
struct GmflowOp {
  GmflowOpKind kind = GmflowOpKind::kConv;
  int conv = -1;
  int stride = 1;
  GmflowRead in;
  GmflowRead skip;
  int out = -1;
  int stats = -1;
};

inline constexpr int kGmflowMaxChannels = 128;  // les statistiques d'une lecture tiennent dans un bloc
inline constexpr int kGmflowStemTaps = 7 * 7 * 3;
inline constexpr int kGmflowStemK = 160;  // les 147 produits de la 7 × 7 sur trois plans, par 16

// Normalisation des trois plans d'entrée FP16. Le banc peut ranger la normalisation dans les poids et le
// biais de la 7 × 7 (kFolded). La normalisation à la lecture (kNormalizeF16) garde les poids ONNX.
enum class GmflowStemInput { kFolded, kNormalizeF16, kAlreadyNormalized };

// Le squelette : CNNEncoder de GMFlow (num_output_scales = 2). pad est la moyenne FP16 par plan, utilisée
// comme valeur de bord en kFolded. Avec des poids ONNX d'origine, le halo vaut zéro. La normalisation se
// fait dans le stem en kNormalizeF16, ou à la réduction de l'image en kAlreadyNormalized.
struct GmflowBackbone {
  std::array<float, 3> pad{};
  float epsilon = 1e-5F;  // des InstanceNorm, toutes sans paramètres
  std::vector<GmflowConv> convs;
  std::vector<GmflowTensor> tensors;
  std::vector<GmflowOp> ops;
  int stats = 0;     // nombre d'InstanceNorm
  int image = 0;     // le tenseur d'entrée
  int quarter = -1;  // les sorties, 128 caractéristiques au quart (trident de pas 1) et au huitième (pas 2)
  int eighth = -1;
  GmflowStemInput stem_input = GmflowStemInput::kFolded;
  std::array<float, 3> deviation{};  // écarts FP16 convertis en float pour la division du stem
};

}  // namespace kirari::prepare
