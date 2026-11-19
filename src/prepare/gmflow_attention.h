// Les couches d'attention du transformeur de GMFlow (le flot du graphe « paire » de GMFSS Fortuna union)
// telles que les noyaux natifs les font (docs/gmflow-natif.md, PR 2) : les poids, lus par leurs noms dans
// l'ONNX de la paire par prepare/gmflow_attention_native.h, et le plan qui enchaîne les six blocs, que le
// backend exécute à la lettre et que tools/gmfss/gmflow_attention_emule.py rejoue. Sans std::expected, car
// nvcc lit cet en-tête en C++20.
#pragma once

#include <array>
#include <vector>

namespace kirari::prepare {

inline constexpr int kGmflowModel = 128;  // canaux d'un jeton, une seule tête
inline constexpr int kGmflowQkv = 3 * kGmflowModel;
inline constexpr int kGmflowBlocks = 6;
inline constexpr int kGmflowLayers = 2 * kGmflowBlocks;  // self puis cross dans chaque bloc

// Une couche d'attention : q = x · Wq, k = t · Wk, v = t · Wv (sans biais), softmax(q kᵀ · scale + masque) v
// par fenêtres, puis la fusion, puis la LayerNorm. En self-attention t = x et la couche rend x + LN (le
// résiduel) ; en cross-attention t est l'autre moitié du lot telle qu'elle entrait dans le bloc, et la couche
// rend LN seule, le message que le MLP (PR 3) lit avec x. Poids rangés [sortie][entrée] (Linear de PyTorch),
// depuis les MatMul ONNX [entrée][sortie].
struct GmflowAttentionLayer {
  bool cross = false;
  bool shifted = false;      // fenêtres décalées d'une demi-fenêtre (Swin), masque entre régions
  std::vector<float> qkv;    // [384][128] : les lignes de q, puis de k, puis de v
  std::vector<float> merge;  // [128][128]
  std::vector<float> gamma;  // LayerNorm (norm1), 128 valeurs
  std::vector<float> beta;
};

// Une opération du plan, sur des tenseurs de jetons [lot][hauteur][largeur][128] FP16.
//  - kAttention : la couche layer, q lu dans query, k et v dans keys (dans l'autre moitié du lot si
//    other_half : le jeton du lot (b + lot/2) mod lot), out = query + LN si residual, LN sinon.
//  - kMlp : le MLP fondu de la PR 3, out = query + LN2(MLP([query, message])) ; ses poids ne sont pas
//    lus ici, l'opération dit seulement où le bloc se ferme.
enum class GmflowAttnOpKind : int { kAttention, kMlp };
struct GmflowAttnOp {
  GmflowAttnOpKind kind = GmflowAttnOpKind::kAttention;
  int layer = -1;
  int query = -1;
  int keys = -1;
  bool other_half = false;
  bool residual = false;
  int message = -1;
  int out = -1;
};

// Le transformeur, mêmes poids aux deux échelles : six blocs (self, cross, MLP). Le tenseur input est
// l'entrée (caractéristiques + position), output la sortie du dernier bloc. Les fenêtres : splits × splits
// par image, 2 × 2 au huitième, 8 × 8 au quart ; le décalage d'une couche décalée est d'une demi-fenêtre
// (arrondie en bas) et le masque vaut mask entre deux régions différentes de l'image décalée (trois par axe).
struct GmflowAttention {
  float scale = 0.F;                         // de q kᵀ : 1/√128
  float mask = 0.F;                          // -100
  float epsilon = 0.F;                       // des LayerNorm norm1
  std::array<int, 2> splits{};               // fenêtres par côté, au huitième puis au quart
  std::vector<GmflowAttentionLayer> layers;  // kGmflowLayers : self du bloc b en 2b, cross en 2b + 1
  int tensors = 0;
  int input = 0;
  int output = -1;
  std::vector<GmflowAttnOp> ops;
};

}  // namespace kirari::prepare
