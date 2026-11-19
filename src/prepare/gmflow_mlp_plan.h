// Le MLP des couches de cross-attention du transformeur de GMFlow (le flot du graphe « paire » de GMFSS
// Fortuna union) tel que le noyau natif le fait (docs/gmflow-natif.md, « MLP natif (PR 3) ») : les poids des
// six couches, lus par prepare/gmflow_mlp.h dans l'ONNX de la paire, et leurs emplois aux deux échelles du
// transformeur, que le backend exécute à la lettre et que tools/gmfss/gmflow_mlp_emule.py rejoue. Sans
// std::expected, car nvcc lit cet en-tête en C++20.
#pragma once

#include <string>
#include <vector>

namespace kirari::prepare {

inline constexpr int kGmflowMlpChannels = 128;  // d du transformeur : x et message, sortie
// La tranche cachée du noyau : la couche cachée en est un multiple.
inline constexpr int kGmflowMlpHiddenTile = 64;
inline constexpr int kGmflowMlpMaxHidden = 4096;

// Une couche : out = x + LayerNorm(W2 · GELU(W1 · [x, message])), GELU exacte (erf), sans biais aux deux
// produits. w1 : [hidden][2 · channels], la sortie cachée n puis l'entrée k (x de 0 à channels - 1, puis
// message), rangés depuis le MatMul ONNX [2 · channels][hidden] ; w2 : [channels][hidden], depuis
// [hidden][channels]. gamma et beta : l'échelle et le décalage de la LayerNorm (norm2), channels valeurs ;
// epsilon : le sien.
struct GmflowMlpLayer {
  std::vector<float> w1;
  std::vector<float> w2;
  std::vector<float> gamma;
  std::vector<float> beta;
  float epsilon = 1e-5F;
};

// Un emploi d'une couche dans le graphe : l'échelle (0 : le huitième, 1 : le quart), la couche, et les noms
// ONNX de ses entrées (source : x, le résiduel ; message : la sortie de norm1) et de sa sortie (l'Add du
// résiduel), que l'émulation lit sous ONNX Runtime.
struct GmflowMlpUse {
  int scale = 0;
  int layer = 0;
  std::string source;
  std::string message;
  std::string output;
};

// Le MLP : les couches (les mêmes poids aux deux échelles) et leurs emplois, dans l'ordre (échelle, couche).
struct GmflowMlp {
  int channels = kGmflowMlpChannels;
  int hidden = 0;
  std::vector<GmflowMlpLayer> layers;
  std::vector<GmflowMlpUse> uses;
};

}  // namespace kirari::prepare
