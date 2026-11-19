// Le graphe ONNX d'un paquet « .span8 » (package.h) : le SPAN « unshuffle » d'AnimeJaNai tel que forward.h le
// calcule, rebâti de ses poids, pour un moteur qui exécute des graphes (le moteur Vulkan, qui l'importe par
// gpu/ir/import.h comme un fichier ONNX). Avec quantize, chacun des 1 + 3·blocs tenseurs que le paquet
// quantifie passe par un FakeQuantE4M3 (l'arrondi à la grille E4M3 de son échelle, gpu/ir/onnx_fp8.h) devant
// la convolution qui le lit, et les poids des convolutions de bloc sont ramenés sur la grille E4M3 de leur
// canal de sortie (quantize.h, reconstruct) : le moteur y reconnaît les convolutions FP8
// (ConvAttrs::in_quant, ir::recover_fp8_weights) et les passe à ses matrices coopératives FP8 quand la carte
// en a.
//
// Ce qui diffère des noyaux CUTLASS du backend TensorRT (span_fp8_body.cuh) : le saut de chaque bloc
// (x + c3) lit x tel quel, pas son arrondi E4M3 (le graphe d'ONNX, et un tenseur de moins à arrondir) ; les
// biais sont en FP16 comme dans l'ONNX. La porte de la dérivation juge le moteur qui joue, pas les noyaux
// CUTLASS : l'écart ne compte que pour la comparaison des reçus d'un moteur à l'autre, qui ne se fait pas
// (gpu_arch).
//
// Le dégroupage 2×2 passe par un SpaceToDepth d'ONNX (ordre DCR, canal (dy·2 + dx)·3 + c), les entrées de
// conv_1 rangées en conséquence : le calcul est celui du dégroupage de PyTorch (canal c·4 + dy·2 + dx).
// Les canaux réels du paquet seulement (real_channels) : un b5f48 calcule en 48 canaux.
#pragma once

#include <expected>
#include <string>
#include <vector>

#include "core/types.h"
#include "gpu/ir/onnx_model.h"
#include "gpu/span_fp8/package.h"

namespace kirari::gpu::span_fp8 {

// Noms de l'entrée et de la sortie du graphe.
inline constexpr const char* kModelInput = "input";
inline constexpr const char* kModelOutput = "output";

struct ModelOptions {
  // Les FakeQuantE4M3 aux échelles du paquet et les poids des blocs sur la grille E4M3 ; sans effet sur un
  // paquet FP16 (Package::fp16), dont rien ne se quantifie. Faux : la référence FP16 de la dérivation.
  bool quantize = true;
  // Les 1 + 3·blocs tenseurs quantifiés en sorties du graphe, après la sortie de l'image, dans l'ordre des
  // échelles (peak_names) : la dérivation en relève les maxima.
  bool peaks = false;
};

// Les noms des tenseurs quantifiés, dans l'ordre des échelles : [0] sortie de conv_1, puis pour chaque bloc
// le premier Relu, le second, la sortie du bloc.
std::vector<std::string> peak_names(std::size_t blocks);

// Refus FormatUnsupported : nombre de blocs ou de canaux hors de ceux que parse accepte, échelle non
// positive ou non finie quand le graphe quantifie.
std::expected<ir::onnx::Model, Refusal> package_model(const Package& package,
                                                      const ModelOptions& options = {});

}  // namespace kirari::gpu::span_fp8
