// Le FeatureNet et le MetricNet de la paire de GMFSS Fortuna Union lus pour les noyaux natifs du backend
// TensorRT (--gmfss-native, docs/gmfss-natif.md) : les poids des convolutions et les pentes des PReLU, pris
// par leurs noms dans l'ONNX qu'exporte tools/gmfss/export_gmfss.py.
//   initialiseurs : gmfss.feat_ext.<bloc>.<i>, gmfss.metricnet.<module> ;
//   PReLU (Relu, Neg, Relu, Mul par une constante, Sub) : /feat_ext/<bloc>/<bloc>.<i>/Mul,
//   /metricnet/<module>/<module>.0/Mul.
// La structure est celle du code de la roue, relevée une fois et tenue ici en table, comme pour le GridNet
// (gmfss_native.h). Le FeatureNet n'est lu qu'une fois : les deux images passent par les mêmes poids.
//
// Les poids se lisent dans une paire FP16 (gmfss_fortuna_union_pair.onnx, ou _s05) : ils y sont les mêmes à
// toutes les échelles du flot.
#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "core/types.h"
#include "prepare/gmfss_pair_grid.h"

namespace kirari::prepare {

// Les deux réseaux. Refus : ModelUnreadable (ONNX illisible), FieldMissing (un poids ou une pente manque, ou
// n'a pas la forme attendue, avec son nom ; un réseau qui ne se tient pas).
std::expected<GmfssPairNets, Refusal> gmfss_pair_nets(std::span<const std::byte> pair);

// Les réseaux se tiennent-ils ? Chaque convolution lit un tenseur déjà écrit, de ses canaux (rembourrés à
// 16), à la taille que son pas demande ; le résiduel a la taille et les canaux de la sortie ; l'image n'est
// lue que par une convolution de pas 2 à 3 canaux ; toutes les sorties sont écrites. Refus FieldMissing.
std::expected<void, Refusal> gmfss_pair_check(const GmfssPairNets& nets);

}  // namespace kirari::prepare
