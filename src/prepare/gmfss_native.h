// Le GridNet de la fusion de GMFSS Fortuna Union lu pour les noyaux natifs du backend TensorRT
// (--gmfss-native, docs/gmfss-natif.md) : les poids des convolutions et les pentes des PReLU, pris par leurs
// noms dans l'ONNX qu'exporte tools/gmfss/export_gmfss.py (initialiseurs « gmfss.fusionnet.<module>.<i> »,
// PReLU exportés en Relu, Neg, Relu, Mul par une constante, Sub). La structure (qui lit quoi, quelles sommes)
// est celle du graphe exporté, relevée une fois et tenue ici en table : un autre graphe se trahit par un nom
// ou une forme qui manque, ou par l'essai au chargement contre TensorRT.
#pragma once

#include <cstddef>
#include <expected>
#include <span>

#include "core/types.h"
#include "prepare/gmfss_grid.h"

namespace kirari::prepare {

// Le réseau : les entrées x0 à x3, puis 21 blocs de deux convolutions et la queue (trois convolutions).
// Refus : ModelUnreadable (ONNX illisible), FieldMissing (un poids ou une pente manque, ou n'a pas la forme
// attendue, avec son nom).
std::expected<GmfssGrid, Refusal> gmfss_grid(std::span<const std::byte> fusion);

// Le réseau se tient-il ? Chaque convolution lit un tenseur déjà écrit, de ses canaux (rembourrés à 16), à la
// taille que son pas demande ; le résiduel a la taille et les canaux de la sortie. Refus FieldMissing.
std::expected<void, Refusal> gmfss_grid_check(const GmfssGrid& grid);

}  // namespace kirari::prepare
