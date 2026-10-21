// Réécriture de l'interface d'un fichier ONNX : dans un autre type flottant (with_io_type), ou à
// des formes fixes (with_input_dims). Dans un autre type, chaque entrée ou sortie du graphe en
// FLOAT ou FLOAT16 qui n'est pas dans le type voulu le prend, et un Cast ajouté la relie au reste
// du graphe, inchangé (les nœuds qui la lisaient ou l'écrivaient passent par un nom interne). Le
// reste du fichier est recopié octet par octet, poids compris.
//
// Pour les runtimes qui prennent le fichier ONNX tel quel (Core ML par ONNX Runtime) : le réseau
// lit et écrit les tenseurs de la chaîne dans leur type, sans conversion par l'hôte à chaque image.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::gpu::ir::onnx {

// Le fichier réécrit, ou vide quand toutes les entrées et sorties flottantes sont déjà dans
// elem_type (kFloat ou kFloat16). Refus FeatureUnavailable : un graphe à sous-graphes (If,
// Loop…), dont les portées ne se renomment pas ici, un nom à la fois entrée et sortie, une
// sortie qui est un poids, un opset sans Cast typé (avant 6) ; OnnxMalformed, OnnxUnsupported :
// ceux du décodeur.
std::expected<std::vector<std::byte>, Refusal> with_io_type(std::span<const std::byte> model,
                                                            std::int32_t elem_type);

// Une entrée du graphe et la forme où la fixer.
struct InputDims {
  std::string_view name;
  std::span<const std::int64_t> dims;
};

// Les entrées nommées à ces formes : chaque dimension libre (nommée, -1 ou sans valeur) prend la
// sienne, le reste du fichier est recopié octet par octet. Le fichier réécrit, ou vide quand
// elles sont toutes déjà fixées. Pour Core ML par ONNX Runtime : une dimension libre sans nom
// échappe aux surcharges par nom, et le fournisseur CoreML refuse alors chaque nœud qui demande
// une forme connue (Reshape, DepthToSpace CRD, Resize qui réduit), qui retombe sur le CPU.
// Refus InputMismatch : une entrée absente, un autre rang, une dimension fixe à une autre
// valeur ; OnnxMalformed, OnnxUnsupported : ceux du décodeur.
std::expected<std::vector<std::byte>, Refusal> with_input_dims(std::span<const std::byte> model,
                                                               std::span<const InputDims> inputs);

}  // namespace ajp::gpu::ir::onnx
