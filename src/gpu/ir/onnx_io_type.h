// Réécriture de l'interface d'un fichier ONNX : dans un autre type flottant (with_io_type), ou à
// des formes fixes (with_input_dims) ; et du modèle entier en FLOAT (with_float32). Dans un autre
// type, chaque entrée ou sortie du graphe en FLOAT ou FLOAT16 qui n'est pas dans le type voulu le
// prend, et un Cast ajouté la relie au reste du graphe, inchangé (les nœuds qui la lisaient ou
// l'écrivaient passent par un nom interne). Le reste du fichier est recopié octet par octet, poids
// compris.
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

// Le modèle entier en FLOAT : poids, constantes et formes FLOAT16 convertis, le type cible
// FLOAT16 d'un Cast (ou d'un générateur, dtype) remplacé par FLOAT ; l'interface aussi, que
// with_io_type remet ensuite dans le type de la chaîne. Pour le NPU d'AMD (VitisAI), dont le
// compilateur prend un modèle FLOAT pour le calculer en BF16. Le fichier réécrit, ou vide quand
// rien n'y est en FLOAT16. Refus FeatureUnavailable : un sous-graphe (If, Loop…), un tenseur
// creux, des fonctions locales, un poids FLOAT16 hors du fichier ; OnnxMalformed,
// OnnxUnsupported : ceux du décodeur.
std::expected<std::vector<std::byte>, Refusal> with_float32(std::span<const std::byte> model);

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

// Le modèle avec l'entrée key = value dans ses métadonnées (metadata_props du ModelProto), à la
// place d'une entrée de même clé ; le reste du fichier recopié octet par octet. Le fichier
// réécrit, ou vide quand l'entrée y est déjà. Pour Core ML par ONNX Runtime : COREML_CACHE_KEY
// nomme le modèle compilé du cache, qui sans elle est retrouvé par le chemin du fichier ONNX.
// Refus OnnxMalformed : celui du lecteur protobuf.
std::expected<std::vector<std::byte>, Refusal> with_metadata(std::span<const std::byte> model,
                                                             std::string_view key, std::string_view value);

}  // namespace ajp::gpu::ir::onnx
