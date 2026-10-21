// Réécritures de l'interface d'un fichier ONNX. Dans un autre type flottant : chaque entrée ou
// sortie du graphe en FLOAT ou FLOAT16 qui n'est pas dans le type voulu le prend, et un Cast
// ajouté la relie au reste du graphe, inchangé (les nœuds qui la lisaient ou l'écrivaient passent
// par un nom interne). Aux dimensions fixées : les entrées déclarent la forme de la chaîne. Le
// reste du fichier est recopié octet par octet, poids compris.
//
// Pour les runtimes qui prennent le fichier ONNX tel quel (Core ML par ONNX Runtime) : le réseau
// lit et écrit les tenseurs de la chaîne dans leur type, sans conversion par l'hôte à chaque image.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
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

// Une entrée du graphe et les dimensions qu'on lui fixe.
struct InputDims {
  std::string name;
  std::vector<std::int64_t> dims;
};

// Le fichier dont ces entrées déclarent leurs dimensions fixées (dim_value) au lieu de
// dimensions symboliques ou anonymes, ou vide quand elles les déclaraient déjà toutes. Pour Core
// ML par ONNX Runtime : une dimension sans nom (le -1 des RIFE de vs-mlrt) échappe à
// AddFreeDimensionOverrideByName, et le fournisseur Core ML laisse alors au processeur chaque
// nœud dont il ne connaît pas la forme (Resize, GridSample, Reshape…). Refus InputMismatch :
// entrée absente, rang différent, dimension non positive ou qui contredit une dimension fixe du
// fichier ; OnnxMalformed, OnnxUnsupported : ceux du décodeur.
std::expected<std::vector<std::byte>, Refusal> with_input_dims(std::span<const std::byte> model,
                                                               std::span<const InputDims> inputs);

}  // namespace ajp::gpu::ir::onnx
