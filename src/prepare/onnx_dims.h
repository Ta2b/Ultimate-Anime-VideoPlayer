// Fixer les dimensions variables d'un graphe ONNX, au niveau du format protobuf, sans dépendance
// onnx : un graphe exporté pour toute taille (« H », « W/8 »…) devient le graphe statique d'une
// taille, que TensorRT construit sans profil, entrée par entrée. Les nœuds et les poids sont
// recopiés octet pour octet.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"

namespace ajp::prepare {

struct NamedDim {
  std::string name;  // dim_param du graphe
  std::int64_t value = 0;
};

// Le modèle dont chaque dimension nommée de dims prend sa valeur, dans les entrées, les sorties et
// les value_info du graphe principal. Refus : ModelUnreadable (protobuf malformé, pas de graphe),
// FieldMissing (une entrée ou une sortie garde une dimension nommée absente de dims),
// FieldOutOfRange (une valeur nulle ou négative).
std::expected<std::vector<std::byte>, Refusal> with_fixed_dims(std::span<const std::byte> model,
                                                               std::span<const NamedDim> dims);

}  // namespace ajp::prepare
