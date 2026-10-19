// Changer les sorties d'un graphe ONNX, au niveau du format protobuf, sans dépendance onnx : sert à
// dériver de RIFE sa variante « mouvement », qui sort le flot et le masque au lieu de l'image
// (kernels/spec/warp.h). Tout le reste du modèle (nœuds, poids, métadonnées) est recopié octet
// pour octet ; les nœuds devenus inutiles restent, TensorRT les écarte.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "core/types.h"

namespace ajp::prepare {

struct OnnxOutput {
  std::string name;             // tenseur produit par un nœud du graphe principal
  std::int32_t elem_type = 10;  // TensorProto.DataType : 10 = FLOAT16, 1 = FLOAT
};

// Les noms des tenseurs produits par les nœuds du graphe principal.
std::expected<std::vector<std::string>, Refusal> onnx_node_outputs(std::span<const std::byte> model);

// Le modèle dont le graphe principal sort exactement outputs, dans cet ordre. Refus :
// ModelUnreadable (protobuf malformé, pas de graphe), FieldMissing (un tenseur demandé n'est
// produit par aucun nœud).
std::expected<std::vector<std::byte>, Refusal> with_outputs(std::span<const std::byte> model,
                                                            std::span<const OnnxOutput> outputs);

}  // namespace ajp::prepare
