// Une métadonnée d'un ONNX (ModelProto.metadata_props), lue dans le fichier sans charger le graphe ni
// les poids : le lecteur la consulte à chaque recherche de modèle, sur des fichiers de dizaines de Mio.
// Au niveau du format protobuf, sans dépendance onnx.
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include "core/types.h"

namespace ajp::prepare {

// La valeur de la métadonnée key du modèle file (la dernière, si la clé est répétée), ou nullopt si le
// modèle ne la porte pas. Les champs du modèle sont parcourus en sautant les messages volumineux (le
// graphe) par un déplacement dans le fichier. Refus : ModelUnreadable (fichier illisible ou vide,
// protobuf malformé ou tronqué).
std::expected<std::optional<std::string>, Refusal> onnx_metadata(const std::filesystem::path& file,
                                                                 std::string_view key);

}  // namespace ajp::prepare
