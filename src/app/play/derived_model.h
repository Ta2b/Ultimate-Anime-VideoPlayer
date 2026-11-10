// Un modèle dérivé d'un ONNX livré (variante mouvement de RIFE, DistilDRBA à la taille
// fixée) : écrit une fois dans le dossier de données du lecteur, jamais à côté du modèle ni dans
// le dépôt, sous un nom qui porte l'empreinte du modèle source.
#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::app::play {

// La version des transformations : à monter avec toute correction de l'une d'elles (A36-04).
inline constexpr int kDerivedModelVersion = 2;

using ModelTransform =
    std::function<std::expected<std::vector<std::byte>, Refusal>(std::span<const std::byte> model)>;

// dir/<nom de source>-<suffix>-t<kDerivedModelVersion>-<16 premiers chiffres du SHA-256 de source>.onnx,
// créé par transform s'il manque (écrit à côté puis renommé : jamais un fichier à moitié écrit) ; les
// anciens dérivés de même source et même suffixe sont alors retirés. Refus : ceux de transform,
// ModelUnreadable, FileUnwritable.
std::expected<std::filesystem::path, Refusal> derived_model(const std::filesystem::path& source,
                                                            std::string_view suffix,
                                                            const std::filesystem::path& dir,
                                                            const ModelTransform& transform);

}  // namespace ajp::app::play
