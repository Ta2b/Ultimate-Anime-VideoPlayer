// Entrées et sorties du cache des moteurs : messages d'erreur, écriture atomique,
// dossiers de travail marqués à leur propriétaire et nettoyage des restes.
//
// Repris de animejanai-inference-rtx@118c272 :
//   src/cache_io.h:9-45           io_message, cleanup_stage
//   src/cache_stage_owner.h:8-45  mark_stage, owns_stage, has_owned_stages, recover_owned_stages
//   src/model_identity.cpp:33-41  reserve (dossier au nom aléatoire)
#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

#include "core/types.h"

namespace ajp::prepare {

// « opération : chemin : erreur », suivi du conseil du fork (place, droits).
std::string io_message(std::string_view operation, const std::filesystem::path& path, std::error_code error);
Refusal io_refusal(std::string_view operation, const std::filesystem::path& path, std::error_code error);

// Écrit bytes dans un temporaire du même dossier puis le renomme : un lecteur
// voit l'ancien contenu ou le nouveau, jamais un fichier à moitié écrit.
std::expected<void, Refusal> write_atomic(const std::filesystem::path& destination, std::string_view bytes);

// Réserve un dossier neuf « parent/<prefix><aléa> » et y écrit son propriétaire.
std::expected<std::filesystem::path, Refusal> reserve_stage(const std::filesystem::path& parent,
                                                            std::string_view prefix, std::string_view owner);

// Le dossier porte-t-il exactement ce propriétaire ? Un lien symbolique n'est jamais à nous.
bool owns_stage(const std::filesystem::path& folder, std::string_view owner);
bool has_owned_stages(const std::filesystem::path& parent, std::string_view prefix, std::string_view owner);

// Supprime les dossiers de ce propriétaire. L'appelant tient le verrou de la clé
// et ne crée ses dossiers qu'après l'avoir pris : tout dossier trouvé est orphelin.
std::expected<void, Refusal> recover_owned_stages(const std::filesystem::path& parent,
                                                  std::string_view prefix, std::string_view owner);

// Supprime un dossier de travail. Garde le marqueur si le parent n'est pas
// inscriptible : une demande suivante le récupérera une fois les droits rendus.
std::expected<void, Refusal> cleanup_stage(const std::filesystem::path& folder);

}  // namespace ajp::prepare
