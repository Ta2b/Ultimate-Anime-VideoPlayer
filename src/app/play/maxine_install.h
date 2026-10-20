// RTX HDR sous Linux installé par le lecteur : le SDK Video Effects de NVIDIA ne se redistribue
// pas, il se télécharge depuis NGC avec la clé d'API de l'utilisateur (compte NVIDIA gratuit),
// après qu'il en a accepté la licence à l'écran. Le lecteur obtient un jeton NGC, télécharge le
// cœur du SDK (engines.lock, [maxine_vfx]), l'extrait, lance son install_feature.sh pour TrueHDR
// (la clé passe par l'environnement, jamais par la ligne de commande), puis l'importe dans le
// magasin des SDK (vendor_hdr/sdk_store.h), qui le vérifie comme un import manuel.
//
// La clé n'est jamais écrite ailleurs que dans des fichiers 0600 du dossier des téléchargements,
// effacés à la fin quelle qu'elle soit.
//
// Voie préférée, sans clé ni compte : notre paquet fermé (engines.lock, [[engine]] maxine_vfx).
// La licence du SDK permet de le livrer comme partie du lecteur (§1.1.3), jamais sous sa licence
// libre (§8.8) : un paquet à part, qui ne contient que ce que le magasin copie pour TrueHDR,
// sans modification. NGC ne sert que s'il n'est pas publié.
#pragma once

#include <expected>
#include <filesystem>
#include <string>

#include "app/play/engine_install.h"
#include "core/model_index.h"
#include "core/types.h"

namespace ajp::app::play {

// Refus : LibraryUnavailable (curl, tar ou bash absents), DecodeFailed (NGC refuse la clé, ou le
// script échoue), ceux du magasin à l'import, Cancelled.
std::expected<void, Refusal> install_maxine(const EngineCatalog& catalog, const core::NgcSdkSource& source,
                                            const std::string& api_key, const std::filesystem::path& sdk_dir,
                                            const InstallProgress& progress);

// Télécharge et vérifie le paquet comme un moteur, l'importe dans le magasin (vérifié comme un
// import manuel), puis efface le dossier extrait. Refus : ceux d'install_package et du magasin.
std::expected<void, Refusal> install_maxine_package(const EngineCatalog& catalog,
                                                    const core::EnginePackage& package,
                                                    const std::filesystem::path& sdk_dir,
                                                    const InstallProgress& progress);

}  // namespace ajp::app::play
