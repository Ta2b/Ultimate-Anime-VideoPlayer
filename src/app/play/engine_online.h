// Les paquets de moteurs publiés après la release (TensorRT construit plus tard) : la release de cette
// version porte engines-nvidia.lock, que le lecteur relit au lancement et ajoute à l'engines.lock livré avec
// lui. Un paquet TensorRT joint après coup s'installe ainsi depuis l'écran des moteurs, sans refaire la
// release ni réinstaller le lecteur.
//
// Confiance : celle du lecteur lui-même. L'index vient de la release de cette version, en https, et n'est
// cru que pour des archives de cette même release et des fichiers de pypi.nvidia.com (comme l'engines.lock
// livré) ; chaque archive se vérifie ensuite par sa taille et son SHA-256, comme avant.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "app/play/engine_install.h"
#include "app/play/updater.h"

namespace ajp::app::play {

// La release d'une version : « https://github.com/<dépôt>/releases/download/v<version>/ ».
std::string release_download_prefix(std::string_view repo, std::string_view version);

// Ajoute au catalogue les moteurs de text (un engines.lock) pour cet OS dont l'archive est sous prefix ; un
// moteur déjà au catalogue est remplacé, ses fichiers NVIDIA compris (le paquet de la release est le plus
// récent). Rend le nombre de moteurs pris ; une entrée hors de la release est écartée et dite dans problem.
int merge_release_index(EngineCatalog& catalog, std::string_view text, std::string_view prefix);

// Au lancement : relit engines-nvidia.lock de la release (au plus une fois par jour, en quelques secondes au
// plus, gardé dans cache), puis l'ajoute au catalogue. Sans réseau, la copie gardée sert.
void add_release_engines(EngineCatalog& catalog, const std::filesystem::path& cache, std::string_view repo,
                         std::string_view version, const UpdateFetch& fetch);

}  // namespace ajp::app::play
