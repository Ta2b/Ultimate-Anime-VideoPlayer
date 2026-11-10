// Les paquets de moteurs publiés après la release (TensorRT construit plus tard) : la release de cette
// version porte engines-nvidia.lock, que le lecteur relit au lancement et ajoute à l'engines.lock livré avec
// lui. Un paquet TensorRT joint après coup s'installe ainsi depuis l'écran des moteurs, sans refaire la
// release ni réinstaller le lecteur.
//
// Confiance : l'index porte une signature Ed25519 (engines-nvidia.lock.sig) de la clé des mises à jour,
// posée par la CI de release (A26-02) ; sans elle, rien n'est cru. Il n'est cru ensuite que pour des archives
// du dépôt des releases (les paquets TensorRT vivent dans trt-<empreinte> et trt-win-<empreinte>) et des
// fichiers de pypi.nvidia.com (comme l'engines.lock livré) ; chaque archive se vérifie enfin par sa taille et
// son SHA-256, comme avant.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/engine_install.h"
#include "app/play/updater.h"

namespace ajp::app::play {

// La release d'une version : « https://github.com/<dépôt>/releases/download/v<version>/ ».
std::string release_download_prefix(std::string_view repo, std::string_view version);
// Toutes les releases du dépôt : « https://github.com/<dépôt>/releases/download/ ».
std::string release_repo_prefix(std::string_view repo);

// Vrai si signature (hexadécimal, fin de ligne tolérée) signe text pour l'une des clés.
bool release_index_signed(std::string_view text, std::string_view signature,
                          const std::vector<std::string>& keys);

// Ajoute au catalogue les moteurs de text (un engines.lock) pour cet OS dont l'archive est sous prefix ; un
// moteur déjà au catalogue est remplacé, ses fichiers NVIDIA compris (le paquet de la release est le plus
// récent). Rend le nombre de moteurs pris ; une entrée hors de la release est écartée et dite dans problem.
int merge_release_index(EngineCatalog& catalog, std::string_view text, std::string_view prefix);

// Au lancement : relit engines-nvidia.lock et sa signature de la release (au plus une fois par jour, en
// quelques secondes au plus, gardés dans cache et cache.sig), puis l'ajoute au catalogue s'il est signé par
// l'une des keys. Sans réseau, la copie gardée sert, revérifiée.
void add_release_engines(EngineCatalog& catalog, const std::filesystem::path& cache, std::string_view repo,
                         std::string_view version, const std::vector<std::string>& keys,
                         const UpdateFetch& fetch);

}  // namespace ajp::app::play
