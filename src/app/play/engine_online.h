// Les paquets de moteurs publiés après la release (TensorRT construit plus tard) : la release de cette
// version porte engines-nvidia.lock, que le lecteur relit au lancement et ajoute à l'engines.lock livré avec
// lui. Un paquet TensorRT joint après coup s'installe ainsi depuis l'écran des moteurs, sans refaire la
// release ni réinstaller le lecteur.
//
// Confiance : l'index porte une signature Ed25519 (engines-nvidia.lock.sig) de la clé des mises à jour,
// posée par la CI de release (A26-02) ; sans elle, rien n'est cru. Il dit aussi l'étiquette de sa release
// (tag = "v0.12.0") et n'est cru que dans celle-là (B51-06). Il n'est cru ensuite que pour des archives
// du dépôt des releases (les paquets TensorRT vivent dans trt-<empreinte> et trt-win-<empreinte>) et des
// fichiers de pypi.nvidia.com (comme l'engines.lock livré) ; chaque archive se vérifie enfin par sa taille et
// son SHA-256, comme avant.
#pragma once

#include <filesystem>
#include <stop_token>
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

// L'étiquette que l'index dit servir (« tag = "v0.12.0" », écrite par nvidia-index.sh avant la signature) ;
// vide s'il n'en dit pas.
std::string release_index_tag(std::string_view text);

// Vrai si signature (hexadécimal, fin de ligne tolérée) signe text pour l'une des clés.
bool release_index_signed(std::string_view text, std::string_view signature,
                          const std::vector<std::string>& keys);

// Ajoute au catalogue les moteurs de text (un engines.lock) pour cet OS dont l'archive est sous prefix ; un
// moteur déjà au catalogue est remplacé, ses fichiers NVIDIA compris (le paquet de la release est le plus
// récent). Rend le nombre de moteurs pris ; une entrée hors de la release est écartée et dite dans problem.
int merge_release_index(EngineCatalog& catalog, std::string_view text, std::string_view prefix);

// Faux tant qu'une copie gardée a moins d'un jour, qu'un 404 a moins d'un jour, ou qu'un autre échec a moins
// de quelques heures : rien à demander au réseau.
bool release_index_due(const std::filesystem::path& cache);
// Relit engines-nvidia.lock et sa signature de la release si c'est dû (release_index_due), en quelques
// secondes au plus, dans cache et cache.sig ; hors du thread de la fenêtre (A18-02). stop coupe curl.
void refresh_release_index(const std::filesystem::path& cache, std::string_view repo,
                           std::string_view version, const UpdateFetch& fetch, const std::stop_token& stop);
// La copie gardée, lue d'un coup : l'index et sa signature vont ensemble.
struct ReleaseIndexCopy {
  std::string text;
  std::string signature;
};
ReleaseIndexCopy read_release_index(const std::filesystem::path& cache);
// Ajoute au catalogue les moteurs d'une copie lue, si elle est signée par l'une des keys et porte l'étiquette
// de la release de cette version (« v<version> »).
void merge_release_copy(EngineCatalog& catalog, const ReleaseIndexCopy& copy, std::string_view repo,
                        const std::vector<std::string>& keys, std::string_view version);

// refresh_release_index puis merge_release_copy, sur le thread de l'appelant (essais) : relit
// engines-nvidia.lock et sa signature de la release (au plus une fois par jour, en quelques secondes au plus,
// gardés dans cache et cache.sig), puis l'ajoute au catalogue s'il est signé par l'une des keys. Sans réseau,
// la copie gardée sert, revérifiée.
void add_release_engines(EngineCatalog& catalog, const std::filesystem::path& cache, std::string_view repo,
                         std::string_view version, const std::vector<std::string>& keys,
                         const UpdateFetch& fetch);

}  // namespace ajp::app::play
