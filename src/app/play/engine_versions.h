// Les versions installées d'un moteur à côté de celle que l'index demande (<moteurs>/<moteur>/<version>/) :
// un paquet remplacé à version égale, une version de TensorRT changée, une installation coupée en plein
// remplacement. engine_downloads.cpp (update_stale_engines) en tire ce qu'il faut réinstaller au lancement.
#pragma once

#include <filesystem>
#include <optional>

#include "app/play/engine_install.h"

namespace ajp::app::play {

// La note du paquet (kPackageNote) d'un dossier de version ; rien si elle manque ou n'a pas de SHA-256.
std::optional<InstalledPackage> read_package_note(const std::filesystem::path& dir);

enum class PackageState {
  Absent,      // ni cette version ni une autre
  Current,     // cette version, le paquet de l'index (même SHA-256), tous ses fichiers là
  Replaced,    // cette version, un autre paquet (SHA-256 changé, note absente) ou un fichier manquant : à
               // réinstaller (A26-01, B06-11)
  NewVersion,  // seule une autre version est installée : l'index en demande une nouvelle (A06-01)
};

struct InstalledState {
  PackageState state = PackageState::Absent;
  std::filesystem::path dir;             // le dossier installé (cette version ou l'autre)
  std::optional<InstalledPackage> note;  // sa note, si elle existe
  std::string missing;  // un fichier tiers de la note absent ou de taille inattendue (Replaced)
};

// Ce qui est installé pour ce paquet de l'index. arch (C37-02) : l'architecture de la carte du rendu
// (effective_cuda_arch, « sm89 ») ; un fichier tiers qu'elle demande (la ressource de construction
// nvinfer_builder_resource_sm89) et que la note n'a pas fait rendre Replaced, missing = son nom : la carte a
// changé depuis l'installation (autre carte, autre poste, autre carte NVIDIA choisie), et trtexec n'aurait
// que la ressource générique. Vide : pas de vérification.
InstalledState installed_state(const EngineCatalog& catalog, const core::EnginePackage& package,
                               std::string_view arch = {});

// Une autre version installée de ce moteur (un dossier frère qui a le module ou la note) ; la plus récente
// par date de modification s'il y en a plusieurs.
std::optional<std::filesystem::path> other_installed_version(const EngineCatalog& catalog,
                                                             const core::EnginePackage& package);

// Retire les autres dossiers de version de ce moteur, une fois la version de l'index en place (A06-01,
// A36-06). Rend le nombre de dossiers retirés.
int remove_other_versions(const EngineCatalog& catalog, const core::EnginePackage& package);

// Une installation coupée entre ses deux renommages (A06-08) : la version manque, mais « .partiel » est
// complet (il a sa note, écrite en dernier) ou « .ancien » garde la précédente. Remet en place la copie
// complète, la nouvelle d'abord. Si la version est là avec un « .partiel » complet, le remplacement que
// Windows avait refusé (une DLL encore ouverte, A06-09) se fait maintenant, avant tout chargement. Rend vrai
// si une copie a été remise.
bool recover_interrupted_install(const EngineCatalog& catalog, const core::EnginePackage& package);

}  // namespace ajp::app::play
