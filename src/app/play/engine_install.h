// Moteurs installés à la demande : le lecteur livre Vulkan et un index haché (engines.lock, à
// côté de l'exécutable) des paquets de ses autres backends pour cet OS ; l'écran de bienvenue
// propose ceux qui conviennent au GPU, et les cochés se téléchargent, se vérifient par leur
// taille et leur SHA-256, puis s'extraient sous <données>/backends/<moteur>/<version>/.
//
// curl et tar du système font le transfert et l'extraction (posix_spawn, sans shell ni PATH) :
// aucun client HTTP dans le lecteur. Une licence (TensorRT) s'accepte à l'écran avant tout
// téléchargement, jamais d'office.
#pragma once

#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "core/engines.h"
#include "core/model_index.h"
#include "core/types.h"

namespace ajp::app::play {

core::Os host_os();

struct EngineCatalog {
  core::InstallLayout layout;
  std::vector<core::EnginePackage> packages;   // ceux de cet OS
  std::vector<core::VendorFile> vendor_files;  // fichiers tirés des archives de NVIDIA, de cet OS
  std::string problem;                         // index absent ou illisible, pour le journal
};

// L'index index lu pour cet OS, installé sous data_dir ; vide (problem dit pourquoi) s'il manque.
// engines_dir (réglage « Dossier d'installation des moteurs ») : les moteurs et leurs téléchargements y vont
// au lieu de data_dir ; les modèles restent sous data_dir.
EngineCatalog load_catalog(const std::filesystem::path& index, const std::filesystem::path& data_dir,
                           const std::filesystem::path& engines_dir = {});

const core::EnginePackage* package_for(const EngineCatalog& catalog, core::EngineId engine);

// Ce qu'une installation demande au disque, pour l'afficher AVANT de télécharger et refuser proprement si la
// place manque. Les téléchargements ne sont qu'une partie : les fichiers de NVIDIA se décompressent (nvinfer
// : 306 Mo tirés, 663 Mo écrits), et l'ancienne version reste en place jusqu'à ce que la nouvelle soit
// entière.
struct InstallSpace {
  std::uint64_t download = 0;   // octets tirés : l'archive du moteur et les fichiers tiers de la carte
  std::uint64_t installed = 0;  // octets du dossier final
  std::uint64_t peak = 0;       // le plus haut, temporaires comprises, avec l'ancienne version et une marge
};
// arch : l'architecture du GPU (« sm89 ») pour les fichiers tiers ; existing : octets déjà installés dans le
// dossier de ce moteur (gardés jusqu'à la fin). Le pic suit l'ordre d'installation : fichiers tiers un à un,
// chacun avec son téléchargement et sa copie. engines_root (le dossier des versions du moteur) : un fichier
// tiers de même nom et de même taille qui s'y trouve sera repris par un lien dur, il ne compte ni au
// téléchargement ni au pic (A06-06).
InstallSpace install_space(const EngineCatalog& catalog, const core::EnginePackage& package,
                           std::string_view arch, std::uint64_t existing = 0,
                           const std::filesystem::path& engines_root = {});
// Pour le moteur installé (ou à installer) dans installed_dir : ses octets, et ses voisins pour les reprises.
InstallSpace install_space_in(const EngineCatalog& catalog, const core::EnginePackage& package,
                              std::string_view arch, const std::filesystem::path& installed_dir);
// Refus InsufficientSpace, avec les chiffres, si available est sous space.peak.
std::expected<void, Refusal> check_space(const InstallSpace& space, std::uint64_t available,
                                         const std::filesystem::path& where);
// « 1,3 Go », « 742 Mo ».
std::string format_size(std::uint64_t bytes);
// La même taille pour l'interface, dans sa langue (« 1.5 GB », « 1,5 Go » ; A25-12).
std::string shown_size(std::uint64_t bytes);
// L'espace libre là où le moteur s'installe (le dossier qui existe le plus près de layout.engines).
std::uint64_t free_bytes(const std::filesystem::path& where);
// Octets d'un dossier, récursivement ; 0 s'il manque.
std::uint64_t directory_bytes(const std::filesystem::path& dir);
bool installed(const EngineCatalog& catalog, const core::EnginePackage& package);

// Le paquet extrait dans le dossier de version du moteur, d'après kPackageNote qu'install_package
// y écrit : son SHA-256 et la licence acceptée pour lui. Rien pour un moteur installé avant cette
// note : le dossier porte la version du runtime (« 11 » pour TensorRT), pas celle du lecteur, et
// un module d'une version précédente du lecteur y reste après une mise à jour.
inline constexpr std::string_view kPackageNote = "paquet.txt";
struct InstalledPackage {
  std::string sha256;
  std::string license;
  friend bool operator==(const InstalledPackage&, const InstalledPackage&) = default;
};
std::optional<InstalledPackage> installed_package(const EngineCatalog& catalog,
                                                  const core::EnginePackage& package);

// La bibliothèque du backend TensorRT, s'il est installé.
std::vector<std::filesystem::path> installed_backends(const EngineCatalog& catalog);

struct InstallProgress {
  std::function<void(float fraction, const std::string& stage)> report;
  std::stop_token stop;
  // L'architecture du GPU (« sm89 ») pour les fichiers tiers qui en dépendent ; vide : ceux de
  // toutes les architectures (detect_cuda_arch rend vide quand le GPU ne se lit pas).
  std::string cuda_arch;
};

// « sm89 » pour une carte de capacité 8.9, lu par nvidia-smi une fois par processus ; vide s'il manque ou se
// tait. Depuis le thread de la fenêtre, par run_pumped (pumped_task.h) : le premier appel attend nvidia-smi.
// gpu_name : la carte de rendu, choisie parmi celles de nvidia-smi ; la première si aucune ne porte ce nom.
std::string detect_cuda_arch(std::string_view gpu_name = {});
// L'architecture déduite du nom de la carte (« NVIDIA GeForce RTX 4050 Laptop GPU » : sm89) ; vide si
// inconnue.
std::string cuda_arch_from_name(std::string_view gpu_name);
// Présomption de l'accueil, sans moteur chargé, avec la règle du moteur (A37-02) : le FP8 de TensorRT dès
// sm 8.9 (Ada, Hopper, Blackwell), les noyaux CUTLASS de SPAN sur Ada seulement
// (SpanFp8Net::supported_device).
bool fp8_presumed(std::string_view gpu_name);
bool span_fp8_presumed(std::string_view gpu_name);
// Celle qu'on installe : lue par nvidia-smi, sinon d'après le nom, sinon « ptx » (aucun fichier propre à une
// architecture). Jamais vide : une architecture inconnue ne fait plus tirer celles de toutes les cartes
// (0.5.0 : 2 Go tirés, 3,3 écrits).
std::string effective_cuda_arch(std::string_view detected, std::string_view gpu_name);

// Les octets que l'installation transfère : le paquet, plus les fichiers tiers de l'architecture
// du GPU (`arch`, vide : toutes). Ce que l'écran des moteurs annonce avant le téléchargement.
std::uint64_t download_bytes(const EngineCatalog& catalog, const core::EnginePackage& package,
                             std::string_view arch);

// Un fichier de VendorFile : la plage de ses octets dans l'archive (curl --range), décompressée
// vers `dest`, taille et SHA-256 vérifiés. Rien n'est laissé à `dest` sur échec. Refus : ceux de
// install_package.
// P3.8 : un fichier tiers de même nom, taille et SHA-256 dans un autre dossier de engines_root (une autre
// version installée du moteur) est lié en dur à dest (copié si le lien échoue) ; vrai s'il a servi.
// Retire de downloads/ ce que le catalogue ne nomme plus (« <sha256>.part », « <sha256>.morceau ») : un
// téléchargement coupé d'une version remplacée (A06-04). keep (le paquet en cours d'installation) reste
// toujours. Rend le nombre d'entrées retirées.
int purge_stale_downloads(const EngineCatalog& catalog, std::string_view keep = {});
bool reuse_vendor_file(const core::VendorFile& file, const std::filesystem::path& dest,
                       const std::filesystem::path& engines_root, const std::stop_token& stop);
std::expected<void, Refusal> fetch_vendor_file(const core::VendorFile& file,
                                               const std::filesystem::path& dest,
                                               const std::filesystem::path& scratch,
                                               const InstallProgress& progress);

// Télécharge (en reprenant un .part), vérifie et extrait le paquet ; rend la bibliothèque du
// backend. Refus : LibraryUnavailable (curl ou tar
// absents), DecodeFailed (transfert), HashMismatch, FileUnwritable, Cancelled.
std::expected<std::filesystem::path, Refusal> install_package(const EngineCatalog& catalog,
                                                              const core::EnginePackage& package,
                                                              const InstallProgress& progress);

}  // namespace ajp::app::play
