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
  std::vector<core::EnginePackage> packages;  // ceux de cet OS
  std::optional<core::NgcSdkSource> maxine;   // SDK de RTX HDR sous Linux (maxine_install.h)
  std::string problem;                        // index absent ou illisible, pour le journal
};

// L'index index lu pour cet OS, installé sous data_dir ; vide (problem dit pourquoi) s'il manque.
EngineCatalog load_catalog(const std::filesystem::path& index, const std::filesystem::path& data_dir);

const core::EnginePackage* package_for(const EngineCatalog& catalog, core::EngineId engine);
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

// Les bibliothèques des backends installés, par préférence : TensorRT, MIGraphX, OpenVINO.
std::vector<std::filesystem::path> installed_backends(const EngineCatalog& catalog);

// Ce qui manque au système pour que le moteur tourne, que le lecteur ne peut pas installer seul
// (pilote ROCm, pilote compute Intel, Windows 11 24H2 pour MIGraphX de Windows ML), avec la commande
// à lancer ; rien s'il ne manque rien.
std::optional<std::string> missing_prerequisite(core::EngineId engine);

struct InstallProgress {
  std::function<void(float fraction, const std::string& stage)> report;
  std::stop_token stop;
};

// Télécharge (en reprenant un .part), vérifie et extrait le paquet ; rend la bibliothèque du
// backend (le dossier extrait pour RTX HDR, qui n'a pas de module). Refus : LibraryUnavailable (curl ou tar
// absents), DecodeFailed (transfert), HashMismatch, FileUnwritable, Cancelled.
std::expected<std::filesystem::path, Refusal> install_package(const EngineCatalog& catalog,
                                                              const core::EnginePackage& package,
                                                              const InstallProgress& progress);

}  // namespace ajp::app::play
