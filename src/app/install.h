// Installation des modèles et des paquets de moteurs, sous le dossier de données
// (core::InstallLayout) : téléchargement dans downloads/<sha256>.part, vérification
// de la taille et du SHA-256 de l'index, puis un renommage atomique. Un fichier
// installé est donc toujours complet et vérifié ; une coupure ne laisse qu'un .part,
// repris au prochain essai.
//
// Un paquet de moteur est un seul fichier, la bibliothèque du backend
// (core::backend_library) : il s'installe dans engines/<moteur>/<version>/, préparé
// à côté puis renommé d'un coup.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <vector>

#include "app/download.h"
#include "core/engines.h"
#include "core/model_index.h"
#include "core/types.h"

namespace ajp::app {

struct InstallItem {
  std::string label;  // « modèle 2x_AnimeJaNai_V3 », « moteur tensorrt 11.3.0 »
  std::string url;
  std::string sha256;
  std::uint64_t size = 0;
  std::filesystem::path target;          // fichier final
  std::optional<core::EngineId> engine;  // paquet de moteur : son dossier est target.parent_path()
};

InstallItem model_item(const core::InstallLayout& layout, const core::ModelEntry& model);
InstallItem engine_item(const core::InstallLayout& layout, const core::EnginePackage& package);

// Le fichier final existe avec la taille de l'index (le SHA-256 a été vérifié à l'installation).
bool is_installed(const InstallItem& item);

// Modèles des profils (utilisés par au moins un slot) pas encore installés, dans l'ordre de l'index.
std::vector<core::ModelEntry> missing_models(const core::InstallLayout& layout,
                                             std::span<const core::ModelEntry> lock);

// Vérifie le .part complet (taille, SHA-256) puis le met en place d'un renommage.
// Refus HashMismatch (le .part faux est supprimé : il ne peut pas se reprendre) ou FileUnwritable.
std::expected<void, Refusal> install_downloaded(const core::InstallLayout& layout, const InstallItem& item);

// Retire un moteur installé : tout engines/<moteur>/. Refus FileUnwritable.
std::expected<void, Refusal> uninstall_engine(const core::InstallLayout& layout, core::EngineId engine);

struct InstallProgress {
  std::size_t item = 0;  // index dans la liste
  std::size_t items = 0;
  std::string label;
  std::uint64_t done = 0;   // octets reçus, tous éléments confondus
  std::uint64_t total = 0;  // octets à recevoir en tout
};

// Télécharge, vérifie et installe chaque élément dans l'ordre, en sautant ceux déjà
// installés. S'arrête au premier refus : les éléments déjà installés le restent.
std::expected<void, Refusal> install_all(const Downloader& downloader, const core::InstallLayout& layout,
                                         std::span<const InstallItem> items, DownloadOptions options,
                                         const std::function<void(const InstallProgress&)>& on_progress = {},
                                         const std::stop_token& stop = {});

}  // namespace ajp::app
