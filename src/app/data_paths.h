// Dossiers du lecteur : ceux de l'utilisateur, où il écrit, et ceux livrés
// avec lui, en lecture seule. Déterminés une seule fois au démarrage, par SDL3
// ou par --data-dir, puis passés en paramètre partout : aucun module ne lit
// l'environnement ni ne devine un chemin.
//
//   <données>/settings.toml, user.toml, receipts.toml
//   <données>/models/, engines/, downloads/   (core::InstallLayout)
//   <données>/optional-sdks/                  (vendor_hdr::SdkStore)
//   <données>/cache/                          moteurs construits par prepare
//   <ressources>/profile.toml, models.lock, engines.lock
#pragma once

#include <expected>
#include <filesystem>
#include <string_view>

#include "core/engines.h"
#include "core/model_index.h"
#include "core/types.h"

namespace ajp::app {

// Le système de cette construction.
constexpr core::Os host_os() {
#if defined(_WIN32)
  return core::Os::Windows;
#elif defined(__APPLE__)
  return core::Os::MacOs;
#else
  return core::Os::Linux;
#endif
}

struct DataPaths {
  std::filesystem::path user_data;  // absolu ; existe
  std::filesystem::path resources;  // absolu ; peut manquer dans un dossier de construction

  std::filesystem::path settings() const { return user_data / "settings.toml"; }
  std::filesystem::path user_profile() const { return user_data / "user.toml"; }
  std::filesystem::path receipts() const { return user_data / "receipts.toml"; }
  std::filesystem::path engine_cache() const { return user_data / "cache"; }
  std::filesystem::path base_profile() const { return resources / "profile.toml"; }
  std::filesystem::path models_lock() const { return resources / "models.lock"; }
  std::filesystem::path engines_lock() const { return resources / "engines.lock"; }
  core::InstallLayout layout(core::Os os) const { return core::install_layout(os, user_data); }
};

// Nom du dossier de l'application, sous le dossier des préférences du système.
inline constexpr std::string_view kAppFolder = "ajp";

// Ressources livrées, à partir du dossier de l'exécutable : à côté de lui sous
// Windows et dans un paquet macOS (SDL rend déjà Contents/Resources), dans
// ../share/ajp sous Linux.
std::filesystem::path resources_from_base(core::Os os, const std::filesystem::path& base);

// Dossier des données choisi (--data-dir ou SDL) : rendu absolu, créé s'il manque.
// Refus FileUnwritable s'il ne peut pas l'être, ou si c'est un fichier.
std::expected<DataPaths, Refusal> data_paths_in(const std::filesystem::path& user_data,
                                                const std::filesystem::path& resources);

// Dossier des préférences de SDL (SDL_GetPrefPath : ~/.local/share/ajp, %APPDATA%\ajp,
// ~/Library/Application Support/ajp) et dossier de l'exécutable (SDL_GetBasePath).
// SDL crée le dossier des préférences. Refus FileUnwritable avec le message de SDL.
std::expected<DataPaths, Refusal> system_data_paths(core::Os os);

// Ce que SDL_GetBasePath rend, pour l'aide et les tests ; vide si SDL l'ignore.
std::filesystem::path executable_dir();

// Chemin en UTF-8, pour les messages : path::string() lèverait sous Windows sur un
// caractère hors de la page de code.
std::string utf8(const std::filesystem::path& path);
std::filesystem::path path_from_utf8(std::string_view text);

}  // namespace ajp::app
