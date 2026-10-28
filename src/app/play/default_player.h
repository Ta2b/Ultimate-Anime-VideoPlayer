// Le lecteur par défaut : AJP s'inscrit auprès du système pour paraître dans « Ouvrir avec » et
// pouvoir devenir le lecteur des vidéos (« ajp --register », onglet Système des paramètres).
//
//   Linux   : un fichier .desktop dans $XDG_DATA_HOME/applications, ~/.local/share/applications
//             sans la variable (spécification XDG Desktop Entry) ; « Définir par défaut » le
//             donne à xdg-mime pour chaque type vidéo.
//   Windows : les clés du registre de l'utilisateur (HKCU) d'un ProgID, des extensions vidéo et de
//             Capabilities, que Paramètres > Applications par défaut propose. Windows ne laisse
//             aucun programme se choisir lui-même : « Définir par défaut » ouvre cette page.
//   macOS   : rien à écrire, l'Info.plist d'AJP.app déclare les vidéos qu'il ouvre
//             (tools/release/package-macos.sh) ; le Finder fait le reste.
//
// Rien n'est lu de l'environnement : les dossiers de l'utilisateur viennent de SDL, comme ailleurs.
#pragma once

#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"

namespace ajp::app::play {

// Où vit le lecteur.
struct PlayerPlace {
  std::filesystem::path executable;  // ajp ou ajp.exe, chemin absolu
  std::filesystem::path home;        // dossier de l'utilisateur (SDL_GetUserFolder)
  // Données de l'utilisateur, celles de SDL_GetPrefPath (user_data) : sous Linux $XDG_DATA_HOME,
  // ~/.local/share sans la variable. Vide : <home>/.local/share.
  std::filesystem::path data;
};

inline constexpr std::string_view kAppName = "Ultimate Anime Video Player";
inline constexpr std::string_view kDesktopId = "io.github.ta2b.ajp.desktop";
inline constexpr std::string_view kProgId = "AJP.Video";
inline constexpr std::string_view kRegisteredName = "AJP";  // HKCU\Software\RegisteredApplications

// Les types vidéo proposés au système : types MIME (Linux) et extensions (Windows).
std::span<const std::string_view> video_mime_types();
std::span<const std::string_view> video_extensions();

// Le .desktop du lecteur. Refus FieldOutOfRange pour un chemin qu'une ligne ne peut porter (saut
// de ligne, caractère de contrôle, UTF-8 mal formé).
std::expected<std::string, Refusal> desktop_entry(const std::filesystem::path& executable);
// <data>/applications/io.github.ta2b.ajp.desktop ; sans data, sous <home>/.local/share.
std::filesystem::path desktop_file(const PlayerPlace& place);

// Les données de l'utilisateur d'après le dossier de préférences de SDL (<données>/ajp/ajp/,
// SDL_GetPrefPath("ajp", "ajp")) ; vide pour un dossier vide.
std::filesystem::path user_data(std::filesystem::path pref);

// Une valeur chaîne (REG_SZ) du registre de l'utilisateur, en UTF-8 : clé sous HKEY_CURRENT_USER,
// nom (vide : la valeur par défaut de la clé), donnée.
struct RegistryValue {
  std::string key;
  std::string name;
  std::string data;
  friend bool operator==(const RegistryValue&, const RegistryValue&) = default;
};
// Ce que l'inscription écrit pour l'exécutable, chemin Windows en UTF-8 (« C:\…\ajp.exe »).
std::vector<RegistryValue> registry_values(std::string_view executable);
// Ce que le retrait efface : les clés du lecteur entières, puis ses valeurs dans les clés partagées
// (extensions, RegisteredApplications), dont le reste ne lui appartient pas.
std::vector<std::string> registry_trees(std::string_view executable);
std::vector<RegistryValue> registry_shared_values();

// Inscrit le lecteur ; rend ce qui a été fait et ce qui reste à faire soi-même.
std::expected<std::string, Refusal> register_player(const PlayerPlace& place);
// Retire l'inscription (rien à retirer n'est pas une erreur).
std::expected<std::string, Refusal> unregister_player(const PlayerPlace& place);
// Inscrit le lecteur et en fait celui des vidéos là où le système le permet (Linux : xdg-mime),
// sinon ouvre la page du système qui le fait (Windows) ou dit comment faire (macOS).
std::expected<std::string, Refusal> choose_as_default(const PlayerPlace& place);

// Windows : lancé depuis l'Explorateur (double clic, « Ouvrir avec »), le lecteur ferme la console
// que Windows a ouverte pour lui seul ; lancé d'un terminal, il la garde et l'écrit en UTF-8. Rien
// ailleurs.
void leave_lone_console();

}  // namespace ajp::app::play
