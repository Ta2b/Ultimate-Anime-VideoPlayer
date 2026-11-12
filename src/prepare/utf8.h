// Chemins et textes en UTF-8, quel que soit le système.
//
// Sous Windows, path::string() passe par la page de code ANSI et lève une
// exception sur un caractère qu'elle ne sait pas écrire, et path(std::string)
// lit ses octets dans cette même page. Le module ne passe donc que par ces
// fonctions pour écrire un chemin dans un message ou en lire un depuis l'ONNX.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace ajp::prepare {

// Le chemin en UTF-8, pour les messages, les journaux et les lignes de commande.
std::string utf8_path(const std::filesystem::path& path);

// Un chemin écrit en UTF-8 (emplacement de poids ONNX, argument de test…).
std::filesystem::path path_from_utf8(std::string_view text);

// Le chemin à mettre sur la ligne de commande d'un programme qui la relit dans la page de code ANSI de
// Windows (trtexec, tar, curl : main(argc, argv) et non wmain). Un caractère que la page ne sait pas écrire
// (cyrillique, japonais sous une locale française) y devient « ? » : le chemin prend alors son nom court
// (8.3, GetShortPathNameW) sur la partie qui existe déjà, le nom de fichier à créer gardé tel quel. Sans nom
// court (désactivé sur le volume), le chemin reste celui d'utf8_path (ansi_command_path dit alors qu'il ne
// passera pas). La page testée est celle du SYSTÈME, pas CP_ACP du processus (UTF-8 par manifeste). Ailleurs
// : utf8_path.
std::string ansi_safe_path(const std::filesystem::path& path);

// Comme ansi_safe_path, mais rend nullopt quand aucune forme sûre n'existe (nom hors page ANSI et pas de nom
// court sur le volume) : l'appelant le sait avant de lancer le programme et peut le dire (B05-06). Toujours
// une valeur hors de Windows.
std::optional<std::string> ansi_command_path(const std::filesystem::path& path);

// Ce que la page ANSI du système dit d'un texte sans interroger Windows : vrai (représentable), faux (non),
// ou nullopt quand seul WideCharToMultiByte sur cette page peut le dire. Page 65001 (« Utiliser Unicode UTF-8
// ») : tout s'écrit. Page 0 (lecture de la page système échouée) : un texte entièrement ASCII passe, et rien
// d'autre ne se prouve, donc le nom court (C04-07). Portable pour être testé hors de Windows.
std::optional<bool> ansi_page_decides(std::uint32_t page, std::u16string_view text);
// UTF-8 bien formé : ni octet isolé, ni forme trop longue, ni moitié de paire UTF-16.
bool is_utf8(std::string_view text);

}  // namespace ajp::prepare
