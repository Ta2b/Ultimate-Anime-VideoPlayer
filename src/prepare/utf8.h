// Chemins et textes en UTF-8, quel que soit le système.
//
// Sous Windows, path::string() passe par la page de code ANSI et lève une
// exception sur un caractère qu'elle ne sait pas écrire, et path(std::string)
// lit ses octets dans cette même page. Le module ne passe donc que par ces
// fonctions pour écrire un chemin dans un message ou en lire un depuis l'ONNX.
#pragma once

#include <filesystem>
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
// court (désactivé sur le volume), le chemin reste celui d'utf8_path (A05-11, A06-02, A20-02). Ailleurs :
// utf8_path.
std::string ansi_safe_path(const std::filesystem::path& path);

// UTF-8 bien formé : ni octet isolé, ni forme trop longue, ni moitié de paire UTF-16.
bool is_utf8(std::string_view text);

}  // namespace ajp::prepare
