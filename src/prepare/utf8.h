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

// UTF-8 bien formé : ni octet isolé, ni forme trop longue, ni moitié de paire UTF-16.
bool is_utf8(std::string_view text);

}  // namespace ajp::prepare
