// La lecture lâche de play.toml (settings_toml.cpp : parse_settings_lenient).
#pragma once

#include <expected>
#include <string>
#include <vector>

#include "app/play/settings.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {

// La lecture lâche de play.toml (settings_toml.cpp) quand un refus ne nomme pas une clé du fichier : écarte
// de root la première clé (sinon la première section) dont le retrait lève ce refus, et le note dans dropped
// ; faux si aucune. Une nouvelle version du lecteur ne remet pas tout le fichier aux défauts pour une valeur
// qu'elle n'accepte plus.
using TableReader = std::expected<PlaySettings, Refusal> (*)(const toml::table&);
bool drop_offending(toml::table& root, const Refusal& refusal, std::vector<std::string>* dropped,
                    const TableReader& read);

// P2.22 : les clés de existing que read ne lit pas (celles d'une version plus récente du lecteur) passent
// dans fresh, la table que le lecteur va écrire. Une clé est gardée quand l'ôter de existing ne change rien à
// ce que read en tire, et que fresh ne l'a pas déjà. Rien n'est gardé si existing ne se lit pas.
void keep_unknown_keys(toml::table& fresh, const toml::table& existing, const TableReader& read);

// P2.14 : les clés que ce processus a changées depuis before (ce qu'il a lu ou écrit) jusqu'à now passent
// dans disk, le fichier tel qu'un autre processus a pu le réécrire entre-temps ; ses autres clés restent.
// Une clé de before absente de now est retirée de disk. Les tableaux comptent comme une valeur.
void apply_own_changes(toml::table& disk, const toml::table& before, const toml::table& now);

}  // namespace ajp::app::play
