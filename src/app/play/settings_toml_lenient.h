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

// Lecture lâche de root (parse_settings_lenient) : les valeurs refusées en sont ôtées, root garde le reste.
std::expected<PlaySettings, Refusal> read_settings_lenient(toml::table& root,
                                                           std::vector<std::string>* dropped);

// A22-06, A36-01 : les valeurs de existing que la lecture lâche a écartées (pruned : ce qu'il en reste),
// une valeur nouvelle d'une clé connue, repassent telles quelles dans fresh, tant que fresh garde à leur
// place ce que la lecture lâche en avait fait (read_back : les réglages lus, mis en forme). Un retour à une
// version plus ancienne ne remet plus aux défauts un moteur ou un préréglage de la plus récente.
void keep_rejected_values(toml::table& fresh, const toml::table& existing, const toml::table& pruned,
                          const toml::table& read_back);

// P2.14 : les clés que ce processus a changées depuis before (ce qu'il a lu ou écrit) jusqu'à now passent
// dans disk, le fichier tel qu'un autre processus a pu le réécrire entre-temps ; ses autres clés restent.
// Une clé de before absente de now est retirée de disk. Les tableaux comptent comme une valeur.
void apply_own_changes(toml::table& disk, const toml::table& before, const toml::table& now);

}  // namespace ajp::app::play
