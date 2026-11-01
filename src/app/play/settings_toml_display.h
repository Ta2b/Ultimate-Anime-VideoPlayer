// Ce que play.toml garde par écran, sous [hdr] (settings_toml.cpp) : le grade adaptatif calibré
// (display_grades), sous la clé de l'écran (display_edid.h : display_key). Le pic, lui, vient du système
// (system_peak.h) ou du réglage manuel.
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

void read_display_tables(const toml::table& hdr, PlaySettings& s,
                         std::vector<std::expected<void, Refusal>>& reads);
void write_display_tables(const PlaySettings& s, toml::table& hdr);

// La lecture lâche de play.toml (settings_toml.cpp) quand un refus ne nomme pas une clé du fichier : écarte
// de root la première clé (sinon la première section) dont le retrait lève ce refus, et le note dans dropped
// ; faux si aucune. Une nouvelle version du lecteur ne remet pas tout le fichier aux défauts pour une valeur
// qu'elle n'accepte plus.
using TableReader = std::expected<PlaySettings, Refusal> (*)(const toml::table&);
bool drop_offending(toml::table& root, const std::string& detail, std::vector<std::string>* dropped,
                    const TableReader& read);

}  // namespace ajp::app::play
