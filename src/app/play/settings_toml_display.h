// Ce que play.toml garde par écran, sous [hdr] (settings_toml.cpp) : le pic calibré (display_peaks) et le
// grade adaptatif calibré (display_grades), chacun sous la clé de l'écran (display_edid.h : display_key).
#pragma once

#include <expected>
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

}  // namespace ajp::app::play
