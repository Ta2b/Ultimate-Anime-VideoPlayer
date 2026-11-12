// Le refus d'une clé du fichier de réglages (P2.11), partagé par la validation (settings.cpp) et la lecture
// (settings_toml.cpp).
#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <utility>

#include "core/types.h"

namespace ajp::app::play {

// La clé, « audio.delay_ms », dans Refusal::key, que la lecture tolérante écarte sans relire le texte ; le
// détail commence par elle.
inline std::unexpected<Refusal> refuse_key(RefusalCode code, std::string key, std::string_view rest = {}) {
  std::string detail = key;
  detail += rest;
  return std::unexpected(Refusal{.code = code, .detail = std::move(detail), .key = std::move(key)});
}

}  // namespace ajp::app::play
