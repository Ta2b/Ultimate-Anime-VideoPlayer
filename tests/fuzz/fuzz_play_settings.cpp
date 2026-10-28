// play.toml (réglages du lecteur) : texte quelconque, accepté ou refusé sans plantage ; des
// réglages acceptés sont dans leurs plages, se réécrivent, et la réécriture se relit à l'identique.
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "app/play/settings.h"
#include "fuzz_check.h"
#include "fuzz_targets.h"

int ajp::fuzz::play_settings(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  const auto settings = ajp::app::play::parse_settings(text);
  if (!settings) return 0;
  ajp::fuzz::require(ajp::app::play::validate(*settings).has_value(),
                     "des réglages lus sont dans leurs plages");
  const auto again = ajp::app::play::parse_settings(ajp::app::play::format_settings(*settings));
  ajp::fuzz::require(again.has_value(), "la réécriture se relit");
  ajp::fuzz::require(*again == *settings, "la réécriture garde les réglages");
  return 0;
}
