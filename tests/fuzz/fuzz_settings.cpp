// settings.toml (réglages de l'application) : texte quelconque, accepté ou refusé sans plantage ;
// des réglages acceptés se réécrivent, et la réécriture se relit à l'identique.
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "core/settings.h"
#include "fuzz_check.h"
#include "fuzz_targets.h"

int kirari::fuzz::settings(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  const auto settings = kirari::core::parse_settings(text);
  if (!settings) return 0;
  const auto written = kirari::core::format_settings(*settings);
  kirari::fuzz::require(written.has_value(), "des réglages lus se réécrivent");
  const auto again = kirari::core::parse_settings(*written);
  kirari::fuzz::require(again.has_value(), "la réécriture se relit");
  kirari::fuzz::require(*again == *settings, "la réécriture garde les réglages");
  return 0;
}
