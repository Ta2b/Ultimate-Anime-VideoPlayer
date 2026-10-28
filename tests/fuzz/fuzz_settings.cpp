// settings.toml (réglages de l'application) : texte quelconque, accepté ou refusé sans plantage ;
// des réglages acceptés se réécrivent, et la réécriture se relit à l'identique.
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "core/settings.h"
#include "fuzz_check.h"
#include "fuzz_targets.h"

int ajp::fuzz::settings(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  const auto settings = ajp::core::parse_settings(text);
  if (!settings) return 0;
  const auto written = ajp::core::format_settings(*settings);
  ajp::fuzz::require(written.has_value(), "des réglages lus se réécrivent");
  const auto again = ajp::core::parse_settings(*written);
  ajp::fuzz::require(again.has_value(), "la réécriture se relit");
  ajp::fuzz::require(*again == *settings, "la réécriture garde les réglages");
  return 0;
}
