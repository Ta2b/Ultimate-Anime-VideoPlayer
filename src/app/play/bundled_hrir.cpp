#include "app/play/bundled_hrir.h"

#include <array>

namespace ajp::app::play {

std::span<const BundledHrir> bundled_hrirs() {
  static const std::array<BundledHrir, 5> kAll{{
      {"atmos", "Dolby Atmos", kHrirAtmos},
      {"dtshx", "DTS Headphone:X", kHrirDtshx},
      {"sonic", "Windows Sonic", kHrirSonic},
      {"sbx100", "Creative SBX (100 %)", kHrirSbx100},
      {"sbx33", "Creative SBX (33 %)", kHrirSbx33},
  }};
  return kAll;
}

const BundledHrir* find_bundled_hrir(std::string_view hrir_path) {
  constexpr std::string_view kPrefix = "bundled:";
  if (!hrir_path.starts_with(kPrefix)) return nullptr;
  const std::string_view key = hrir_path.substr(kPrefix.size());
  for (const BundledHrir& entry : bundled_hrirs())
    if (entry.key == key) return &entry;
  return nullptr;
}

}  // namespace ajp::app::play
