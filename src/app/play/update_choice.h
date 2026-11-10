// Les choix des mises à jour du lecteur (P5.29) : des énumérations, lues et écrites par leur nom dans
// play.toml ([updates] mode, frequency), plus des chaînes comparées partout. La mise à jour n'est plus qu'une
// notification et le canal Préversions est retiré : un ancien « auto » se lit « notify », et la clé channel
// est ignorée (settings_toml.cpp).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ajp::app::play {

enum class UpdateMode : std::uint8_t { Notify, Never };  // « Me prévenir » est le défaut
enum class UpdateFrequency : std::uint8_t { Launch, Daily, Weekly };

inline constexpr std::array<std::string_view, 2> kUpdateModeNames{"notify", "never"};
inline constexpr std::array<std::string_view, 3> kUpdateFrequencyNames{"launch", "daily", "weekly"};

constexpr std::string_view name(UpdateMode v) {
  return kUpdateModeNames.at(static_cast<std::size_t>(v));
}
constexpr std::string_view name(UpdateFrequency v) {
  return kUpdateFrequencyNames.at(static_cast<std::size_t>(v));
}

// Le choix nommé text ; std::nullopt pour un nom inconnu.
template <typename Choice, std::size_t N>
constexpr std::optional<Choice> choice_named(const std::array<std::string_view, N>& names,
                                             std::string_view text) {
  for (std::size_t i = 0; i < N; ++i)
    if (names[i] == text) return static_cast<Choice>(i);
  return std::nullopt;
}

}  // namespace ajp::app::play
