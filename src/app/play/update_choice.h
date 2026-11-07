// Les choix des mises à jour du lecteur (P5.29) : des énumérations, lues et écrites par leur nom dans
// play.toml ([updates] mode, channel, frequency), plus des chaînes comparées partout.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace ajp::app::play {

enum class UpdateMode : std::uint8_t { Notify, Auto, Never };  // « Me prévenir » est le défaut
enum class UpdateChannel : std::uint8_t { Stable, Preview };   // le manifeste de « latest » ou de « preview »
enum class UpdateFrequency : std::uint8_t { Launch, Daily, Weekly };

inline constexpr std::array<std::string_view, 3> kUpdateModeNames{"notify", "auto", "never"};
inline constexpr std::array<std::string_view, 2> kUpdateChannelNames{"stable", "preview"};
inline constexpr std::array<std::string_view, 3> kUpdateFrequencyNames{"launch", "daily", "weekly"};

constexpr std::string_view name(UpdateMode v) {
  return kUpdateModeNames.at(static_cast<std::size_t>(v));
}
constexpr std::string_view name(UpdateChannel v) {
  return kUpdateChannelNames.at(static_cast<std::size_t>(v));
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
