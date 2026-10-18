// Horodatages : le lecteur compte en microsecondes, libass en millisecondes.
#pragma once

#include <cstdint>

namespace ajp::subs {

// Bornes des horodatages et des durées acceptés, en microsecondes.
inline constexpr std::int64_t kMaxTimestampUs = std::int64_t{100} * 24 * 3600 * 1'000'000;  // 100 jours
inline constexpr std::int64_t kMaxEventDurationUs = std::int64_t{24} * 3600 * 1'000'000;    // 24 heures

constexpr bool timestamp_in_range(std::int64_t us) {
  return -kMaxTimestampUs <= us && us <= kMaxTimestampUs;
}

// Milliseconde la plus proche ; une demie arrondit vers +∞, pour les négatifs aussi.
constexpr std::int64_t to_ms(std::int64_t us) {
  std::int64_t ms = us / 1000;
  std::int64_t rest = us % 1000;
  if (rest < 0) {
    rest += 1000;
    --ms;
  }
  return rest >= 500 ? ms + 1 : ms;
}

}  // namespace ajp::subs
