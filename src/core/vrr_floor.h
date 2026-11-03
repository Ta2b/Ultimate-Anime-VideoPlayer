// Le plancher d'un écran à fréquence variable (VRR) : sous sa fréquence minimale, l'écran répète lui-même
// l'image, à son rythme et pas au nôtre. Mieux vaut représenter l'image tenue à intervalles égaux, tous
// au-dessus du plancher, que de la laisser durer puis rattraper (18,9 ms puis 2 ms).
#pragma once

#include <cstdint>

namespace ajp::core {

// L'intervalle entre deux présentations d'une image tenue hold_us, sans jamais dépasser floor_us :
// hold / ceil(hold / floor). Au plancher ou en dessous, ou sans plancher (floor_us ≤ 0), l'image ne se
// représente pas : hold_us est rendu tel quel.
constexpr std::int64_t floor_step_us(std::int64_t hold_us, std::int64_t floor_us) {
  if (hold_us <= 0 || floor_us <= 0 || hold_us <= floor_us) return hold_us;
  const std::int64_t parts = (hold_us + floor_us - 1) / floor_us;
  return hold_us / parts;
}

// Le plancher en µs d'un écran dont la fréquence minimale est min_hz, avec 10 % de marge ; 0 si inconnue.
constexpr std::int64_t vrr_floor_us(int min_hz) {
  if (min_hz <= 0) return 0;
  return static_cast<std::int64_t>(1'000'000.0 / (static_cast<double>(min_hz) * 1.1));
}

}  // namespace ajp::core
