// Cadence d'une piste ramenée à une cadence de diffusion usuelle.
#pragma once

#include <array>
#include <cstdint>

#include "media/track.h"

namespace ajp::media {

// La cadence standard la plus proche de `rate` à 0,1 % près, sinon `rate` telle quelle. Un MPEG-TS ou un MKV
// sans durée par défaut rend une moyenne estimée (2997/100, 29970/1000, 23,9761) qui ne vaut pas le rationnel
// exact 30000/1001 : sans cette étape, ni le télécinéma ni les niveaux d'interpolation ne la reconnaîtraient
// (C14-07).
constexpr Rational standard_frame_rate(Rational rate) {
  if (rate.num <= 0 || rate.den <= 0) return rate;
  constexpr std::array<Rational, 10> kStandard = {{{24000, 1001},
                                                   {24, 1},
                                                   {25, 1},
                                                   {30000, 1001},
                                                   {30, 1},
                                                   {50, 1},
                                                   {60000, 1001},
                                                   {60, 1},
                                                   {120000, 1001},
                                                   {120, 1}}};
  // Écart relatif au carré près, en entiers : |r - s| / s, avec r = n/d et s = a/b.
  const auto relative_gap_ppm = [&](Rational s) {
    const std::int64_t lhs = std::int64_t{rate.num} * s.den;  // r * (d*b)
    const std::int64_t rhs = std::int64_t{s.num} * rate.den;  // s * (d*b)
    const std::int64_t gap = lhs > rhs ? lhs - rhs : rhs - lhs;
    return gap * 1'000'000 / rhs;
  };
  Rational best = rate;
  std::int64_t best_gap = 1'001;  // 0,1 % : au-delà, la cadence reste telle quelle
  for (const Rational s : kStandard) {
    const std::int64_t gap = relative_gap_ppm(s);
    if (gap < best_gap) {
      best_gap = gap;
      best = s;
    }
  }
  return best;
}

}  // namespace ajp::media
