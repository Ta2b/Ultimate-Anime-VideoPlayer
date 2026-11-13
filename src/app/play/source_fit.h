// La chaîne selon la source : l'interpolation ne tourne qu'à une cadence d'anime. Les deux autres règles de
// la source vivent ailleurs : pas de grade sur une source déjà HDR (render::decide_grade), sortie jamais
// au-dessus de la 4K (chain_work_size, options.h).
#pragma once

#include <array>
#include <cmath>

#include "app/play/settings.h"

namespace ajp::app::play {

// Les cadences d'un anime : 23,976 et 24 (la plupart des séries et des films), 25 (PAL), 29,97 et 30 (NTSC,
// télécinéma). Une autre cadence (50, 59,94, 60 : jeu, captation, vidéo déjà interpolée ; 12 ou 15 : extrait
// ou GIF) n'est pas interpolée : RIFE y inventerait des images entre des images qui n'en manquent pas.
inline constexpr std::array<double, 5> kAnimeRates{24000.0 / 1001.0, 24.0, 25.0, 30000.0 / 1001.0, 30.0};
// L'écart toléré, relatif : une cadence moyenne lue sur un MKV (23,975, 29,971) reste reconnue, 25 et 24 ne
// se confondent pas.
inline constexpr double kAnimeRateTolerance = 0.005;

// Vrai : la source est à une cadence d'anime, ou sa cadence est inconnue (0 : rien ne change pour elle).
inline bool anime_rate(double fps) {
  if (!(fps > 0.0)) return true;
  for (const double rate : kAnimeRates) {
    if (std::abs(fps - rate) <= rate * kAnimeRateTolerance) return true;
  }
  return false;
}

// L'interpolation demandée et coupée pour cette source : sa cadence n'est pas celle d'un anime.
inline bool motion_off_for_rate(const PlaySettings& s, double fps) {
  return s.rife_factor >= 2 && !anime_rate(fps);
}

}  // namespace ajp::app::play
