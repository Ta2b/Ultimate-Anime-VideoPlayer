// Rafraîchissement nominal : le mode annoncé par le système (164,998 Hz, 59,9401 Hz) ramené à
// sa valeur nominale exacte (165, 60000/1001) avant d'entrer dans DisplayInfo. Le sélecteur de
// cadence (core/pacing.h) travaille sur des fractions : une mesure brute y donnerait des
// termes énormes (36133/10501 au lieu de 11011/3200) qui changeraient à chaque lecture, et le
// hachage du Plan avec. mpv fait de même (mp_display_rate_from_gdi).
#pragma once

#include <expected>

#include "core/profile.h"
#include "core/types.h"

namespace ajp::present {

namespace refresh {
// Écart relatif accepté vers une valeur nominale : 200 ppm. Assez large pour 164,998 Hz
// (12 ppm de 165), assez étroit pour séparer 60 Hz de 60000/1001 Hz (999 ppm).
inline constexpr double kNominalTolerance = 2e-4;
// Dénominateur d'une mesure gardée telle quelle : le millihertz du mode Wayland.
inline constexpr std::int64_t kMeasuredDen = 1000;
}  // namespace refresh

// La valeur nominale la plus proche (N ou N·1000/1001 Hz) quand elle tient dans la tolérance,
// la mesure au millihertz sinon. Refuse une fréquence hors de [1, 1000] Hz ou non finie.
std::expected<core::Ratio, Refusal> nominal_refresh(double measured_hz);

// La même chose depuis le rationnel exact du mode (SDL_DisplayMode, numérateur / dénominateur).
std::expected<core::Ratio, Refusal> nominal_refresh(core::Ratio measured);

}  // namespace ajp::present
