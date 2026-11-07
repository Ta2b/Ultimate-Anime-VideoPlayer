// Calage de la cadence sur un écran fixe : la sortie doit tenir un nombre entier k de vsyncs
// par image, sinon le motif 1-2 (x4 à 144 Hz) saccade. Le facteur visé vaut
// rafraîchissement / (k · fps source) :
//   - une petite fraction (dénominateur <= 4) s'il en existe une à 0,125 % près,
//     la vitesse de lecture rattrapant l'écart : x5 ou 5/3 à 120 Hz, 5/2 à 60 Hz,
//     x3 ou x6 à 144 Hz, tous à la vitesse 1,001 pour 23,976 im/s ;
//   - sinon la fraction exacte : 11011/3200 à 165 Hz pour k = 2 (82,5 im/s).
// core/steady_rate.h choisit k ; ce fichier donne le facteur et la vitesse.
#pragma once

#include <cstdint>
#include <optional>

#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

namespace pacing {
// L'audio suit la vitesse par rééchantillonnage, borné à 0,125 %.
inline constexpr double kMaxSpeedChange = 0.00125;
// Une petite fraction garde des instants qui se répètent d'une paire à l'autre.
inline constexpr std::int64_t kMaxSimpleDen = 4;
// Une fraction exacte doit tenir à 1e-5 près : une image d'écart toutes les dix minutes à 165 Hz.
inline constexpr double kExactTolerance = 1e-5;
inline constexpr int kMaxVsyncsPerFrame = 16;
}  // namespace pacing

// La vitesse de lecture qui fait tenir à chaque image de output_fps un nombre entier de vsyncs
// d'un écran fixe à refresh_hz, à kMaxSpeedChange près ; 1 sans calage possible (sortie plus
// rapide que l'écran, rapport trop loin d'un entier, valeur inconnue).
double vsync_speed(double refresh_hz, double output_fps);

// Le facteur qui cale la cible : une petite fraction à pacing::kMaxSpeedChange près, sinon la fraction
// exacte, ou rien (cible ≤ 1).
std::optional<Ratio> lock_factor(double target);

}  // namespace ajp::core
