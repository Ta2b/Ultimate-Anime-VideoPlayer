// Sélecteur de cadence selon l'écran : quel facteur d'interpolation, et sur
// quelle grille présenter les images.
//
// Avec VRR, chaque image passe à son horodatage et le facteur reste celui de la
// règle. Sur un écran fixe, l'écran devient maître : la sortie doit tenir un
// nombre entier k de vsyncs par image, sinon le motif 1-2 (x4 à 144 Hz) saccade.
// Le facteur visé vaut alors rafraîchissement / (k · fps source) :
//   - une petite fraction (dénominateur <= 4) s'il en existe une à 0,125 % près,
//     la vitesse de lecture rattrapant l'écart : x5 ou 5/3 à 120 Hz, 5/2 à 60 Hz,
//     x3 ou x6 à 144 Hz, tous à la vitesse 1,001 pour 23,976 im/s ;
//   - sinon la fraction exacte, t devenant fractionnaire : 11011/3200 à 165 Hz
//     pour k = 2 (82,5 im/s), 11011/4800 pour k = 3 (55 im/s).
// Parmi les k possibles, le facteur le plus proche de celui de la règle gagne.
#pragma once

#include <cstdint>
#include <expected>

#include "core/profile.h"
#include "core/resolve.h"
#include "core/types.h"

namespace ajp::core {

// Grid et CadenceChoice vivent dans core/plan.h : le Plan porte sa cadence.

namespace pacing {
// L'audio suit la vitesse par rééchantillonnage, borné à 0,125 %.
inline constexpr double kMaxSpeedChange = 0.00125;
// Une petite fraction garde des instants qui se répètent d'une paire à l'autre.
inline constexpr std::int64_t kMaxSimpleDen = 4;
// Une fraction exacte doit tenir à 1e-5 près : une image d'écart toutes les dix minutes à 165 Hz.
inline constexpr double kExactTolerance = 1e-5;
inline constexpr int kMaxVsyncsPerFrame = 16;
}  // namespace pacing

// Cale un facteur imposé sur l'écran, sans le changer. Facteur 1 : aucune interpolation.
std::expected<CadenceChoice, Refusal> fit_cadence(const DisplayInfo& display, double source_fps,
                                                  Ratio factor);

// Propose le facteur d'une étape d'interpolation unique, au plus max_factor
// (la borne du produit des facteurs de la règle). Repli : fit_cadence(requested).
std::expected<CadenceChoice, Refusal> choose_cadence(const DisplayInfo& display, double source_fps,
                                                     Ratio requested, double max_factor);

struct PacedRule {
  Rule rule;  // la règle, son facteur d'interpolation éventuellement remplacé
  CadenceChoice cadence;
};

// Seule une règle à une seule étape d'interpolation voit son facteur retouché ;
// les autres gardent leurs facteurs et reçoivent le calage de leur produit.
std::expected<PacedRule, Refusal> pace_rule(const Rule& rule, const DisplayInfo& display, double source_fps);

}  // namespace ajp::core
