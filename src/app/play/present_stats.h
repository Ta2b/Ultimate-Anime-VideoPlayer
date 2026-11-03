// --stats : une ligne par présentation. L'en-tête et la ligne viennent du même tableau de colonnes, pour
// qu'ils ne se désaccordent plus (l'en-tête d'avant n'en nommait que 9 sur 10).
#pragma once

#include <string>

#include "app/play/session.h"

namespace ajp::app::play {

inline constexpr int kStatsColumns = 14;

// Les noms des colonnes, séparés par des tabulations, terminés par un saut de ligne.
std::string stats_header();
// La ligne d'une présentation, dans le même ordre, terminée par un saut de ligne.
std::string stats_line(const PresentRecord& record);

}  // namespace ajp::app::play
