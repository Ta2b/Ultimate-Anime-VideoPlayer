// Les règles du grade adaptatif, en un seul endroit. Le lecteur (app/play/chain_hdr.cpp, adaptive_tick) les
// applique quand il décide (le plan connu par la pré-analyse) ; le shader des statistiques de
// scène (anime-hdr-grade/2-analysis.hook) ne les applique qu'en repli, quand ce plan n'est pas là, et
// les reçoit d'ici à chaque image (grade.cpp, paramètres ar_*). Les valeurs par défaut écrites dans
// 1-params.hook sont les mêmes : un test les compare (tests/unit/render/adaptive_rules_test.cpp).
#pragma once

namespace ajp::render::adaptive_rules {

// La luminosité moyenne de la série de référence (Re:Zero : 0,30, mesuré sur 155 s de son épisode) ; les
// seuils clair/sombre suivent la série en proportion, ce rapport borné à [kRatioMin, kRatioMax].
inline constexpr double kSeriesRef = 0.30;
inline constexpr double kRatioMin = 0.7;
inline constexpr double kRatioMax = 1.8;
inline constexpr double kBrightCap = 0.95;  // le seuil « claire » ne dépasse jamais ce code
// La vibrance suit la couleur : un ciel ou un aplat bleu saturé (part de bleu entre ces bornes) n'en veut
// pas, seulement dans une scène au moins à demi claire (kBlueGate) ; un flash très saturé de toute teinte
// (chroma moyenne entre ces bornes) non plus.
inline constexpr double kBlueLo = 0.05;
inline constexpr double kBlueHi = 0.13;
inline constexpr double kBlueGate = 0.5;
inline constexpr double kChromaLo = 0.40;
inline constexpr double kChromaHi = 0.60;
// Série claire par nature (Slime : plus de 0,5) : plus de contraste et de couleur, entre ces bornes.
inline constexpr double kBrightSeriesLo = 0.36;
inline constexpr double kBrightSeriesHi = 0.50;
// Scène sombre avec éclat : une petite part de points clairs (de 0,2 à 2 %), pas une scène claire (au-delà
// de 12 à 30 %).
inline constexpr double kGlowShareLo = 0.002;
inline constexpr double kGlowShareHi = 0.02;
inline constexpr double kGlowSceneLo = 0.12;
inline constexpr double kGlowSceneHi = 0.30;

}  // namespace ajp::render::adaptive_rules
