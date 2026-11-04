// Méga plan B2 : la mosaïque de calibration du goût (render/curve.h, 8.10). Quatre plans, un par classe de
// scène, se montrent ensemble en deux versions, « 1 » et « 2 », qui ne diffèrent que le long d'une direction
// du goût (un axe : E d'une classe, pied ou vibrance) ; l'utilisateur choisit la meilleure. Chaque choix dit
// de quel côté de l'estimation se trouve son goût : l'estimation y passe et le pas de l'axe se divise par
// deux (une dichotomie, trois tours par axe, dix-huit en tout). Le goût rendu est l'ajustement aux moindres
// carrés de ces bornes avec un léger rappel vers le départ (poids kMosaicPrior contre un par tour).
// La partie sans écran : l'image de la mosaïque (quatre plans gradés côte à côte) se fera dans le panneau de
// calibration ; rien ici ne touche à la lecture ni à play.toml.
#pragma once

#include <array>
#include <cstddef>

#include "app/play/grade_taste.h"

namespace ajp::app::play {

inline constexpr std::size_t kTasteAxes = kSceneClasses + 2;  // E × 4, pied, vibrance
inline constexpr int kMosaicPasses = 3;
inline constexpr int kMosaicRounds = kMosaicPasses * static_cast<int>(kTasteAxes);
inline constexpr double kMosaicStep = 0.5;  // le premier pas, en coordonnées réduites
inline constexpr double kMosaicPrior = 0.25;

// Un goût en coordonnées réduites : chaque axe divisé par sa borne (E / kCurveMaxE, pied / kCurveMaxToe,
// (vibrance − 1) / (kCurveMaxVibrance − 1)) ; la vibrance réduite va de 0 à 1, les autres de −1 à 1.
using TasteVector = std::array<double, kTasteAxes>;
TasteVector reduce(const GradeTaste& taste);
GradeTaste expand(const TasteVector& v);

struct MosaicRound {
  int index = 0;
  GradeTaste one;  // la version « 1 »
  GradeTaste two;  // la version « 2 »
};

class GradeMosaic {
 public:
  explicit GradeMosaic(const GradeTaste& start = {});
  bool done() const { return round_ >= kMosaicRounds; }
  // Le tour à montrer (après le dernier : le même axe, autour du goût trouvé).
  MosaicRound round() const;
  // Le choix de l'utilisateur : 1 ou 2 (toute autre valeur est ignorée).
  void choose(int which);
  // Le goût ajusté sur les choix faits, borné.
  GradeTaste result() const;

 private:
  TasteVector start_{};
  TasteVector centre_{};
  TasteVector step_{};
  std::array<int, kTasteAxes> seen_{};
  int round_ = 0;
};

}  // namespace ajp::app::play
