// Ce que la calibration de l'assistant a mesuré sur cette carte (calibration.toml, machine_calibration.h) :
// en Auto, la chaîne garde ces bornes et s'adapte encore à chaque vidéo (source, cadence, écran). Dans son
// propre en-tête, sans dépendance : auto_config.h l'applique, le panneau des réglages le transporte.
#pragma once

#include <string>

namespace ajp::app::play {

struct CalibratedChain {
  bool span = true;
  bool rife = true;
  int height = 0;          // la plus haute résolution de travail que SPAN tient
  std::string span_model;  // le SPAN mesuré, joué en FP16 ; vide : le modèle par défaut
  std::string rife_model;  // le RIFE mesuré ; vide : RIFE 4.22 lite
};

}  // namespace ajp::app::play
