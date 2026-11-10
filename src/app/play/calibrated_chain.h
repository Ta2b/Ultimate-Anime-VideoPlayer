// Ce que la calibration de l'assistant a mesuré sur cette carte (calibration.toml, machine_calibration.h) :
// en Auto, la chaîne garde ces bornes et s'adapte encore à chaque vidéo (source, cadence, écran). Dans son
// propre en-tête, sans dépendance : auto_config.h l'applique, le panneau des réglages le transporte.
#pragma once

#include <cstdint>
#include <string>

namespace ajp::app::play {

struct CalibratedChain {
  bool span = true;
  bool rife = true;
  int height = 0;          // la plus haute résolution de travail que SPAN tient
  std::string span_model;  // le SPAN mesuré, joué en FP16 ; vide : le modèle par défaut
  std::string rife_model;  // le RIFE mesuré ; vide : RIFE 4.22 lite
};

// La machine telle qu'elle est maintenant, pour juger si une calibration vaut encore (A36-02, A25-02,
// A07-05) : la carte, l'écran, le moteur qui joue SPAN et RIFE et le pilote.
struct CalibrationHost {
  std::string gpu;
  int display_height = 0;  // 0 : inconnue, non comparée
  bool tensorrt = false;   // SPAN et RIFE passent par TensorRT (sinon Vulkan, ou Core ML sur Mac)
  std::string driver;      // vide : inconnu, non comparé
};

// Ce que l'Auto fait d'une calibration. Followed : ses bornes s'appliquent. Les autres : l'Auto garde son
// estimation, et le tableau du banc dit pourquoi.
enum class CalibrationUse : std::uint8_t {
  Followed,
  OldVersion,    // écrite par une version dont les essais ou le verdict n'étaient pas ceux d'aujourd'hui
  OnBattery,     // mesurée sur batterie : carte bridée
  OtherMachine,  // une autre carte ou un autre écran
  OtherEngine,   // mesurée sous Vulkan et jouée sous TensorRT, ou l'inverse : ses bornes ne valent pas
};

}  // namespace ajp::app::play
