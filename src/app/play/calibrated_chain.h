// Ce que la calibration de l'assistant a mesuré sur cette carte (calibration.toml, machine_calibration.h) :
// en Auto, la chaîne garde ces bornes et s'adapte encore à chaque vidéo (source, cadence, écran). Dans son
// propre en-tête, sans dépendance : auto_config.h l'applique, le panneau des réglages le transporte.
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace ajp::app::play {

// Les hauteurs de travail que la calibration essaie, de la plus légère à la plus lourde, jusqu'à celle de la
// séquence (media/bench_clip.h : 1080p). 900p n'y est pas : un essai et un moteur de plus pour une réduction
// de la source 1080p qu'aucun écran n'appelle. L'Auto calibré ne descend que sur ces paliers : un autre
// construirait sous TensorRT un moteur que la calibration n'a jamais mesuré (C07-01).
inline constexpr std::array<int, 5> kCalibrationHeights{360, 480, 540, 720, 1080};
// La cadence de la séquence de la calibration (media/bench_clip.h) : ses hauteurs tiennent à ce débit
// d'images.
inline constexpr double kCalibrationFps = 24.0;

struct CalibratedChain {
  bool span = true;
  bool rife = true;
  int height = 0;          // la plus haute résolution de travail que SPAN tient avec la chaîne recommandée
  std::string span_model;  // le SPAN mesuré, joué en FP16 ; vide : le modèle par défaut
  std::string rife_model;  // le RIFE mesuré ; vide : RIFE 4.22 lite
  // Le plus haut palier où SPAN seul a tenu (0 : aucun) : la priorité Image le prend, avec RIFE en mode
  // Rapide qui s'adapte, quand il est au-dessus de la chaîne recommandée, qui met la fluidité d'abord
  // (C07-04).
  int span_height = 0;
  bool operator==(const CalibratedChain&) const = default;
};

// La machine telle qu'elle est maintenant, pour juger si une calibration vaut encore (A36-02, A25-02,
// A07-05) : la carte, l'écran, le moteur qui joue SPAN et RIFE et le pilote.
struct CalibrationHost {
  std::string gpu;
  int display_height = 0;   // 0 : inconnue, non comparée
  bool tensorrt = false;    // SPAN et RIFE passent par TensorRT (sinon Vulkan, ou Core ML sur Mac)
  std::string driver;       // vide : inconnu, non comparé
  double display_hz = 0.0;  // la fréquence de l'écran (C37-08) ; 0 : inconnue, non comparée
};

// Ce que l'Auto fait d'une calibration. Followed : ses bornes s'appliquent. Les autres : l'Auto garde son
// estimation, et le tableau du banc dit pourquoi.
enum class CalibrationUse : std::uint8_t {
  Followed,
  OldVersion,    // écrite par une version dont les essais ou le verdict n'étaient pas ceux d'aujourd'hui
  OnBattery,     // mesurée sur batterie : carte bridée
  OtherMachine,  // une autre carte ou un autre écran (sa hauteur, sa fréquence)
  OtherEngine,   // mesurée sous Vulkan et jouée sous TensorRT, ou l'inverse : ses bornes ne valent pas
};

}  // namespace ajp::app::play
