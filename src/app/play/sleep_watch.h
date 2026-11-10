// La veille du système et les gels longs du processus, vus de la boucle de lecture (A29-02, A33-03, A31-12 ;
// décision 7 : pause avant la veille, recalage au réveil).
//
// L'horloge hôte (steady_clock) ne voit pas une veille de la même façon partout : sous Linux elle s'arrête
// (CLOCK_MONOTONIC), sous Windows elle continue (QPC). Une seconde horloge qui compte le temps de veille
// (CLOCK_BOOTTIME, GetTickCount64, mach_continuous_time) montre le trou dans les deux cas : deux tours de la
// boucle séparés de plus de kSleepGapUs sur l'une ou l'autre, c'est une veille ou un gel (SIGSTOP, débogueur,
// machine virtuelle). La lecture se met alors en pause à l'image d'avant le trou et recale ses horloges par
// un seek exact ; l'appareil est vérifié avant de reprendre.
//
// Et l'économiseur d'écran : SDL le coupe pour toute la vie du processus (A33-02). Il ne l'est plus que
// pendant qu'une vidéo avance.
#pragma once

#include <cstdint>
#include <optional>

namespace ajp::app::play {

// Plus long que tout tour de boucle ordinaire (une fenêtre déplacée sous Windows tient la boucle une ou deux
// secondes), plus court que toute veille.
inline constexpr std::int64_t kSleepGapUs = 5'000'000;

// Microsecondes d'une horloge qui avance aussi pendant la veille du système.
std::int64_t suspend_aware_now_us();

class SleepWatch {
 public:
  // Un tour de la boucle : la durée du trou depuis le tour d'avant s'il dépasse kSleepGapUs sur l'une des
  // deux horloges, sinon rien. Le premier tour ne voit aucun trou.
  std::optional<std::int64_t> turn(std::int64_t steady_us, std::int64_t aware_us);

 private:
  std::optional<std::int64_t> steady_us_;
  std::int64_t aware_us_ = 0;
};

// L'économiseur d'écran et la veille automatique restent coupés seulement quand une vidéo avance : en pause,
// à l'accueil, dans l'assistant et après la fin, l'écran peut s'éteindre (dalles OLED).
constexpr bool must_stay_awake(bool playing, bool paused, bool finished) {
  return playing && !paused && !finished;
}

// SDL_DisableScreenSaver ou SDL_EnableScreenSaver selon awake, seulement quand l'état change.
class ScreenSaverHold {
 public:
  void want(bool awake);

 private:
  std::optional<bool> awake_;
};

}  // namespace ajp::app::play
