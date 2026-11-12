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

#include <atomic>
#include <cstdint>
#include <optional>

namespace ajp::app::play {

// Plus long que tout tour de boucle ordinaire (une fenêtre déplacée sous Windows tient la boucle une ou deux
// secondes), plus court que toute veille.
inline constexpr std::int64_t kSleepGapUs = 5'000'000;

// Le trou, veille ou gel de la boucle (B31-13, B29-06, B11-13). Seule la veille met en pause avec « Paused
// after sleep » ; un gel (barre de titre tenue sous Windows, qui fait entrer la boucle modale du système,
// SIGSTOP, débogueur) laisse la lecture se recaler d'elle-même, sans message.
enum class GapKind : std::uint8_t { Sleep, Freeze };
// Un événement de la fenêtre traité pendant le trou, plus de kHeldAfterUs après son début : SDL tournait dans
// la boucle modale de Windows (déplacement, redimensionnement), la machine ne dormait pas.
inline constexpr std::int64_t kHeldAfterUs = 250'000;
// window_held : un tel événement ; clocks_split : steady s'arrête pendant la veille et l'horloge de veille
// non (Linux, macOS), la veille se reconnaît alors à leur écart. Sous Windows, les deux comptent la veille :
// sans fenêtre tenue, un trou reste une veille. Pur.
constexpr GapKind gap_kind(std::int64_t steady_gap_us, std::int64_t aware_gap_us, bool window_held,
                           bool clocks_split) {
  if (window_held) return GapKind::Freeze;
  if (!clocks_split) return GapKind::Sleep;
  return aware_gap_us - steady_gap_us >= kSleepGapUs ? GapKind::Sleep : GapKind::Freeze;
}

// Microsecondes d'une horloge qui avance aussi pendant la veille du système.
std::int64_t suspend_aware_now_us();

class SleepWatch {
 public:
  // Un tour de la boucle : la durée du trou depuis le tour d'avant s'il dépasse kSleepGapUs sur l'une des
  // deux horloges, sinon rien. Le premier tour ne voit aucun trou.
  std::optional<std::int64_t> turn(std::int64_t steady_us, std::int64_t aware_us);
  // Le dernier trou rendu par turn, veille ou gel. window_event_aware_us : l'instant (horloge de veille) du
  // dernier événement de la fenêtre (WindowEvents::last_aware_us), 0 sans aucun.
  GapKind last_kind(std::int64_t window_event_aware_us) const;

 private:
  std::optional<std::int64_t> steady_us_;
  std::int64_t aware_us_ = 0;
  std::int64_t gap_steady_us_ = 0;  // le dernier trou, sur chaque horloge
  std::int64_t gap_aware_us_ = 0;
  std::int64_t gap_from_aware_ = 0;  // son début, horloge de veille
};

// Les événements de la fenêtre (déplacée, redimensionnée, à redessiner) vus par un filtre de SDL, appelé même
// depuis la boucle modale de Windows où la boucle de lecture est arrêtée : l'instant du dernier, en horloge
// de veille, pour SleepWatch::last_kind. Un seul à la fois, le temps d'une lecture.
class WindowEvents {
 public:
  WindowEvents();
  ~WindowEvents();
  WindowEvents(const WindowEvents&) = delete;
  WindowEvents& operator=(const WindowEvents&) = delete;
  std::int64_t last_aware_us() const;

 private:
  std::atomic<std::int64_t> aware_us_{0};  // écrit par le veilleur d'événements de SDL
};

// L'économiseur d'écran et la veille automatique restent coupés seulement quand une vidéo avance : en pause,
// à l'accueil, dans l'assistant et après la fin, l'écran peut s'éteindre (dalles OLED).
constexpr bool must_stay_awake(bool playing, bool paused, bool finished) {
  return playing && !paused && !finished;
}

// L'écran reste éveillé le temps d'une attente sans touche ni image qui avance : construction d'un moteur,
// téléchargement, calibration (jusqu'à vingt minutes). Sans lui l'écran s'éteignait, la session se
// verrouillait, et Windows mettait le portable en veille pendant que le GPU travaillait.
class AwakeWhile {
 public:
  AwakeWhile();
  ~AwakeWhile();
  AwakeWhile(const AwakeWhile&) = delete;
  AwakeWhile& operator=(const AwakeWhile&) = delete;
};

// SDL_DisableScreenSaver ou SDL_EnableScreenSaver selon awake, seulement quand l'état change.
class ScreenSaverHold {
 public:
  void want(bool awake);

 private:
  std::optional<bool> awake_;
};

}  // namespace ajp::app::play
