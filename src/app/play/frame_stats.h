// Mesures de la présentation pour l'OSD et le rapport : temps par image (écart entre deux
// présentations d'images neuves), temps de composition, images sautées, et la fluidité vue de
// l'écran : images neuves par seconde, images manquées, retard sur l'horloge.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/frame.h"

namespace ajp::app::play {

// Fenêtre glissante de durées, en microsecondes, sans allocation après construction.
class Window {
 public:
  static constexpr std::size_t kSize = 240;  // 10 s à 24 im/s
  void add(std::int64_t us);
  std::size_t count() const { return count_; }
  std::int64_t latest() const { return count_ == 0 ? 0 : values_[(next_ + kSize - 1) % kSize]; }
  std::int64_t median() const;
  std::int64_t percentile(int percent) const;  // 0 à 100
  double stddev() const;

 private:
  std::array<std::int64_t, kSize> values_{};
  std::size_t next_ = 0;
  std::size_t count_ = 0;
};

struct FrameStats {
  std::int64_t presents = 0;       // présentations, redessins compris
  std::int64_t new_frames = 0;     // images neuves montrées
  std::int64_t dropped = 0;        // images arrivées trop tard, jamais montrées
  std::int64_t stale = 0;          // images d'une époque passée (seek), jetées
  std::int64_t subtitle_gaps = 0;  // présentations sans sous-titres alors que des réponses attendaient
  Window frame_interval;           // entre deux images neuves
  Window present_interval;         // entre deux présentations : la période de l'écran en FIFO
  Window render;                   // composition et soumission
  std::int64_t last_new_present_us = 0;
  std::int64_t last_present_us = 0;
  // Images attendues jamais montrées, d'après l'écart des pts de deux images neuves : sautées à la
  // présentation, ou jamais produites (paire non interpolée, GPU en retard). Ce que « dropped »
  // ne voit pas quand le GPU suit de justesse : la file reste vide et l'écran garde l'image.
  std::int64_t missing = 0;
  Window lag;                             // retard des images neuves sur l'horloge à leur présentation, ≥ 0
  double shown_fps = 0.0;                 // images neuves par seconde, relu chaque seconde
  std::int64_t last_new_pts_us = kNoPts;  // pts de la dernière image neuve ; kNoPts après un seek
  std::int64_t fps_since_us = 0;
  std::int64_t fps_frames = 0;

  // Une présentation : render_us de composition, present_us instant hôte, fresh si
  // l'image est neuve.
  void presented(std::int64_t render_us, std::int64_t present_us, bool fresh);
  // Une image neuve de pts pts_us présentée lag_us après son instant d'horloge ; period_us : la
  // période attendue entre deux images (0 : inconnue, rien n'est compté manquant).
  void paced(std::int64_t pts_us, std::int64_t lag_us, std::int64_t period_us);
  // Le seek ou la reprise d'une nouvelle époque : l'écart de pts suivant n'est pas un manque.
  void restart_pacing();
};

// « 41,7 ms » : millisecondes à une décimale, virgule française.
std::string format_ms(std::int64_t us);
// « 59,9 » : images par seconde à une décimale, virgule française.
std::string format_fps(double fps);
// « 01:23:45.678 » : position média.
std::string format_position(std::int64_t us);

}  // namespace ajp::app::play
