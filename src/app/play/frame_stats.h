// Mesures de la présentation pour l'OSD et le rapport : temps par image (écart entre deux
// présentations d'images neuves), temps de composition, images sautées.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ajp::app::play {

// Fenêtre glissante de durées, en microsecondes, sans allocation après construction.
class Window {
 public:
  static constexpr std::size_t kSize = 240;  // 10 s à 24 im/s
  void add(std::int64_t us);
  std::size_t count() const { return count_; }
  std::int64_t median() const;
  std::int64_t percentile(int percent) const;  // 0 à 100
  double stddev() const;

 private:
  std::array<std::int64_t, kSize> values_{};
  std::size_t next_ = 0;
  std::size_t count_ = 0;
};

struct FrameStats {
  std::int64_t presents = 0;    // présentations, redessins compris
  std::int64_t new_frames = 0;  // images neuves montrées
  std::int64_t dropped = 0;     // images arrivées trop tard, jamais montrées
  std::int64_t stale = 0;       // images d'une époque passée (seek), jetées
  Window frame_interval;        // entre deux images neuves
  Window present_interval;      // entre deux présentations : la période de l'écran en FIFO
  Window render;                // composition et soumission
  std::int64_t last_new_present_us = 0;
  std::int64_t last_present_us = 0;

  // Une présentation : render_us de composition, present_us instant hôte, fresh si
  // l'image est neuve.
  void presented(std::int64_t render_us, std::int64_t present_us, bool fresh);
};

// « 41,7 ms » : millisecondes à une décimale, virgule française.
std::string format_ms(std::int64_t us);
// « 01:23:45.678 » : position média.
std::string format_position(std::int64_t us);

}  // namespace ajp::app::play
