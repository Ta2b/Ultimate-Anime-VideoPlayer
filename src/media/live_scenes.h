// Les scènes de la lecture d'après l'analyse des images à l'ingestion (gpu/kernel_ops.h,
// frame_stats_start) : le second producteur de l'analyse v2, pour la première lecture d'une vidéo et
// après un saut, tant que la pré-analyse des plans (media/shot_probe.h) n'y est pas. Une coupe franche
// (frame_distance au-dessus de kLiveCut) ouvre une scène ; ses valeurs sont la moyenne de ses images
// vues jusque-là. Le fil du GPU écrit, le fil principal lit : sous verrou.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>

#include "kernels/spec/frame_stats.h"
#include "media/flash_guard.h"
#include "media/shot_cutter.h"

namespace ajp::media {

inline constexpr float kLiveCut = 0.35F;
inline constexpr std::size_t kLiveKept = 64;  // images gardées, environ 2,5 s à 24 i/s

struct LiveScene {
  ShotStats stats;
  ShotLight light;           // moyenne des images de la scène
  std::uint64_t serial = 0;  // change à chaque coupe ou saut
  bool flashing = false;     // flashs dangereux autour de cette image (flash_guard.h)
};

class LiveScenes {
 public:
  // L'analyse d'une image source. Un pts qui recule (un saut) ouvre une scène.
  void push(std::int64_t pts_us, const kernels::FrameStats& stats);
  // La scène de l'image à pts_us : la dernière analysée à ou avant elle, à moins de max_gap_us.
  std::optional<LiveScene> at(std::int64_t pts_us, std::int64_t max_gap_us = 500'000) const;
  // Une image source dont l'analyse a été perdue (relecture GPU en retard) : le limiteur en tient compte.
  void missed(std::int64_t pts_us);
  // Ce que le grade fait à la lumière de la source, pour le limiteur de flashs (flash_guard.h).
  void set_flash_gain(double gain);
  void clear();

 private:
  struct Entry {
    std::int64_t pts_us = 0;
    LiveScene scene;
  };
  mutable std::mutex mutex_;
  std::deque<Entry> kept_;
  std::optional<ShotFrame> previous_;
  std::array<float, kShotBins> histogram_{};
  ShotLight light_{};
  float mean_ = 0.0F, blue_ = 0.0F, chroma_ = 0.0F;
  std::size_t frames_ = 0;
  std::uint64_t serial_ = 0;
  FlashGuard flashes_;
};

}  // namespace ajp::media
