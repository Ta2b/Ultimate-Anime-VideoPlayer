#pragma once

#include <cstdint>

namespace ajp::app::play {

// La cadence des images vidéo distinctes qui arrivent à l'écran (la source, multipliée par l'interpolation),
// estimée sur l'écart de leurs pts : le grade y règle ses lissages pour qu'ils durent le même temps à 24 et à
// 165 images par seconde. La même image représentée (pause, cadence de l'écran) ne compte pas ; une coupure
// d'époque (seek) ou un écart hors bornes ne donne pas d'écart.
class VideoRate {
 public:
  static constexpr double kDefaultFps = 24.0;

  void observe(std::int64_t pts_us, std::uint32_t epoch) {
    if (has_last_ && epoch == epoch_ && pts_us > last_pts_us_) {
      const auto delta = static_cast<double>(pts_us - last_pts_us_);
      if (delta >= kMinFrameUs && delta <= kMaxFrameUs)
        frame_us_ = frame_us_ == 0.0 ? delta : frame_us_ + ((delta - frame_us_) * kWeight);
    }
    has_last_ = true;
    epoch_ = epoch;
    last_pts_us_ = pts_us;
  }

  double fps() const { return frame_us_ > 0.0 ? 1e6 / frame_us_ : kDefaultFps; }

 private:
  static constexpr double kMinFrameUs = 1'000.0;    // 1000 i/s
  static constexpr double kMaxFrameUs = 250'000.0;  // 4 i/s : au-delà, un saut, pas une cadence
  static constexpr double kWeight = 0.05;           // une vingtaine d'images
  bool has_last_ = false;
  std::uint32_t epoch_ = 0;
  std::int64_t last_pts_us_ = 0;
  double frame_us_ = 0.0;
};

}  // namespace ajp::app::play
