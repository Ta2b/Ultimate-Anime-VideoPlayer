// Le retard d'une image de sortie : de sa soumission par gpu (RingFrame::submitted_us) à l'instant
// où main la voit écrite. main le mesure à chaque tour ; gpu le lit à chaque paire pour savoir
// quels instants seront passés quand ils seront prêts. Sous surcharge, ce retard dépasse de loin
// une période de sortie : l'estimer à une période laissait calculer des images déjà en retard, que
// main jetait, et le GPU débordé prenait plus de retard encore.
//
// Médiane des kSamples dernières mesures : un à-coup isolé (premier passage d'un moteur, mémoire
// reprise par le système) ne la déplace pas, une charge qui dure la déplace en quelques images.
// Un seul écrivain (main), des lecteurs sur d'autres threads.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "core/clock.h"

namespace ajp::app::play {

class FrameLatency {
 public:
  static constexpr std::size_t kSamples = 9;
  // Au-delà, la mesure ne dit rien du GPU (pause, fenêtre cachée, main arrêté) : ignorée.
  static constexpr std::int64_t kMaxSampleUs = 1'000'000;

  // main seul.
  void observe(std::int64_t latency_us) {
    if (latency_us < 0 || latency_us > kMaxSampleUs) return;
    samples_[next_] = latency_us;
    next_ = (next_ + 1) % kSamples;
    count_ = std::min(count_ + 1, kSamples);
    std::array<std::int64_t, kSamples> sorted = samples_;
    const auto end = sorted.begin() + static_cast<std::ptrdiff_t>(count_);
    const auto middle = sorted.begin() + static_cast<std::ptrdiff_t>(count_ / 2);
    std::nth_element(sorted.begin(), middle, end);
    median_.store(*middle, std::memory_order_relaxed);
  }

  // 0 : aucune mesure encore.
  std::int64_t us() const { return median_.load(std::memory_order_relaxed); }

 private:
  std::array<std::int64_t, kSamples> samples_{};
  std::size_t count_ = 0;
  std::size_t next_ = 0;
  std::atomic<std::int64_t> median_{0};
};

// L'instant média où sera prête une image soumise à host_us : au plus tôt une période de sortie
// plus loin, au plus tard après le retard mesuré des images. Une image dont le pts le précède
// paraîtrait en retard : main la jetterait.
inline std::int64_t ready_media_us(const ClockSnapshot& now, std::int64_t host_us, std::int64_t period_us,
                                   std::int64_t latency_us) {
  return std::max(now.media_at(host_us) + period_us, now.media_at(host_us + latency_us));
}

}  // namespace ajp::app::play
