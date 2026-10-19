// Réglages que main change pendant la lecture et que le thread gpu lit à chaque image (facteur
// RIFE, filtre de réduction, réglages de RTX HDR), et les compteurs de RIFE que main lit pour les
// statistiques. Tout est atomique ou sous un verrou court : ni main ni gpu n'attendent l'autre
// plus que la copie d'un réglage.
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

#include "app/play/settings.h"
#include "gpu/kernel_ops.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::app::play {

struct RifeCounts {
  std::uint64_t pairs = 0;
  std::uint64_t interpolated = 0;
  std::uint64_t cuts = 0;
  std::uint64_t held = 0;
};

// Le filtre de réduction du noyau d'ingestion pour un choix du menu. Bicubic : B = 1, C = 0, comme
// le « bicubic » de libplacebo et de mpv ; Mitchell : B = C = 1/3 ; Catmull-Rom : B = 0, C = 1/2.
inline gpu::Reduction reduction_of(Downscale filter) {
  switch (filter) {
    case Downscale::Area:
      return {};
    case Downscale::Bilinear:
      return {.kind = gpu::ReduceKind::Triangle};
    case Downscale::Bicubic:
      return {.kind = gpu::ReduceKind::Cubic, .b = 1.0F, .c = 0.0F};
    case Downscale::Mitchell:
      return {.kind = gpu::ReduceKind::Cubic, .b = 1.0F / 3.0F, .c = 1.0F / 3.0F};
    case Downscale::CatmullRom:
      return {.kind = gpu::ReduceKind::Cubic, .b = 0.0F, .c = 0.5F};
  }
  return {};
}

class LiveSettings {
 public:
  // 0 ou 1 : aucune interpolation (le moteur RIFE reste chargé).
  void set_rife_factor(std::uint32_t factor) { rife_factor_.store(factor); }
  std::uint32_t rife_factor() const { return rife_factor_.load(); }

  void set_reduction(const gpu::Reduction& reduction) {
    const std::scoped_lock lock(mutex_);
    reduction_ = reduction;
  }
  gpu::Reduction reduction() const {
    const std::scoped_lock lock(mutex_);
    return reduction_;
  }

  // Réglages de RTX HDR à appliquer à l'image suivante ; gpu les prend une fois.
  void set_true_hdr(const vendor_hdr::TrueHdrParams& params) {
    const std::scoped_lock lock(mutex_);
    true_hdr_ = params;
  }
  std::optional<vendor_hdr::TrueHdrParams> take_true_hdr() {
    const std::scoped_lock lock(mutex_);
    return std::exchange(true_hdr_, std::nullopt);
  }

  // Compteurs de RIFE, écrits par gpu.
  void count_pair() { pairs_.fetch_add(1, std::memory_order_relaxed); }
  void count_interpolated() { interpolated_.fetch_add(1, std::memory_order_relaxed); }
  void count_cut() { cuts_.fetch_add(1, std::memory_order_relaxed); }
  void count_held() { held_.fetch_add(1, std::memory_order_relaxed); }
  RifeCounts counts() const {
    return RifeCounts{.pairs = pairs_.load(std::memory_order_relaxed),
                      .interpolated = interpolated_.load(std::memory_order_relaxed),
                      .cuts = cuts_.load(std::memory_order_relaxed),
                      .held = held_.load(std::memory_order_relaxed)};
  }

 private:
  std::atomic<std::uint32_t> rife_factor_{0};
  mutable std::mutex mutex_;
  gpu::Reduction reduction_;
  std::optional<vendor_hdr::TrueHdrParams> true_hdr_;
  std::atomic<std::uint64_t> pairs_{0};
  std::atomic<std::uint64_t> interpolated_{0};
  std::atomic<std::uint64_t> cuts_{0};
  std::atomic<std::uint64_t> held_{0};
};

}  // namespace ajp::app::play
