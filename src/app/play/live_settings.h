// Réglages que main change pendant la lecture et que le thread gpu lit à chaque image (facteur
// RIFE, filtre de réduction, réglages de RTX HDR), et les compteurs de RIFE que main lit pour les
// statistiques. Tout est atomique ou sous un verrou court : ni main ni gpu n'attendent l'autre
// plus que la copie d'un réglage.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "app/play/frame_latency.h"
#include "app/play/settings.h"
#include "core/auto_rate.h"
#include "gpu/kernel_ops.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::app::play {

struct RifeCounts {
  std::uint64_t pairs = 0;
  std::uint64_t interpolated = 0;
  std::uint64_t cuts = 0;
  std::uint64_t held = 0;
  std::uint64_t late = 0;  // non interpolées pour rattraper l'horloge
};

// Le filtre de réduction du noyau d'ingestion pour un choix du menu. Spline36 : celui
// d'AnimeJaNai (3 lobes) ; Bicubic : B = 1, C = 0, comme
// le « bicubic » de libplacebo et de mpv ; Mitchell : B = C = 1/3 ; Catmull-Rom : B = 0, C = 1/2.
inline gpu::Reduction reduction_of(Downscale filter) {
  switch (filter) {
    case Downscale::Area:
      return {};
    case Downscale::Spline36:
      return {.kind = gpu::ReduceKind::Spline36};
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

// Les 32 marches quasi noires de la source (8 bits limité : la marche k vaut k/219 de la plage),
// en codes RGBA8 pleine plage tels que TrueHDR les reçoit, et en niveaux 0 à 1.
inline std::array<std::uint8_t, render::kBlackRampSteps> black_ramp_codes() {
  std::array<std::uint8_t, render::kBlackRampSteps> codes{};
  for (std::size_t k = 0; k < codes.size(); ++k)
    codes[k] = static_cast<std::uint8_t>((static_cast<int>(k) * 255 + 109) / 219);
  return codes;
}
inline std::array<float, render::kBlackRampSteps> black_ramp_levels() {
  std::array<float, render::kBlackRampSteps> levels{};
  for (std::size_t k = 0; k < levels.size(); ++k) levels[k] = static_cast<float>(k) / 219.0F;
  return levels;
}

class LiveSettings {
 public:
  // ×num/den ; num ≤ den : aucune interpolation (le moteur reste chargé). Les deux termes dans un
  // seul atomique : jamais l'un sans l'autre.
  void set_rife_factor(Factor factor) {
    rife_factor_.store((static_cast<std::uint32_t>(std::max(factor.num, 0)) << 8U) |
                       static_cast<std::uint32_t>(std::clamp(factor.den, 1, 255)));
  }
  Factor rife_factor() const {
    const std::uint32_t packed = rife_factor_.load();
    return {.num = static_cast<int>(packed >> 8U), .den = static_cast<int>(packed & 0xffU)};
  }

  // Mode auto : le multiplicateur ×k suit ce que le GPU tient (core/auto_rate.h). Quand il est
  // actif, factor_now() remplace rife_factor(). `ceiling` : le palier le plus haut que l'écran permet.
  void set_auto(bool on, int ceiling) {
    const std::scoped_lock lock(mutex_);
    auto_on_ = on;
    auto_.set_ceiling(ceiling);
  }
  // gpu : l'issue de la paire qui vient d'être traitée.
  void auto_pair(bool late) {
    const std::scoped_lock lock(mutex_);
    if (auto_on_) auto_.pair(late);
  }
  // Seek, saut d'époque : la mesure du mode auto repart après quelques paires.
  void auto_settle() {
    const std::scoped_lock lock(mutex_);
    if (auto_on_) auto_.settle();
  }
  // Le facteur à appliquer maintenant : le palier auto, ou le facteur réglé.
  Factor factor_now() const {
    const std::scoped_lock lock(mutex_);
    return auto_on_ ? Factor{.num = auto_.step(), .den = 1} : rife_factor();
  }

  // Garder les poses (kernels/spec/poses.h), lu par gpu au début de chaque paire.
  void set_keep_poses(bool keep) { keep_poses_.store(keep); }
  bool keep_poses() const { return keep_poses_.load(); }

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

  // Vrai noir OLED sur la route TrueHDR : main dit s'il le faut ; gpu repasse alors la rampe de 32
  // marches par TrueHDR après de nouveaux réglages et dépose les codes PQ obtenus.
  void set_black_wanted(bool wanted) { black_wanted_.store(wanted); }
  bool black_wanted() const { return black_wanted_.load(); }
  void set_black_ramp(std::vector<float> pq) {
    const std::scoped_lock lock(mutex_);
    black_ramp_ = std::move(pq);
  }
  std::optional<std::vector<float>> take_black_ramp() {
    const std::scoped_lock lock(mutex_);
    return std::exchange(black_ramp_, std::nullopt);
  }

  // Compteurs de RIFE, écrits par gpu.
  void count_pair() { pairs_.fetch_add(1, std::memory_order_relaxed); }
  void count_interpolated() { interpolated_.fetch_add(1, std::memory_order_relaxed); }
  void count_cut() { cuts_.fetch_add(1, std::memory_order_relaxed); }
  void count_held() { held_.fetch_add(1, std::memory_order_relaxed); }
  void count_late() { late_.fetch_add(1, std::memory_order_relaxed); }
  // Images source sautées par gpu avant tout calcul, déjà passées à leur tour (RingStage::passed).
  void count_skipped() { skipped_.fetch_add(1, std::memory_order_relaxed); }
  std::uint64_t skipped() const { return skipped_.load(std::memory_order_relaxed); }
  // Retard des images de sortie, mesuré par main (Session), lu par gpu.
  FrameLatency& latency() { return latency_; }
  const FrameLatency& latency() const { return latency_; }
  RifeCounts counts() const {
    return RifeCounts{.pairs = pairs_.load(std::memory_order_relaxed),
                      .interpolated = interpolated_.load(std::memory_order_relaxed),
                      .cuts = cuts_.load(std::memory_order_relaxed),
                      .held = held_.load(std::memory_order_relaxed),
                      .late = late_.load(std::memory_order_relaxed)};
  }

 private:
  std::atomic<std::uint32_t> rife_factor_{1};  // ×0/1
  std::atomic<bool> keep_poses_{false};
  mutable std::mutex mutex_;
  gpu::Reduction reduction_;
  std::optional<vendor_hdr::TrueHdrParams> true_hdr_;
  std::atomic<bool> black_wanted_{false};
  std::optional<std::vector<float>> black_ramp_;
  std::atomic<std::uint64_t> pairs_{0};
  std::atomic<std::uint64_t> interpolated_{0};
  std::atomic<std::uint64_t> cuts_{0};
  std::atomic<std::uint64_t> held_{0};
  std::atomic<std::uint64_t> late_{0};
  std::atomic<std::uint64_t> skipped_{0};
  FrameLatency latency_;
  bool auto_on_ = false;
  core::AutoRate auto_;
};

}  // namespace ajp::app::play
