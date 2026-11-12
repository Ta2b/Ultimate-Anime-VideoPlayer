#include "app/play/pair_marks.h"

#include <algorithm>
#include <format>

#include "gpu/kernel_ops.h"
#include "kernels/spec/detect.h"

namespace ajp::app::play {

void PairMarks::add(const PairMark& mark) {
  const std::scoped_lock lock(mutex_);
  marks_[next_] = mark;
  next_ = (next_ + 1) % kKept;
  count_ = std::min(count_ + 1, kKept);
}

std::optional<PairVerdict> PairMarks::at(std::uint32_t epoch, std::int64_t pts_us) const {
  const std::scoped_lock lock(mutex_);
  // De la plus récente à la plus ancienne : une paire refaite (même intervalle) répond par son dernier
  // verdict.
  for (std::size_t k = 1; k <= count_; ++k) {
    const PairMark& mark = marks_[(next_ + kKept - k) % kKept];
    if (mark.epoch == epoch && pts_us > mark.from_us && pts_us <= mark.to_us) return mark.verdict;
  }
  return std::nullopt;
}

void PairMarks::clear() {
  const std::scoped_lock lock(mutex_);
  count_ = 0;
  next_ = 0;
}

std::string pair_mark_line(const PairMark& mark, const gpu::PairScores& scores, std::optional<bool> known_cut,
                           double useful_area) {
  const double worst = static_cast<double>(scores.worst_block) / 256.0;
  const double held_at = static_cast<double>(kernels::kHeldThreshold) / 100.0;
  // Le score que la coupe compare (rife_plan.cpp, useful_score) : rapporté à la zone hors bandes noires.
  const double score = scores.scene_score / std::clamp(useful_area, 0.25, 1.0);
  const char* plans = "hors plans";
  if (known_cut) plans = *known_cut ? "coupe" : "même plan";
  return std::format(
      "paire {} {:.3f} s → {:.3f} s, pire bloc {:.2f} (tenue ≤ {:.2f}), score {:.3f} (coupe > "
      "{:.3f}), plans : {}",
      mark.verdict == PairVerdict::Held ? "tenue" : "coupée", static_cast<double>(mark.from_us) / 1e6,
      static_cast<double>(mark.to_us) / 1e6, worst, held_at, score, kernels::kSceneCutThreshold, plans);
}

PairTint pair_tint(PairVerdict verdict) {
  if (verdict == PairVerdict::Cut)
    return {.wash = {1.0F, 0.20F, 0.15F, 0.22F}, .frame = {1.0F, 0.20F, 0.15F, 0.85F}};
  return {.wash = {0.15F, 0.55F, 1.0F, 0.22F}, .frame = {0.15F, 0.55F, 1.0F, 0.85F}};
}

}  // namespace ajp::app::play
