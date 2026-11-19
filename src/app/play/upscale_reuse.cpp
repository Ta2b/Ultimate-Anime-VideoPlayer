// La règle de reprise de SPAN (upscale_reuse.h).
#include "app/play/upscale_reuse.h"

#include <algorithm>
#include <format>

#include "kernels/spec/detect.h"

namespace kirari::app::play {

ReuseKind SpanReuse::decide(const ReuseFacts& facts) {
  ++seen_;
  if (facts.corrupt) ++corrupt_;
  const auto compute = [&] {
    in_row_ = 0;
    anchor_bad_ = facts.corrupt;  // la nouvelle ancre, abîmée ou non
    return ReuseKind::Compute;
  };
  if (!enabled || !facts.same_epoch || facts.corrupt || anchor_bad_ || !facts.has_output || !facts.pair_held)
    return compute();
  if (in_row_ >= kMaxInRow) {
    ++stopped_limit_;
    return compute();
  }
  if (facts.anchor) {
    if (!kernels::is_held(facts.anchor->worst_block)) return compute();
    if (in_row_ >= kPairOnlyInRow && facts.anchor->mean_levels > kFadeMean) {
      ++stopped_fade_;
      return compute();
    }
  } else if (in_row_ >= kPairOnlyInRow) {  // la paire seule : la règle d'avant
    ++stopped_limit_;
    return compute();
  }
  take();
  return ReuseKind::Whole;
}

void SpanReuse::force() {
  ++seen_;
  take();
}

void SpanReuse::take() {
  ++in_row_;
  ++reused_;
  longest_ = std::max<std::uint64_t>(longest_, static_cast<std::uint64_t>(in_row_));
}

std::string SpanReuse::summary() const {
  if (seen_ == 0 && reused_ == 0) return {};
  const double share = seen_ > 0 ? 100.0 * static_cast<double>(reused_) / static_cast<double>(seen_) : 0.0;
  return std::format(
      "SPAN repris : {} sur {} images ({:.1f} %), série la plus longue {}, arrêts à la limite {}, au fondu "
      "{}, "
      "images abîmées {}{}",
      reused_, seen_, share, longest_, stopped_limit_, stopped_fade_, corrupt_,
      anchor_measure ? "" : " (sans mesure contre l'ancre : deux de suite au plus)");
}

}  // namespace kirari::app::play
