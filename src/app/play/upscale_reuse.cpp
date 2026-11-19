// La règle de reprise de SPAN (upscale_reuse.h).
#include "app/play/upscale_reuse.h"

#include <algorithm>
#include <format>
#include <utility>

#include "kernels/spec/detect.h"

namespace kirari::app::play {

ReuseKind SpanReuse::decide(const ReuseFacts& facts) {
  ++seen_;
  if (facts.corrupt) ++corrupt_;
  tile_plan = TilePlan{};
  const auto compute = [&] {
    in_row_ = 0;
    anchor_ready_ = false;
    const bool behind_bad = std::exchange(anchor_bad_, facts.corrupt);  // la nouvelle ancre, abîmée ou non
    if (!tiles.active()) return ReuseKind::Compute;
    const bool restart = !facts.same_epoch || facts.corrupt || behind_bad || !facts.has_output;
    // Les tuiles suivent chaque image calculée, même entière : leur état repart de là.
    tile_plan = tiles.plan(TileFacts{
        .stale = facts.tiles_stale != TileStale::Chain && restart ? TileStale::Restart : facts.tiles_stale,
        .can_shift = facts.can_shift,
        .scene = facts.scene,
        .spread_px = facts.spread_px,
        .input_key = facts.input_key});
    return tile_plan.tiled ? ReuseKind::Tiles : ReuseKind::Compute;
  };
  if (!enabled || !facts.same_epoch || facts.corrupt || anchor_bad_ || !facts.has_output || !facts.pair_held)
    return compute();
  if (in_row_ >= kMaxInRow) {
    ++stopped_limit_;
    return compute();
  }
  if (facts.anchor) {
    if (!kernels::is_held(facts.anchor->worst_block)) {
      if (in_row_ > 0) ++stopped_anchor_;  // à 0, la mesure est celle de la paire
      return compute();
    }
  } else if (in_row_ >= kPairOnlyInRow) {  // la paire seule : la règle d'avant
    ++stopped_limit_;
    return compute();
  }
  take();
  tiles.held(facts.scene);
  return ReuseKind::Whole;
}

void SpanReuse::force() {
  ++seen_;
  anchor_ready_ = false;  // la série imposée reprend a sans en copier l'image
  tile_plan = TilePlan{};
  tiles.held(nullptr);
  take();
}

void SpanReuse::note_pair(std::uint32_t worst_block) {
  // Les bornes en centièmes de niveau, comme kernels::kHeldThreshold.
  static constexpr std::array<std::uint32_t, 6> kBounds{70, 100, 150, 200, 300, 500};
  std::size_t bucket = 0;
  while (bucket < kBounds.size() && !kernels::is_held(worst_block, kBounds[bucket])) ++bucket;
  ++pairs_[bucket];
}

void SpanReuse::take() {
  ++in_row_;
  ++reused_;
  longest_ = std::max<std::uint64_t>(longest_, static_cast<std::uint64_t>(in_row_));
}

std::string SpanReuse::summary() const {
  if (seen_ == 0 && reused_ == 0) return {};
  const double share = seen_ > 0 ? 100.0 * static_cast<double>(reused_) / static_cast<double>(seen_) : 0.0;
  std::string line = std::format(
      "SPAN repris : {} sur {} images ({:.1f} %), série la plus longue {}, arrêts à la limite {}, à l'ancre "
      "{}, images abîmées {}{}",
      reused_, seen_, share, longest_, stopped_limit_, stopped_anchor_, corrupt_,
      anchor_measure ? "" : " (sans mesure contre l'ancre : deux de suite au plus)");
  if (std::ranges::any_of(pairs_, [](std::uint64_t n) { return n > 0; }))
    line += std::format(
        " ; paires par pire bloc (niveaux) : jusqu'à 0.7 {}, 1 {}, 1.5 {}, 2 {}, 3 {}, 5 {}, au-delà {}",
        pairs_[0], pairs_[1], pairs_[2], pairs_[3], pairs_[4], pairs_[5], pairs_[6]);
  return line;
}

}  // namespace kirari::app::play
