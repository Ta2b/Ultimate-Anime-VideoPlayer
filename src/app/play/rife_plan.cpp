#include "app/play/rife_plan.h"

#include <algorithm>

#include "kernels/spec/detect.h"

namespace ajp::app::play {

namespace {

// Le score rapporté à la zone hors bandes noires : les bandes, identiques d'une image à l'autre, le diluent.
double useful_score(const PairFacts& facts) {
  return facts.scores->scene_score / std::clamp(facts.useful_area, 0.25, 1.0);
}

}  // namespace

bool measured_cut(const PairFacts& facts) {
  return facts.scores && kernels::is_scene_cut(useful_score(facts));
}

bool pair_cut(const PairFacts& facts) {
  const bool measured = measured_cut(facts);
  if (facts.known_cut && *facts.known_cut) {
    // Une coupe des plans sur deux images presque identiques est un effet (éclair, fumée) : pas de coupe.
    // Le plancher se compare au score rapporté, comme la mesure, et ne défait jamais une coupe mesurée
    // (A09-22).
    return !facts.scores || measured || useful_score(facts) >= kPlanCutFloor;
  }
  // P3.31 : « même plan » selon la pré-analyse ne l'emporte plus sur une coupe franche mesurée par RIFE.
  return measured;
}

bool rife_runs(const PairFacts& facts) {
  if (facts.factor <= 1 || facts.factor > kMaxRifeFactor || !facts.same_epoch) return false;
  if (facts.gap_us <= 0 || facts.gap_us > kMaxPairGapUs || !facts.scores) return false;
  return !pair_cut(facts) && !kernels::is_held(facts.scores->worst_block);
}

}  // namespace ajp::app::play
