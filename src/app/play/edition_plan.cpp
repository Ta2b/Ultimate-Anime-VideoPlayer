#include "app/play/edition_plan.h"

namespace ajp::app::play {

bool EditionPlan::inside(std::size_t index, std::int64_t pts_us, std::int64_t period_us) const {
  const PlayPart& part = parts_[index];
  return pts_us >= part.start_us - period_us && pts_us < part.end_us;
}

std::optional<std::int64_t> EditionPlan::seek(std::int64_t target_us) {
  if (!active()) return target_us;
  entered_ = false;
  jumped_ = false;
  // La partie en cours d'abord : une plage reprise deux fois (un générique) garde sa place dans l'édition.
  if (inside(index_, target_us, 0)) return target_us;
  for (std::size_t n = 0; n < parts_.size(); ++n) {
    if (inside(n, target_us, 0)) {
      index_ = n;
      return target_us;
    }
  }
  std::optional<std::size_t> next;
  for (std::size_t n = 0; n < parts_.size(); ++n) {
    if (parts_[n].start_us > target_us && (!next || parts_[n].start_us < parts_[*next].start_us)) next = n;
  }
  if (!next) return std::nullopt;
  index_ = *next;
  return parts_[index_].start_us;
}

EditionPlan::Step EditionPlan::shown(std::int64_t pts_us, std::int64_t period_us) {
  if (!active()) return {};
  if (!entered_) {
    if (inside(index_, pts_us, period_us)) {
      entered_ = true;
      jumped_ = false;
    } else if (!jumped_) {
      // Le début du fichier n'est pas celui de l'édition, ou l'image montrée est hors de la partie.
      jumped_ = true;
      return Jump{parts_[index_].start_us};
    } else {
      return {};  // le saut n'a pas encore d'image
    }
  }
  const PlayPart& part = parts_[index_];
  if (part.end_us == PlayPart::kToEnd || pts_us + period_us < part.end_us) return {};
  if (index_ + 1 >= parts_.size()) return End{};
  ++index_;
  // La partie suivante prolonge celle-ci dans le fichier : la lecture continue sans seek.
  if (parts_[index_].start_us == part.end_us) return {};
  entered_ = false;
  jumped_ = true;
  return Jump{parts_[index_].start_us};
}

}  // namespace ajp::app::play
