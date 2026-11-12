#include "app/play/subtitle_queue.h"

#include <algorithm>
#include <utility>

#include "core/frame.h"

namespace ajp::app::play {
namespace {

bool same_look(const subs::SubtitleFrame& a, const subs::SubtitleFrame& b) {
  const bool drawn_a = a.atlas && !a.rects.empty();
  const bool drawn_b = b.atlas && !b.rects.empty();
  if (drawn_a != drawn_b) return false;
  if (!drawn_a) return true;  // rien à dessiner, des deux côtés
  // B17-01 : la montée des dialogues compte aussi. Une réponse redemandée avec la barre est la même image
  // pour libass (panneau \pos), mais l'affichage décale celle rendue sans montée : la jeter servirait la
  // mauvaise.
  return a.atlas->generation == b.atlas->generation && a.width == b.width && a.height == b.height &&
         a.line_lift == b.line_lift && a.rects == b.rects;
}

}  // namespace

void SubtitleQueue::push(subs::SubtitleFrame&& frame, std::int64_t shown_pts) {
  // subs répond dans l'ordre des demandes ; une réponse en retard se range quand même à sa place.
  auto at = std::upper_bound(frames_.begin(), frames_.end(), frame.pts_us,
                             [](std::int64_t pts, const subs::SubtitleFrame& f) { return pts < f.pts_us; });
  // Les mêmes sous-titres que la réponse d'avant : elle sert déjà ce pts (la dernière de pts <= pts). Une
  // réponse d'après identique devient inutile, celle-ci la couvre. Les réponses gardées comptent ainsi les
  // changements de sous-titres, pas les images du ring.
  if (at != frames_.begin() && same_look(*(at - 1), frame)) {
    if (shown_pts != kNoPts) forget_before(shown_pts);
    return;
  }
  if (at != frames_.end() && same_look(*at, frame)) at = frames_.erase(at);
  frames_.insert(at, std::move(frame));
  if (shown_pts != kNoPts) forget_before(shown_pts);
  while (frames_.size() > capacity_) frames_.pop_back();
}

const subs::SubtitleFrame* SubtitleQueue::for_pts(std::int64_t pts) {
  if (pts == kNoPts) return nullptr;
  forget_before(pts);
  if (frames_.empty() || frames_.front().pts_us > pts) return nullptr;
  return &frames_.front();
}

void SubtitleQueue::forget_before(std::int64_t pts) {
  // Garde la dernière réponse de pts <= pts (celle de l'image) et tout ce qui la suit.
  while (frames_.size() > 1 && frames_[1].pts_us <= pts) frames_.pop_front();
}

}  // namespace ajp::app::play
