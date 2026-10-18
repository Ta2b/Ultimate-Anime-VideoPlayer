#include "subs/atlas_pool.h"

namespace ajp::subs {

std::unique_ptr<SubtitleAtlas> AtlasPool::take() {
  if (!free_.empty()) {
    std::unique_ptr<SubtitleAtlas> atlas = std::move(free_.back());
    free_.pop_back();
    return atlas;
  }
  if (created_ == kMaxAtlasesInFlight) return nullptr;
  ++created_;
  return std::make_unique<SubtitleAtlas>();
}

void AtlasPool::give_back(std::unique_ptr<SubtitleAtlas> atlas) {
  // Un atlas qui ne vient pas d'ici dépasserait la borne : il est détruit.
  if (atlas == nullptr || free_.size() == created_) return;
  free_.push_back(std::move(atlas));
}

std::shared_ptr<const SubtitleAtlas> AtlasPool::share(std::unique_ptr<SubtitleAtlas> atlas) const {
  // shared_ptr rend au suppresseur le pointeur d'origine, non constant : on en
  // reprend la propriété pour le renvoyer. Le suppresseur (copie de la route)
  // existe avant que l'atlas ne quitte son unique_ptr : rien ne fuit si la copie échoue.
  auto release = [route = route_](SubtitleAtlas* released) {
    std::unique_ptr<SubtitleAtlas> owned(released);
    if (route) route(std::move(owned));
  };
  return {atlas.release(), std::move(release)};
}

}  // namespace ajp::subs
