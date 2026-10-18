// Réserve bornée d'atlas : au plus kMaxAtlasesInFlight existent. Un atlas
// rempli part en lecture seule ; sa dernière référence le renvoie par la route
// de retour, qui le pousse dans la boîte du thread subs (AtlasReturned).
// La réserve elle-même n'appartient qu'au thread subs.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "subs/subtitle_frame.h"

namespace ajp::subs {

// Appelée par le thread qui relâche la dernière référence d'un atlas, depuis un
// destructeur : elle ne bloque pas et ne lève rien. Si elle ne garde pas
// l'atlas, il est détruit.
using AtlasReturn = std::function<void(std::unique_ptr<SubtitleAtlas>)>;

class AtlasPool {
 public:
  explicit AtlasPool(AtlasReturn route) : route_(std::move(route)) {}

  // Un atlas libre, créé s'il en manque, ou nullptr si tous sont en vol.
  std::unique_ptr<SubtitleAtlas> take();
  // Remet un atlas rendu, ou non utilisé après take().
  void give_back(std::unique_ptr<SubtitleAtlas> atlas);
  // Partage un atlas rempli ; la route le ramène quand plus personne ne le lit.
  std::shared_ptr<const SubtitleAtlas> share(std::unique_ptr<SubtitleAtlas> atlas) const;

  // Atlas sortis de la réserve et pas encore rendus.
  std::size_t in_flight() const { return created_ - free_.size(); }

 private:
  AtlasReturn route_;
  std::vector<std::unique_ptr<SubtitleAtlas>> free_;
  std::size_t created_ = 0;
};

}  // namespace ajp::subs
