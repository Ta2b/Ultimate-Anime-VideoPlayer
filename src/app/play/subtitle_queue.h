// Images de sous-titres déjà rendues par subs, en attente de l'image vidéo qu'elles habillent.
//
// Règle : les sous-titres se composent sur CHAQUE image présentée, au temps de présentation de
// cette image (après l'interpolation, jamais au pts d'une source). L'image d'un pts donné prend
// la dernière réponse de pts inférieur ou égal ; la réponse dont l'image à l'écran a besoin n'est
// jamais jetée parce que d'autres, plus en avance, sont arrivées. Quand la place manque, c'est la
// réponse la plus en avance qui part : son image reprendra la précédente, au pire une image tôt,
// jamais une image sans sous-titres.
//
// Clignotement du 19 octobre : l'ancienne file gardait 3 réponses et jetait la plus ancienne. Avec
// 4 slots dans le ring (l'image à l'écran plus 3 en avance), les 3 réponses en avance chassaient
// celle de l'image affichée, qui perdait ses sous-titres jusqu'à l'image suivante.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>

#include "subs/subtitle_frame.h"

namespace ajp::app::play {

class SubtitleQueue {
 public:
  explicit SubtitleQueue(std::size_t capacity) : capacity_(capacity) {}

  // Une réponse de subs de l'époque courante ; shown_pts est le pts de l'image à l'écran
  // (kNoPts s'il n'y en a pas encore).
  void push(subs::SubtitleFrame&& frame, std::int64_t shown_pts);
  // Sous-titres de l'image de pts donné : la dernière réponse de pts <= pts, ou nullptr. Oublie
  // les réponses plus anciennes, qui ne serviront plus.
  const subs::SubtitleFrame* for_pts(std::int64_t pts);
  void clear() { frames_.clear(); }
  std::size_t size() const { return frames_.size(); }

 private:
  void forget_before(std::int64_t pts);

  std::size_t capacity_;
  std::deque<subs::SubtitleFrame> frames_;  // pts croissants
};

}  // namespace ajp::app::play
