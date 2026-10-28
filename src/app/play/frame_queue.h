// Images du ring arrivées au thread main, pas encore montrées, et le choix de celle à
// montrer à la prochaine vsync d'après l'horloge maître.
//
// Doublure du futur ordonnanceur de present (cadence par vsync, VRR) : ici, la règle la
// plus simple qui tienne une lecture correcte à l'horloge audio. À chaque tour, l'image
// montrée est la dernière dont le pts est atteint ; celles qu'elle double sont en retard,
// jamais montrées, et leurs slots rendus tout de suite.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <vector>

#include "core/ring.h"

namespace ajp::app::play {

class FrameQueue {
 public:
  // gpu livre dans l'ordre des pts, époque par époque.
  void push(const RingFrame& frame) { frames_.push_back(frame); }

  // Retire et rend les images d'une époque < epoch (seek, changement de Plan).
  std::vector<RingFrame> take_stale(std::uint32_t epoch);

  // Retire et rend toutes les images (arrêt).
  std::vector<RingFrame> take_all();

  // Première image de l'époque, si elle est arrivée.
  const RingFrame* first_of(std::uint32_t epoch) const;

  // L'image à montrer quand l'horloge vaut media_now_us à la vsync : la dernière dont
  // pts ≤ media_now_us + lead_us. Celles qui la précèdent vont dans late. Une image sans
  // pts (kNoPts) est montrée dès qu'elle est en tête. Rien si la tête n'est pas atteinte.
  // ready, s'il est donné : seules comptent les images de tête qu'il dit écrites ; la première
  // qui ne l'est pas arrête le choix, et elle reste en file avec celles d'après.
  std::optional<RingFrame> pick(std::int64_t media_now_us, std::int64_t lead_us, std::vector<RingFrame>& late,
                                const std::function<bool(const RingFrame&)>& ready = {});

  // Retire l'image de tête (montrée sans horloge, à l'amorce d'une époque).
  std::optional<RingFrame> pop_front();

  bool empty() const { return frames_.empty(); }
  std::size_t size() const { return frames_.size(); }

 private:
  std::deque<RingFrame> frames_;
};

}  // namespace ajp::app::play
