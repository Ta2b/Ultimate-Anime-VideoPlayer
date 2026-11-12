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
#include <optional>
#include <vector>

#include "core/frame.h"
#include "core/ring.h"

namespace ajp::app::play {

class FrameQueue {
 public:
  // B13-08 : un vecteur réservé, pas une deque qui allouait et libérait un bloc toutes les quelques images
  // sur le thread qui présente. Chaque image tient un slot du ring : jamais plus de kMaxRingSlots (16) d'une
  // époque, et les époques passées sont rendues au seek ; la marge couvre un seek pendant que gpu livre.
  static constexpr std::size_t kReserved = 64;
  FrameQueue() { frames_.reserve(kReserved); }

  // gpu livre dans l'ordre des pts, époque par époque.
  void push(const RingFrame& frame) { frames_.push_back(frame); }

  // Retire et rend les images d'une époque < epoch (seek, changement de Plan).
  std::vector<RingFrame> take_stale(std::uint32_t epoch);

  // Retire et rend toutes les images (arrêt).
  std::vector<RingFrame> take_all();

  // Première image de l'époque, si elle est arrivée.
  const RingFrame* first_of(std::uint32_t epoch) const;

  // B13-01 : la première image source (t = 0) de l'époque après after_pts et au plus à until_pts, si elle et
  // toutes celles qui la précèdent sont écrites (ready) ; nullptr sinon, ou si une image sans pts la précède.
  template <class Ready>
  const RingFrame* written_source_after(std::uint32_t epoch, std::int64_t after_pts, std::int64_t until_pts,
                                        const Ready& ready) const {
    for (const RingFrame& f : frames_) {
      if (f.epoch != epoch) continue;
      if (f.pts_us == kNoPts || f.pts_us > until_pts || !ready(f)) return nullptr;
      if (f.pts_us > after_pts && f.t_num == 0) return &f;
    }
    return nullptr;
  }

  // L'image à montrer quand l'horloge vaut media_now_us à la vsync : la dernière dont
  // pts ≤ media_now_us + lead_us. Celles qui la précèdent vont dans late. Une image sans
  // pts (kNoPts) est montrée dès qu'elle est en tête. Rien si la tête n'est pas atteinte.
  // ready : seules comptent les images de tête qu'il dit écrites ; la première qui ne l'est pas arrête le
  // choix, et elle reste en file avec celles d'après.
  template <class Ready>
  std::optional<RingFrame> pick(std::int64_t media_now_us, std::int64_t lead_us, std::vector<RingFrame>& late,
                                const Ready& ready) {
    return pick_by([&](const RingFrame& frame) { return frame.pts_us <= media_now_us + lead_us; }, late,
                   ready);
  }
  // Le même choix, l'échéance dite par due (6.6 : par numéro de vblank) au lieu du seuil de pts. Des modèles
  // et non des std::function : la lambda du choix par vblank en capture quatre, une allocation par tour.
  template <class Due, class Ready>
  std::optional<RingFrame> pick_by(const Due& due, std::vector<RingFrame>& late, const Ready& ready) {
    std::size_t taken = 0;
    std::optional<RingFrame> chosen;
    while (taken < frames_.size()) {
      const RingFrame& head = frames_[taken];
      const bool untimed = head.pts_us == kNoPts;
      if (!(untimed || due(head)) || !ready(head)) break;
      if (chosen) late.push_back(*chosen);
      chosen = head;
      ++taken;
      if (untimed) break;  // sans horodatage : montrée seule, à son tour
    }
    frames_.erase(frames_.begin(), frames_.begin() + static_cast<std::ptrdiff_t>(taken));
    return chosen;
  }

  // Retire l'image de tête (montrée sans horloge, à l'amorce d'une époque).
  std::optional<RingFrame> pop_front();

  bool empty() const { return frames_.empty(); }
  std::size_t size() const { return frames_.size(); }

 private:
  std::vector<RingFrame> frames_;
};

}  // namespace ajp::app::play
