// Désentrelacement des images logicielles (P3.25) : bwdif de libavfilter, une image par image
// (send_frame), les seules images marquées entrelacées traitées. bwdif lit l'image suivante : la
// sortie a une image de retard, rendue par flush() à la fin du flux.
#pragma once

#include <expected>
#include <memory>

#include "core/types.h"
#include "media/av_util.h"

namespace ajp::media {

class Deinterlacer {
 public:
  Deinterlacer();
  ~Deinterlacer();
  Deinterlacer(Deinterlacer&&) noexcept;
  Deinterlacer& operator=(Deinterlacer&&) noexcept;

  // Une image décodée, en microsecondes ; le graphe se refait si la taille ou le format change.
  std::expected<void, Refusal> push(FramePtr frame);
  // La fin du flux : les images retenues sortent par pull().
  std::expected<void, Refusal> flush();
  // Une image désentrelacée, ou rien tant que bwdif attend la suivante.
  std::expected<FramePtr, Refusal> pull();
  // Un saut : le graphe est jeté, refait à la prochaine image.
  void reset();
  bool active() const;  // une image entrelacée est passée depuis l'ouverture ou le dernier saut

 private:
  struct Graph;
  std::unique_ptr<Graph> graph_;
  bool active_ = false;
};

}  // namespace ajp::media
