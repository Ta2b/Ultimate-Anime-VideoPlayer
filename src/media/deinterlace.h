// Désentrelacement des images logicielles (P3.25) : bwdif de libavfilter, une image par image
// (send_frame), les seules images marquées entrelacées traitées. bwdif lit l'image suivante : la
// sortie a une image de retard, rendue par flush() à la fin du flux.
//
// A15-02 : un flux 29,97 i/s entrelacé est presque toujours, en anime, du 23,976 télécinéma (3:2). Il passe
// par fieldmatch (les champs d'une même image recousus), bwdif sur les seules images restées peignées, puis
// decimate (l'image en double de chaque cycle de cinq jetée) : 23,976 images entières au lieu de 29,97 images
// refaites d'un champ.
#pragma once

#include <deque>
#include <expected>
#include <memory>

#include "core/types.h"
#include "media/av_util.h"

namespace ajp::media {

enum class FieldMode {
  Deinterlace,      // bwdif seul
  InverseTelecine,  // fieldmatch, bwdif sur les images peignées, decimate (A15-02)
  MatchFields,  // fieldmatch, bwdif sur les images peignées, sans decimate : 2:2, MBAFF progressif (C15-01)
};

class Deinterlacer {
 public:
  explicit Deinterlacer(FieldMode mode = FieldMode::Deinterlace);
  ~Deinterlacer();
  Deinterlacer(Deinterlacer&&) noexcept;
  Deinterlacer& operator=(Deinterlacer&&) noexcept;

  // Une image décodée, en microsecondes ; le graphe se refait si la taille ou le format change, après avoir
  // rendu les images qu'il retenait (A15-04).
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
  std::expected<FramePtr, Refusal> take();  // une image du graphe, en microsecondes
  std::expected<void, Refusal> rebuild(const AVFrame& frame);
  std::unique_ptr<Graph> graph_;
  std::deque<FramePtr> drained_;  // les images de l'ancien graphe, rendues avant celles du nouveau
  FieldMode mode_ = FieldMode::Deinterlace;
  bool active_ = false;
};

}  // namespace ajp::media
