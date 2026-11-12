// La peau des visages d'une image de la pré-analyse (media/shot_probe.h) : la cascade de visages
// (media/face_cascade.h) cherche sur la luma réduite à kFaceSearchHeight lignes, puis la palette de chaque
// visage se lit sous ses yeux (media/faces.h).
#pragma once

#include <string>
#include <vector>

#include "core/frame.h"
#include "media/face_cascade.h"
#include "media/faces.h"
#include "media/shot_cutter.h"
#include "media/shot_timeline.h"

namespace ajp::media {

// La hauteur de l'image où la cascade cherche : un visage de 24 lignes y fait 8 % de la hauteur de l'image.
inline constexpr int kFaceSearchHeight = 288;
// Une image sur kFaceEvery passe par la cascade (deux fois et demie par seconde à 24 images par seconde).
inline constexpr int kFaceEvery = 10;

// Les palettes des visages d'une image décodée en mémoire hôte (Nv12, Yuv420p, P010, Yuv420p10) ; vide hors
// de ces formats, sans visage, ou pour un visage dont moins du quart des points sont d'une teinte de
// peau.
std::vector<SkinTone> frame_skin_tones(const DecodedFrame& frame, const FaceCascade& cascade);

// La recherche d'une analyse : une image sur kFaceEvery, ses palettes posées dans ses mesures ; rien si elle
// est éteinte ou si la cascade embarquée ne se lit pas.
class FaceSearch {
 public:
  explicit FaceSearch(bool on) : cascade_(on ? FaceCascade::anime() : nullptr) {}
  void look(const DecodedFrame& frame, ShotFrame& measured);
  // Pour le journal de fin d'analyse : « , visages dans N plans » ; vide si la recherche est éteinte.
  std::string summary(const ShotTimeline& timeline) const;

 private:
  const FaceCascade* cascade_;
  int seen_ = 0;
};

}  // namespace ajp::media
