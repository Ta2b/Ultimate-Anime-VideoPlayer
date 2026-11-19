// Le mouvement d'une image face à la précédente, d'après le flux optique de NVIDIA dans la pré-analyse des
// plans (media/probe_flow.h) : ce que le découpeur (media/shot_cutter.h) et le rythme des plans en lisent.
// Pixels de la luma réduite de la pré-analyse (256 × 144), blocs de 4 × 4.
#pragma once

namespace kirari::media {

// Le mouvement d'une image face à la précédente, sur ses blocs de 4 × 4.
struct ProbeMotion {
  // La part des blocs texturés de l'image qui ne retrouvent pas leur contenu dans l'image d'avant, de 0 à 1.
  float unmatched = 0.0F;
  // La part des blocs texturés (assez de détail pour juger) ; les aplats et les bandes noires n'en sont pas.
  float textured = 0.0F;
  // La translation dominante (médiane des vecteurs des blocs appariés) et l'amplitude moyenne des vecteurs
  // des blocs appariés, en pixels de la luma réduite.
  float shift_x = 0.0F;
  float shift_y = 0.0F;
  float mean_motion = 0.0F;
  // L'écart moyen des vecteurs des blocs appariés à la translation dominante, en pixels : le mouvement des
  // dessins face à la caméra (grand sur le sakuga, petit sur un panoramique).
  float residual = 0.0F;
  // La part des blocs appariés qui n'ont pas bougé (vecteur de moins d'un demi-pixel).
  float still = 0.0F;
  friend bool operator==(const ProbeMotion&, const ProbeMotion&) = default;
};

}  // namespace kirari::media
