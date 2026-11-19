// Contrat du mouvement d'une caméra (tri des paires de GMFSS, app/play/rife_triage.h) : une paire où la carte
// de la scène (spec/scene_flow.h) ne voit bouger que la caméra (panoramique pur), ou presque rien qui
// corresponde (paire impossible), se fait par le warp de l'étage, sans réseau. Le mouvement est écrit au
// format de spec/warp.h, estimé au milieu de la paire, et lu par warp_blend à chaque instant.
//
// Entrée : la caméra affine de la carte (SceneMotion::camera, en pixels de travail : le point (X, Y) de a est
// en (c0 + c1 X + c2 Y, c3 + c4 X + c5 Y) dans b), l'échelle s des pixels de travail aux pixels de la fenêtre
// (s > 0), et la fenêtre du mouvement dans ses plans. Sortie : pour chaque point (x, y) des plans :
//   hors de la fenêtre : déplacements nuls, masque 1/2 ;
//   dans la fenêtre, (X, Y) = ((x − wx) / s, (y − wy) / s) en pixels de travail ;
//   d = s (c0 + (c1 − 1) X + c2 Y, c3 + c4 X + (c5 − 1) Y), en pixels de la fenêtre ;
//   f0 = −d / 2 (vers a), f1 = d / 2 (vers b), masque 1/2.
// Premier ordre : d est lu au point du milieu plutôt qu'au point de a. Exact pour une translation ; l'écart
// vaut |A − I| |d| / 2, sous le pixel pour un zoom de caméra de 1 % et 100 pixels de mouvement.
// La caméra identité donne le fondu enchaîné de a et b : mouvement nul, masque 1/2 (spec/warp.h penche alors
// vers l'image la plus proche, comme le fondu de spec/motion_doubt.h).
// Tests : tests/unit/kernels/motion_camera_test.cpp ; port : gpu/cuda/backend/doubt_kernels.cu.
#pragma once

#include <array>

#include "kernels/spec/fp_off_begin.h"  // IWYU pragma: keep

namespace kirari::kernels {

// La caméra identité : le fondu enchaîné.
inline constexpr std::array<float, 6> kCameraStill{0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F};

// Le déplacement de a vers b au point (u, v) de la fenêtre (relatif à son coin), en pixels de la fenêtre.
struct CameraShift {
  float x = 0.0F;
  float y = 0.0F;
};
constexpr CameraShift camera_shift(const std::array<float, 6>& c, float scale, int u, int v) {
  const float px = static_cast<float>(u) / scale;
  const float py = static_cast<float>(v) / scale;
  return {.x = (c[0] + (c[1] - 1.0F) * px + c[2] * py) * scale,
          .y = (c[3] + c[4] * px + (c[5] - 1.0F) * py) * scale};
}

}  // namespace kirari::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
