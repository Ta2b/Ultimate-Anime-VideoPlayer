// Contrat du fondu local du mouvement de RIFE (tri des paires par la carte de la scène, rife_triage.h) : là
// où le flux optique de NVIDIA ne trouve pas de correspondance (effet, occlusion, paire impossible), le warp
// de RIFE fait des fantômes ; le mouvement y est tiré vers un fondu enchaîné de a et b, sans déplacement.
//
// Entrée : le mouvement de RIFE (spec/warp.h : 4 plans de déplacements f0x, f0y, f1x, f1y et le masque, poids
// de a, tous pw × ph, la fenêtre de l'image dedans), une grille de doute cols × rows (1 ≤ cols, rows ≤
// kDoubtMaxCells) de poids 8 bits posée sur la fenêtre (case (c, r) : son centre en
// ((c + 0,5) fw / cols, (r + 0,5) fh / rows) de la fenêtre), et le masque du fondu fade (poids de a).
// Sortie : un mouvement de même forme. Pour chaque point (x, y) des plans :
//   hors de la fenêtre : recopié ;
//   dans la fenêtre, (u, v) = ((x − wx + 0,5) cols / fw − 0,5, (y − wy + 0,5) rows / fh − 0,5) ;
//   w = lecture bilinéaire de la grille en (u, v), bords en clamp (spec/warp.h), en poids / 255 ;
//   w ≤ 0 : recopié ; w ≥ 1 : déplacements nuls, masque = fade (l'entrée n'est pas lue, NaN compris) ;
//   sinon : déplacements × (1 − w), masque + (fade − masque) w.
// Le masque du fondu : 0,5 pour un mouvement estimé au milieu (la loi de mélange de spec/warp.h donne alors
// à l'instant t un fondu qui penche vers l'image la plus proche, celui de RIFE sur une paire sans
// mouvement) ; 1 − t pour un mouvement lu tel quel à l'instant t (at_instant : DistilDRBA, RIFE direct).
// Tests : tests/unit/kernels/motion_doubt_test.cpp ; port : gpu/cuda/backend/doubt_kernels.cu.
#pragma once

#include <cstdint>

#include "kernels/spec/fp_off_begin.h"  // IWYU pragma: keep

namespace kirari::kernels {

// Côté maximal de la grille de doute : 48 × 48 poids 8 bits tiennent dans les paramètres d'un noyau CUDA
// (4 Ko). La carte de la scène a des tuiles de 32 pixels de travail en 1080p (60 × 34) : l'hôte les
// regroupe en gardant le plus fort doute de chaque groupe (rife_triage.cpp).
inline constexpr int kDoubtMaxCells = 48;

// Le poids d'une case, de 0 (RIFE tel quel) à kDoubtFull (fondu entier).
inline constexpr std::uint8_t kDoubtFull = 255;

// Lecture bilinéaire de la grille au point (u, v) de la grille, bords en clamp, en [0, 1].
constexpr float doubt_at(const std::uint8_t* weights, int cols, int rows, float u, float v) {
  const float cu = u < 0.0F ? 0.0F : (u > static_cast<float>(cols - 1) ? static_cast<float>(cols - 1) : u);
  const float cv = v < 0.0F ? 0.0F : (v > static_cast<float>(rows - 1) ? static_cast<float>(rows - 1) : v);
  const int c0 = static_cast<int>(cu);
  const int r0 = static_cast<int>(cv);
  const int c1 = c0 + 1 < cols ? c0 + 1 : cols - 1;
  const int r1 = r0 + 1 < rows ? r0 + 1 : rows - 1;
  const float fx = cu - static_cast<float>(c0);
  const float fy = cv - static_cast<float>(r0);
  const float top = static_cast<float>(weights[r0 * cols + c0]) * (1.0F - fx) +
                    static_cast<float>(weights[r0 * cols + c1]) * fx;
  const float bottom = static_cast<float>(weights[r1 * cols + c0]) * (1.0F - fx) +
                       static_cast<float>(weights[r1 * cols + c1]) * fx;
  return (top * (1.0F - fy) + bottom * fy) / static_cast<float>(kDoubtFull);
}

// La position d'un point de la fenêtre (relatif à son coin) dans la grille, sur un axe.
constexpr float doubt_coord(int x, int cells, int window) {
  return (static_cast<float>(x) + 0.5F) * static_cast<float>(cells) / static_cast<float>(window) - 0.5F;
}

// Un déplacement et un masque tirés vers le fondu au poids w (w déjà lu ; w ≥ 1 n'utilise pas l'entrée).
constexpr float doubt_flow(float flow, float w) {
  return w >= 1.0F ? 0.0F : (w <= 0.0F ? flow : flow * (1.0F - w));
}
constexpr float doubt_mask(float mask, float fade, float w) {
  return w >= 1.0F ? fade : (w <= 0.0F ? mask : mask + (fade - mask) * w);
}

}  // namespace kirari::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
