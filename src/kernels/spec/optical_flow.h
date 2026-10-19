// Contrat du mouvement tiré du flux optique matériel (NVIDIA Optical Flow, l'OFA des RTX) :
// deux flots, de a vers b et de b vers a, deviennent le mouvement au milieu de la paire que le
// warp de spec/warp.h attend (f0 vers a, f1 vers b, masque poids de a). RIFE n'y tourne pas :
// le flux matériel coûte bien moins cher, au prix d'un mouvement moins fin.
//
// Écrit pour ce dépôt. Tests : tests/unit/kernels/optical_flow_test.cpp.
#pragma once

#include <cstdint>

namespace ajp::kernels {

// Un flot du flux matériel : un vecteur par bloc de grid × grid pixels de la luma (grille de
// sortie 1, 2 ou 4), en S10.5 (5 bits de fraction), en pixels de la luma. Le vecteur du bloc
// (i, j) part du centre du bloc, ((i + 0,5) grid − 0,5 ; (j + 0,5) grid − 0,5) : a(x) ≈ b(x + F(x))
// pour le flot avant F, b(x) ≈ a(x + B(x)) pour le flot arrière B.
inline constexpr float kFlowFraction = 32.0F;

// Un vecteur S10.5 en pixels de la grille (l'unité du mouvement de spec/warp.h, dont la fenêtre
// est la grille entière).
constexpr float flow_to_grid(std::int16_t s10_5, int grid) {
  return static_cast<float>(s10_5) / (kFlowFraction * static_cast<float>(grid));
}

// Le mouvement au milieu, pour chaque point p de la grille, tout en pixels de la grille, flots
// lus en bilinéaire bornés à la grille (spec/warp.h) :
//
// 1. Déplacement de toute la paire m, vu du milieu, par point fixe de kMidFlowIterations pas :
//    depuis F : m = F(p), puis m = F(p − m/2) (le point parti de p − m/2 arrive en p + m/2) ;
//    depuis B : n = −B(p), puis n = −B(p + n/2).
// 2. Accord d'un déplacement v avec chaque image : l'échantillon de a pris en p − v/2 doit
//    bouger de v, celui de b pris en p + v/2 revenir de v :
//    ea(v) = |F(p − v/2) − v|, eb(v) = |B(p + v/2) + v| ;
//    confiance c(e) = 1 / (1 + (e / kFlowConsistency)²), score s(v) = c(ea(v)) c(eb(v)).
//    mid est l'estimation au meilleur score, m à égalité : un mélange des deux flouterait le
//    bord des objets, où l'une des deux suit le fond.
// 3. f0 = −mid/2 (vers a), f1 = +mid/2 (vers b).
// 4. Masque : c(ea(mid)) / (c(ea(mid)) + c(eb(mid))). Un fond découvert entre a et b est caché
//    dans a par l'objet qui part : l'échantillon de a bouge avec l'objet, pas avec le fond, et le
//    masque penche vers b ; un fond que l'objet recouvre penche vers a. Deux côtés d'accord : 1/2.
inline constexpr int kMidFlowIterations = 3;
inline constexpr float kFlowConsistency = 0.5F;  // pixels de la grille

// Grille de sortie préférée du flux matériel, la plus fine que l'appareil propose parmi 2 et 4 :
// à 720p, 640 × 360 vecteurs (grille 2) suffisent au warp, et la grille 1 coûte quatre fois plus.
inline constexpr int kPreferredFlowGrid = 2;

// Vecteurs d'une grille pour une luma de `pixels` : les blocs du bord peuvent être partiels.
constexpr int flow_cells(int pixels, int grid) {
  return (pixels + grid - 1) / grid;
}

}  // namespace ajp::kernels
