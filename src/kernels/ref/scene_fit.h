// L'ajustement de la caméra de la carte sur ses tuiles (ref::scene_camera_of, kernels/ref/scene_flow.h) :
// interne à la référence, entre scene_flow.cpp (les tuiles, le classement) et scene_fit.cpp (l'ajustement).
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "kernels/spec/scene_flow.h"

namespace kirari::kernels::ref::scene_fit {

// Une tuile lue sur l'hôte : ses moyennes.
struct TileView {
  float cx = 0.0F;  // centre, pixels du flux
  float cy = 0.0F;
  float mx = 0.0F;  // vecteur moyen, pixels du flux
  float my = 0.0F;
  float spread = 0.0F;   // écart type des vecteurs
  float matched = 0.0F;  // part de vecteurs sûrs
  float texture = 0.0F;  // par pixel
  float change = 0.0F;   // |b − a| moyen
  float shift = 0.0F;    // (b − a) moyen
  float cost = 0.0F;     // coût moyen des vecteurs
  bool valid = false;
};

inline bool fit_candidate(const TileView& v) {
  return v.valid && v.matched >= scene::kFitMatched && v.texture >= scene::kFitTexture &&
         v.spread <= scene::kSpreadMax;
}

inline float texture_weight(const TileView& v) {
  return std::clamp(v.texture / scene::kTextureFloor, 0.0F, 1.0F);
}

// Le déplacement du modèle en (x, y) : d = (c0 + c1 x + c2 y, c3 + c4 x + c5 y), pixels du flux.
struct Displacement {
  std::array<double, 6> c{};
  double dx(double x, double y) const { return c[0] + c[1] * x + c[2] * y; }
  double dy(double x, double y) const { return c[3] + c[4] * x + c[5] * y; }
};

struct CameraFit {
  bool fit = false;
  Displacement model;  // en coordonnées centrées et réduites
  double residual = 0.0;
  std::vector<std::uint8_t> decor;  // par tuile : gardée au dernier tour
};

// La médiane, celle de numpy : la moyenne des deux du milieu pour un compte pair.
double median_of(std::vector<double> values);

// La caméra sur les tuiles views, repère centré en (cx, cy) et réduit par scale ; unit : pixels du flux par
// pixel à 720p. fit faux s'il reste trop peu de tuiles.
CameraFit fit_camera(const std::vector<TileView>& views, double cx, double cy, double scale, double unit);

}  // namespace kirari::kernels::ref::scene_fit
