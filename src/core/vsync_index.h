// Le choix du vsync où présenter une image, avec hystérésis (étape 6.6).
//
// Une image visée près du milieu de deux vsyncs part tantôt sur l'un, tantôt sur l'autre selon la
// gigue des dates : à 24 im/s sur 60 Hz, le motif 3-2 devient 3-2-3-3-2. Près du milieu, le sens
// d'arrondi déjà pris est donc gardé ; il ne change que lorsque la date sort de la bande
// [0,5 − h ; 0,5 + h] du côté opposé.
#pragma once

#include <cmath>
#include <cstdint>

namespace ajp::core {

inline constexpr double kVsyncHysteresis = 0.15;

class VsyncPicker {
 public:
  explicit VsyncPicker(double hysteresis = kVsyncHysteresis) : hysteresis_(hysteresis) {}

  // x : numéro fractionnaire du vblank visé (VblankGrid::index_of). Rend le numéro entier choisi.
  std::int64_t pick(double x) {
    const double floor_x = std::floor(x);
    const double fraction = x - floor_x;
    if (fraction < 0.5 - hysteresis_) {
      up_ = false;
    } else if (fraction > 0.5 + hysteresis_) {
      up_ = true;
    }
    return static_cast<std::int64_t>(floor_x) + (up_ ? 1 : 0);
  }

 private:
  double hysteresis_;
  bool up_ = false;
};

}  // namespace ajp::core
