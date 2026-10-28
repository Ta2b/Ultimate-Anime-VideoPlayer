// Contrat des fonctions de transfert (PQ, HLG, sRGB, BT.1886) et des primaires BT.709 ↔ BT.2020.
//
// Sources : SMPTE ST 2084 et ITU-R BT.2100 (PQ, HLG), IEC 61966-2-1 (sRGB), ITU-R BT.1886,
// ITU-R BT.709, BT.2020 et BT.2087 (conversion de gamut en lumière linéaire). Les constantes PQ
// sont celles du hook anime-hdr-grade.hook du lecteur actuel (ta2b/mpv-animejanai-hdr).
// Tests : tests/unit/kernels/transfer_test.cpp.
#pragma once

#include <array>
#include <cstddef>

namespace ajp::kernels {

// PQ : E' signal [0, 1] ↔ luminance affichée [0, 10000] cd/m².
//   EOTF : p = E'^(1/m2), Y = (max(p − c1, 0) / (c2 − c3 p))^(1/m1) × 10000 ;
//   inverse : y = Y / 10000, E' = ((c1 + c2 y^m1) / (1 + c3 y^m1))^m2.
// Entrées bornées à leur domaine. Toutes ces constantes sont exactes en binaire.
namespace pq {
inline constexpr float kM1 = 2610.0F / 16384.0F;          // 0,1593017578125
inline constexpr float kM2 = 2523.0F / 4096.0F * 128.0F;  // 78,84375
inline constexpr float kC1 = 3424.0F / 4096.0F;           // 0,8359375
inline constexpr float kC2 = 2413.0F / 4096.0F * 32.0F;   // 18,8515625
inline constexpr float kC3 = 2392.0F / 4096.0F * 32.0F;   // 18,6875
inline constexpr float kPeakNits = 10000.0F;
}  // namespace pq

// HLG (BT.2100), relatif à la scène. OETF : E' = sqrt(3E) pour E ≤ 1/12, a ln(12E − b) + c sinon.
// OOTF : Fd = Lw Ys^(γ − 1) E par composante, Ys = luma BT.2020 de la scène,
// γ = 1,2 + 0,42 log10(Lw / 1000), noir affiché 0. Lw dans [kMinPeak, kMaxPeak].
namespace hlg {
inline constexpr float kA = 0.17883277F;
inline constexpr float kB = 0.28466892F;  // 1 − 4a
inline constexpr float kC = 0.55991073F;  // 0,5 − a ln(4a)
inline constexpr float kKnee = 1.0F / 12.0F;
inline constexpr float kGammaAt1000 = 1.2F;
inline constexpr float kGammaSlope = 0.42F;
inline constexpr float kMinPeak = 400.0F;
inline constexpr float kMaxPeak = 2000.0F;
}  // namespace hlg

// sRGB (IEC 61966-2-1) : L = V / 12,92 pour V ≤ 0,04045, ((V + 0,055) / 1,055)^2,4 sinon.
namespace srgb {
inline constexpr float kEncodedKnee = 0.04045F;
inline constexpr float kLinearKnee = 0.0031308F;
inline constexpr float kSlope = 12.92F;
inline constexpr float kOffset = 0.055F;
inline constexpr float kGamma = 2.4F;
}  // namespace srgb

// BT.1886 : L = a max(V + b, 0)^γ, γ = 2,4, a = (Lw^(1/γ) − Lb^(1/γ))^γ,
// b = Lb^(1/γ) / (Lw^(1/γ) − Lb^(1/γ)). Blanc Lw > noir Lb ≥ 0, en cd/m² ou relatifs.
namespace bt1886 {
inline constexpr float kGamma = 2.4F;
}  // namespace bt1886

// Matrice 3 × 3 ligne par ligne, en double : les ports la reçoivent en float.
struct Mat3 {
  std::array<double, 9> m{};

  static constexpr std::size_t index(int row, int column) {
    return static_cast<std::size_t>(row) * 3 + static_cast<std::size_t>(column);
  }
  constexpr double at(int row, int column) const { return m[index(row, column)]; }
};

constexpr Mat3 multiply(const Mat3& a, const Mat3& b) {
  Mat3 out;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out.m[Mat3::index(r, c)] = a.at(r, 0) * b.at(0, c) + a.at(r, 1) * b.at(1, c) + a.at(r, 2) * b.at(2, c);
    }
  }
  return out;
}

// Inverse par la comatrice ; les matrices de gamut ne sont jamais singulières.
constexpr Mat3 inverse(const Mat3& a) {
  const auto cof = [&a](int r0, int r1, int c0, int c1) {
    return a.at(r0, c0) * a.at(r1, c1) - a.at(r0, c1) * a.at(r1, c0);
  };
  const double c00 = cof(1, 2, 1, 2);
  const double c01 = -cof(1, 2, 0, 2);
  const double c02 = cof(1, 2, 0, 1);
  const double det = a.at(0, 0) * c00 + a.at(0, 1) * c01 + a.at(0, 2) * c02;
  return {{c00 / det, -cof(0, 2, 1, 2) / det, cof(0, 1, 1, 2) / det,  //
           c01 / det, cof(0, 2, 0, 2) / det, -cof(0, 1, 0, 2) / det,  //
           c02 / det, -cof(0, 2, 0, 1) / det, cof(0, 1, 0, 1) / det}};
}

struct Chromaticity {
  double x = 0.0;
  double y = 0.0;
};

struct Gamut {
  Chromaticity red;
  Chromaticity green;
  Chromaticity blue;
  Chromaticity white;
};

inline constexpr Chromaticity kD65{.x = 0.3127, .y = 0.3290};
inline constexpr Gamut kBt709Gamut{
    .red = {0.640, 0.330}, .green = {0.300, 0.600}, .blue = {0.150, 0.060}, .white = kD65};
inline constexpr Gamut kBt2020Gamut{
    .red = {0.708, 0.292}, .green = {0.170, 0.797}, .blue = {0.131, 0.046}, .white = kD65};

// RGB linéaire → XYZ : colonnes des primaires (x/y, 1, z/y), mises à l'échelle pour que
// RGB = (1, 1, 1) donne le blanc de Y = 1.
constexpr Mat3 rgb_to_xyz(const Gamut& g) {
  const auto column = [](const Chromaticity& c) {
    return std::array{c.x / c.y, 1.0, (1.0 - c.x - c.y) / c.y};
  };
  const auto r = column(g.red);
  const auto gr = column(g.green);
  const auto b = column(g.blue);
  const Mat3 primaries{{r[0], gr[0], b[0], r[1], gr[1], b[1], r[2], gr[2], b[2]}};
  const auto w = column(g.white);
  const Mat3 inv = inverse(primaries);
  const double sr = inv.at(0, 0) * w[0] + inv.at(0, 1) * w[1] + inv.at(0, 2) * w[2];
  const double sg = inv.at(1, 0) * w[0] + inv.at(1, 1) * w[1] + inv.at(1, 2) * w[2];
  const double sb = inv.at(2, 0) * w[0] + inv.at(2, 1) * w[1] + inv.at(2, 2) * w[2];
  return {
      {r[0] * sr, gr[0] * sg, b[0] * sb, r[1] * sr, gr[1] * sg, b[1] * sb, r[2] * sr, gr[2] * sg, b[2] * sb}};
}

// Conversion de gamut en lumière linéaire, même blanc : XYZ de la source puis RGB de la cible.
constexpr Mat3 gamut_conversion(const Gamut& from, const Gamut& to) {
  return multiply(inverse(rgb_to_xyz(to)), rgb_to_xyz(from));
}

inline constexpr Mat3 kBt709ToBt2020 = gamut_conversion(kBt709Gamut, kBt2020Gamut);
inline constexpr Mat3 kBt2020ToBt709 = gamut_conversion(kBt2020Gamut, kBt709Gamut);

}  // namespace ajp::kernels
