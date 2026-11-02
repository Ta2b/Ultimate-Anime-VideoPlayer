// Contrat de l'analyse d'une image pour le grade (analyse v2, H4 de l'audit) : une seule définition,
// mesurée par la pré-analyse des plans sur l'hôte (media/shot_probe.h) et par le noyau GLSL à
// l'ingestion (frame_stats.comp), la référence étant kernels/ref/frame_stats.cpp.
//
// Points mesurés : une grille de kGridW × kGridH points de l'image RGB non linéaire (R'G'B' dans
// [0, 1]), le point (gx, gy) lu en x = min((2 gx + 1) W / (2 kGridW), W − 1), de même en y.
// Bandes noires : les colonnes et les lignes de la grille dont la luma gamma reste sous
// kBarLuma, depuis chaque bord ; gardées entières si le reste fait moins du quart de la grille.
//
// Deux échelles :
// - gamma, celle de gpu::KernelOps::SceneStats : luma BT.709 des canaux encodés, histogramme de
//   kLumaBins cases, grille kZones × kZones de lumas moyennes, bleu et chroma moyens ;
// - linéaire : le signal décodé en cd/m² (SDR : BT.1886 γ 2,4 noir 0, blanc kSdrWhiteNits ;
//   PQ : ST 2084 ; HLG : OETF inverse puis OOTF à kHlgPeakNits). maxRGB dans les primaires de la
//   source, I d'ICtCp (BT.2100, sur les primaires BT.2020, SDR converti de BT.709) ; histogrammes
//   de kLinearBins cases sur le code PQ de maxRGB (une source PQ : son plus grand canal encodé) et
//   sur I ; APL (luminance moyenne) ; parts de
//   surface claire, de hautes lumières isolées et de peau.
#pragma once

#include <array>
#include <cstdint>

namespace ajp::kernels {

enum class FrameTransfer : std::uint8_t { Sdr, Pq, Hlg };

namespace frame_stats {

inline constexpr int kGridW = 128;
inline constexpr int kGridH = 72;
inline constexpr int kLumaBins = 64;
inline constexpr int kLinearBins = 128;
inline constexpr int kZones = 4;
inline constexpr float kBarLuma = 0.03F;
inline constexpr float kSdrWhiteNits = 203.0F;  // blanc papier SDR (BT.2408)
inline constexpr float kHlgPeakNits = 1000.0F;
// Surface claire : maxRGB au-dessus de kBrightNits.
inline constexpr float kBrightNits = 100.0F;
// Haute lumière isolée : maxRGB au-dessus de kBrightNits et d'au moins kIsolatedRatio fois la
// moyenne des maxRGB de ses huit voisins de la grille (bornés au cadre hors bandes).
inline constexpr float kIsolatedRatio = 2.0F;
// Peau (sur R'G'B') : r > g > b, r ≥ kSkinMinRed, teinte (g − b) / (r − b) dans
// [kSkinHueLo, kSkinHueHi], saturation (r − b) / r dans [kSkinSatLo, kSkinSatHi].
inline constexpr float kSkinMinRed = 0.25F;
inline constexpr float kSkinHueLo = 0.25F;
inline constexpr float kSkinHueHi = 0.75F;
inline constexpr float kSkinSatLo = 0.12F;
inline constexpr float kSkinSatHi = 0.65F;
// Centiles de maxRGB rendus, en pour cent.
inline constexpr std::array<float, 6> kPercentiles{1.0F, 5.0F, 50.0F, 95.0F, 99.0F, 99.9F};

}  // namespace frame_stats

// Ce que le noyau et la référence comptent, sur les points hors bandes. Disposition fixe, lue telle
// quelle depuis le tampon du noyau : uint32 et float de 4 octets, sans trou.
struct FrameCounts {
  std::array<std::uint32_t, frame_stats::kLumaBins> luma_hist{};
  std::array<std::uint32_t, frame_stats::kLinearBins> maxrgb_hist{};
  std::array<std::uint32_t, frame_stats::kLinearBins> i_hist{};
  std::array<float, frame_stats::kZones * frame_stats::kZones> zone_sum{};
  std::array<std::uint32_t, frame_stats::kZones * frame_stats::kZones> zone_count{};
  std::uint32_t points = 0;
  std::uint32_t bright = 0;
  std::uint32_t highlights = 0;
  std::uint32_t skin = 0;
  float luma_sum = 0.0F;
  float blue_sum = 0.0F;
  float chroma_sum = 0.0F;
  float nits_sum = 0.0F;                         // luminance linéaire
  float maxrgb_sum = 0.0F;                       // maxRGB linéaire
  std::uint32_t x0 = 0, x1 = 0, y0 = 0, y1 = 0;  // cadre hors bandes, en points de la grille
};
static_assert(sizeof(FrameCounts) == 4 * (64 + 128 + 128 + 16 + 16 + 4 + 5 + 4));

// Les mesures d'une image, tirées de FrameCounts (frame_stats_of).
struct FrameStats {
  // Échelle gamma.
  std::array<float, frame_stats::kLumaBins> luma_hist{};  // parts (somme 1)
  std::array<float, frame_stats::kZones * frame_stats::kZones> zones{};
  float mean = 0.0F;
  float blue = 0.0F;
  float chroma = 0.0F;
  // Échelle linéaire.
  std::array<float, frame_stats::kLinearBins> maxrgb_hist{};
  std::array<float, frame_stats::kLinearBins> i_hist{};
  std::array<float, frame_stats::kPercentiles.size()> maxrgb_nits{};  // centiles kPercentiles
  float apl_nits = 0.0F;
  float maxrgb_mean_nits = 0.0F;
  float bright = 0.0F;
  float highlights = 0.0F;
  float skin = 0.0F;
  // Cadre hors bandes, en parts de l'image (0 à 1).
  float left = 0.0F, right = 1.0F, top = 0.0F, bottom = 1.0F;
};

}  // namespace ajp::kernels
