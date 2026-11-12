// Le modèle de l'écran (H1) : ce que le panneau affiche, et dans quelles unités le lecteur doit le viser.
//
// - Échelle contenu → panneau. KWin affiche le blanc de référence du contenu PQ (203 cd/m², BT.2408) au blanc
//   de référence réglé par l'utilisateur (la « luminosité SDR ») : le panneau montre contenu × référence /
//   203. Le pic du panneau, en unités du contenu, vaut donc pic × 203 / référence. Windows affiche le PQ tel
//   quel (échelle 1).
// - Pic en petite fenêtre et limiteur (ABL) : L_max(surface), la luminance la plus haute que le panneau tient
//   quand une part `surface` de l'image est blanche. Une dalle OLED suit la courbe d'une QD-OLED 27" 4K
//   (RTINGS, PG27UCDM, dalle de la MAG 272U E16) : 1016 / 460 / 367 / 310 / 261 cd/m² à 2 / 10 / 25 / 50 /
//   100 %, ramenée à son pic ; un LCD tient son pic sur toute la surface.
#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace ajp::render {

inline constexpr double kContentReferenceNits = 203.0;

// Le limiteur par surface est dans le contrat d'écran (DisplayContract::abl, default_abl).
struct DisplayModel {
  double panel_peak_nits = 1000.0;                // le pic du panneau, en cd/m² affichées
  double reference_nits = kContentReferenceNits;  // le blanc de référence du compositeur

  // Panneau = contenu × scale().
  double scale() const;
  // Le pic à viser dans le signal PQ.
  double content_peak_nits() const;
};

// --hdr-metadata : ce que la sortie annonce à KWin et ce que la passe « kwin » du grade vise.
// max_content_nits est le pic du contenu (annoncé à la swapchain : KWin le ramène à son blanc, panneau =
// contenu × référence / 203) ; panel_peak_nits, le même pic en nits affichés, que la passe écrête.
struct KwinHint {
  double panel_peak_nits = 0.0;
  double max_content_nits = 0.0;
};
KwinHint kwin_hint(double content_peak_nits, double reference_nits);

// Le contrat d'écran (chantier 8.2) : ce que le lecteur sait du panneau, en nits affichés, et ce qui s'en
// déduit en unités du contenu. k = référence / 203 : KWin affiche le contenu × k (H1 ci-dessus).
struct Chromaticities {
  double rx = 0.0, ry = 0.0, gx = 0.0, gy = 0.0, bx = 0.0, by = 0.0, wx = 0.0, wy = 0.0;
  friend bool operator==(const Chromaticities&, const Chromaticities&) = default;
};
inline constexpr std::array<double, 5> kAblAreas{0.02, 0.10, 0.25, 0.50, 1.00};
struct DisplayContract {
  double reference_nits = kContentReferenceNits;  // blanc de référence du compositeur
  double panel_peak_nits = 1000.0;                // pic du panneau, cd/m² affichées
  std::array<double, 5> abl{};                    // L_max aux surfaces kAblAreas, cd/m² affichées
  std::optional<Chromaticities> primaries;        // de l'EDID ; rien : inconnues
  double min_nits = 0.0;                          // noir annoncé, cd/m² affichées
  bool oled = false;
  bool from_compositor = false;  // la référence et le pic lus chez le compositeur (KDE), pas supposés
  // D'où vient le pic, pour le journal (A10-19) : sans KDE, ni marge de SDL, ni EDID, les 1000 nits ne sont
  // qu'une valeur par défaut.
  enum class PeakSource : std::uint8_t { Default, Compositor, Headroom, Edid };
  PeakSource peak_source = PeakSource::Default;
  // D'où vient la courbe du limiteur (C11-04) : le plein champ du système (DXGI), de DisplayID, le MaxFALL de
  // l'EDID, ou la table QD-OLED par défaut.
  enum class AblSource : std::uint8_t { Default, System, DisplayId, Edid };
  AblSource abl_source = AblSource::Default;

  double k() const { return reference_nits > 0.0 ? reference_nits / kContentReferenceNits : 1.0; }
  double content_peak_nits() const { return panel_peak_nits / k(); }      // C_pk
  double content_frame_average_nits() const { return abl.back() / k(); }  // MaxFALL du contenu
  friend bool operator==(const DisplayContract&, const DisplayContract&) = default;
};
// Les métadonnées HDR10 du contrat v2 (8.4), en unités du contenu : KWin affiche le contenu × k, il les
// ramène donc à l'écran telles quelles. Primaires de l'EDID, sinon Display P3 ; noir et MaxFALL divisés par k
// ; max_luma = max_cll = C_pk.
struct ContractMetadata {
  Chromaticities primaries;
  double min_luma = 0.0;
  double max_luma = 0.0;
  double max_cll = 0.0;
  double max_fall = 0.0;
  friend bool operator==(const ContractMetadata&, const ContractMetadata&) = default;
};
inline constexpr Chromaticities kDisplayP3{0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290};
ContractMetadata contract_metadata(const DisplayContract& contract);

// L_max par surface faute de mesure : pour un OLED, la courbe QD-OLED relevée ramenée au pic du panneau (pic
// × oled_abl_ratio, C41-06 : une dalle de 800 cd/m² ne tient pas en plein champ les 261 d'une QD-OLED de
// 1016) ; le pic partout pour un LCD.
std::array<double, 5> default_abl(double panel_peak_nits, bool oled);
// L_max par surface quand l'écran dit ce qu'il tient plein champ (C11-04 : DXGI MaxFullFrameLuminance,
// DisplayID, MaxFALL de l'EDID) : la forme de la courbe QD-OLED, en log, ramenée entre le pic (2 %) et cette
// valeur (100 %). Plein champ au pic ou plus : le pic partout.
std::array<double, 5> abl_to_full_frame(double panel_peak_nits, double full_frame_nits);

// La part de L_max(2 %) que la courbe QD-OLED garde à cette surface (1 à 2 % et moins, 0,257 à 100 %).
double oled_abl_ratio(double area);

}  // namespace ajp::render
