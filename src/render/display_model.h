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

#include <optional>

namespace ajp::render {

inline constexpr double kContentReferenceNits = 203.0;

struct DisplayModel {
  double panel_peak_nits = 1000.0;                // le pic du panneau, en cd/m² affichées
  double reference_nits = kContentReferenceNits;  // le blanc de référence du compositeur
  bool oled = false;

  // Panneau = contenu × scale().
  double scale() const;
  // Le pic à viser dans le signal PQ.
  double content_peak_nits() const;
  // L_max(surface), surface dans ]0, 1], en cd/m² affichées.
  double max_nits(double area) const;
  // La moyenne d'image (MaxFALL) que tient le panneau plein écran, en unités du contenu ; rien pour un LCD.
  std::optional<double> frame_average_content_nits() const;
};

// --hdr-metadata : ce que la sortie annonce à KWin et ce que la passe « kwin » du grade vise.
// max_content_nits est le pic du contenu (annoncé à la swapchain : KWin le ramène à son blanc, panneau =
// contenu × référence / 203) ; panel_peak_nits, le même pic en nits affichés, que la passe écrête.
struct KwinHint {
  double panel_peak_nits = 0.0;
  double max_content_nits = 0.0;
};
KwinHint kwin_hint(double content_peak_nits, double reference_nits);

// La part de L_max(2 %) que la courbe QD-OLED garde à cette surface (1 à 2 % et moins, 0,257 à 100 %).
double oled_abl_ratio(double area);

}  // namespace ajp::render
