// Contrat de DistilDRBA en temps réel : un RIFE à trois images (routineLife1/DistilDRBA, MIT)
// entraîné à imiter DRBA, exporté en un seul graphe par tools/drba/export_distil_drba.py.
//
// Repris de DistilDRBA, infer.py et models/vfi.py :
//   l'image img1 est le centre ; l'instant tau de [0,5 ; 1,5] se lit entre le milieu de la paire
//   (img0, img1) et celui de la paire (img1, img2), tau = 1 étant img1 elle-même ;
//   la tête du flux répète img0 (I0, I0, I1), la fin répète img2 (I0, I1, I1) ;
//   une coupe de scène n'est pas interpolée.
// Le réseau règle l'instant de chaque pixel d'après le mouvement des deux côtés de img1 : un
// personnage animé en deux tient sa pose pendant que le fond glisse.
#pragma once

namespace ajp::kernels {

// Le graphe, entrées dans l'ordre :
//   « pair01 », 1 × 7 × ph × pw : rife_input(img0, img1, u), en 7 plans (RIFE v2) ;
//   « pair12 », 1 × 7 × ph × pw : rife_input(img1, img2, u) ;
// avec u = tau / 2, dans ]0, 1[ comme l'instant que rife_input accepte. Sortie « out »,
// 1 × 3 × ph × pw, recadrée par rife_crop. Plans remplis comme ceux de RIFE (rife_geometry, au
// multiple de 64, ou 128 pour un export à l'échelle 0,5) ; « H » et « W » sont fixés par le
// lecteur avant la construction du moteur.
inline constexpr int kDistilDrbaInputPlanes = 7;

// L'appel du réseau pour l'instant t de ]0, 1[ entre les images a et b (a puis b) : centré sur a
// (img0 = l'image d'avant a, img1 = a, img2 = b) avant le milieu, sur b (img0 = a, img1 = b,
// img2 = l'image d'après b) à partir du milieu, comme les fenêtres [idx − 0,5 ; idx + 0,5)
// d'infer.py.
struct DrbaCall {
  bool on_b = false;  // centré sur b : img2 est l'image d'après b
  float u = 0.75F;    // plan 6 des deux entrées, tau / 2
};

constexpr DrbaCall drba_call(float t) {
  if (t < 0.5F) return {.on_b = false, .u = (1.0F + t) / 2.0F};
  return {.on_b = true, .u = t / 2.0F};
}

}  // namespace ajp::kernels
