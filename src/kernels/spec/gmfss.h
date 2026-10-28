// Contrat de GMFSS Fortuna en temps réel : le réseau coupé en deux graphes ONNX autour du
// softmax splatting, que les backends exécutent eux-mêmes (le seul morceau sans export ONNX).
//
// Repris de vsgmfss_fortuna 1.0.0 (HolyWu, MIT), GMFSS.py et softsplat.py :
//   reuse() : FeatureNet sur les deux images, GMFlow dans les deux sens à mi-résolution,
//             MetricNet (tanh × 10) ;
//   forward() : six softsplat « soft » (images et caractéristiques, trois niveaux, deux côtés),
//               puis GridNet ;
//   softsplat_out : addition bilinéaire en FP32 par atomicAdd, normalisation + 1e-7.
// L'export : tools/gmfss/export_gmfss.py. Tests : tests/unit/kernels/gmfss_test.cpp.
#pragma once

namespace ajp::kernels {

// Les images entrent comme celles de RIFE : RGB [0, 1] de la conversion de RIFE, remplies au
// multiple de 64 et centrées (rife_geometry), bords noirs ; la sortie se recadre pareil.
// vsgmfss étire l'image au multiple de 64 au lieu de la remplir : la fenêtre garde ici ses
// pixels exacts.

// Le graphe « paire » (une fois par paire) :
//   entrée « frames », 1 × 6 × ph × pw : image a en RGB (plans 0-2), image b (plans 3-5) ;
//   sorties, pour le côté k (0 : depuis a, 1 : depuis b) et le niveau l (1, 2, 3, soit 1/2,
//   1/4 et 1/8 de ph × pw) :
//     « source{k}_{l} » : ce qui est splatté, C_l plans (niveau 1 : l'image réduite de moitié
//                         puis les caractéristiques, 3 + 64 ; niveau 2 : 128 ; niveau 3 : 192) ; au
//                         niveau 1 des exports de version 2 (app/play/gmfss_model.h), pas de tenseur
//                         de 67 plans : « feature{k}_1 » (les 64 caractéristiques) et « half{k} » (les
//                         3 plans de l'image) se splattent chacun dans sa tranche de « warped{k}_1 »,
//                         l'image d'abord, ce qui rend le même résultat (chaque plan est normalisé par
//                         le poids de son seul pixel),
//     « motion{k}_{l} » : 3 plans, le flot fx, fy en pixels du niveau et la métrique z,
//                         tous pour t = 1 (flot de a vers b au côté 0, de b vers a au côté 1) ;
//     « half{k} » : l'image réduite de moitié (bilinéaire), 3 plans.
// Le graphe « fusion » (une fois par image intermédiaire) :
//   entrées « half0 », « half1 », « warped{k}_{l} » (source{k}_{l} splattée à l'instant) ;
//   sortie « out », 1 × 3 × ph × pw, bornée à [0, 1].
// Les dimensions variables des deux graphes s'écrivent « H », « W », « H/2 », « W/8 »… : le
// lecteur les fixe à la taille remplie avant de construire le moteur.
inline constexpr int kGmfssLevels = 3;
inline constexpr int kGmfssFramePlanes = 6;

// Plans de source{k}_{l} au niveau l de 1 à kGmfssLevels (C_l ci-dessus).
constexpr int gmfss_source_channels(int level) {
  return level == 1 ? 3 + 64 : level == 2 ? 128 : 192;
}
inline constexpr int kGmfssMotionChannels = 3;

// Instant du côté k : t depuis a, 1 − t depuis b.
constexpr float gmfss_side_time(int side, float t) {
  return side == 0 ? t : 1.0F - t;
}

// Softmax splatting à l'instant s du côté (softsplat « soft ») : pour chaque pixel source
// (x, y), en FP32,
//   fx' = s fx, fy' = s fy, z' = s z ; X = x + fx', Y = y + fy' (rien si X ou Y n'est pas fini) ;
//   x0 = ⌊X⌋, y0 = ⌊Y⌋ ; poids nord-ouest (x0 + 1 − X)(y0 + 1 − Y), nord-est (X − x0)(y0 + 1 − Y),
//   sud-ouest (x0 + 1 − X)(Y − y0), sud-est (X − x0)(Y − y0), chacun ajouté seulement si son
//   pixel est dans le plan ;
//   accumulation de source_c × e^z' × poids dans le plan c et de e^z' × poids dans le poids.
// Sortie : plan c accumulé / (poids accumulé + kGmfssNormalizeEpsilon). Un pixel que rien
// n'atteint vaut 0. L'ordre des additions est libre (atomiques) : les ports s'écartent de la
// référence par les arrondis de cet ordre seulement.
inline constexpr float kGmfssNormalizeEpsilon = 1e-7F;

// La métrique sort de tanh × 10 : e^z' reste dans [e^-10, e^10], sans débordement en FP32.
inline constexpr float kGmfssMetricLimit = 10.0F;

// Le mouvement sur une autre grille que la source (soft_splat_at) : le pixel source (x, y) lit le
// mouvement en u = (x + 1/2) / scale − 1/2 + ox, v = (y + 1/2) / scale − 1/2 + oy, bilinéaire
// borné au plan du mouvement (centres des pixels aux entiers, comme spec/warp.h), le flot lu
// multiplié par scale, la métrique telle quelle. scale 1 et décalages nuls sur une grille de même
// taille : le mouvement du pixel lui-même, tel quel (un flot non fini n'apporte alors rien, sans
// toucher ses voisins). Ainsi le flot du niveau 1 (moitié de la taille remplie) pousse l'image de
// travail (scale 2, ox = gauche / 2, oy = haut / 2 de rife_geometry) ou son agrandissement SPAN
// (scale 4, mêmes décalages).
// Couverture : avec un plan de sortie de plus que la source, ce plan vaut
// poids accumulé / (poids accumulé + kGmfssNormalizeEpsilon) : 1 où une source arrive, 0 où rien
// n'arrive.
struct MotionGrid {
  float scale = 1.0F;
  float x = 0.0F;
  float y = 0.0F;
};

// Déformation après SPAN (gmfss_compose) : SPAN agrandit chaque image source une fois, puis
// l'image intermédiaire se compose des sources agrandies splattées par le flot du niveau 1,
// corrigées du résidu de la fusion : out = H + up(F − L), H le mélange des sources agrandies
// splattées, L le même mélange des sources à la taille de travail, F la sortie de la fusion dans
// sa fenêtre, up l'agrandissement bilinéaire. Là où rien n'arrive, H et L sont nuls : la fusion
// agrandie reste seule. Pour le pixel (X, Y) de la sortie (scale × la taille de travail, scale
// entier) :
//   xw = (X + 1/2) / scale − 1/2, yw de même ;
//   chaque côté k donne (R, G, B, c) de sa source agrandie en (X, Y) et de sa source de travail en
//   (xw, yw) (bilinéaire borné au plan) ; poids w0 = (1 − t) c0, w1 = t c1 ;
//   M = (w0 P0 + w1 P1) / (w0 + w1 + kGmfssNormalizeEpsilon), sur chaque grille (H, L) ;
//   F lu en (xw + fenêtre.x, yw + fenêtre.y), bilinéaire borné à la fenêtre ;
//   out = clamp(H + F − L, 0, 1), par canal.

}  // namespace ajp::kernels
