// Contrat de GMFSS Fortuna Union (docs/interpolation.md) : le réseau coupé en deux graphes ONNX autour du
// softmax splatting, que les backends exécutent eux-mêmes (le seul morceau sans export ONNX). Le mélange s'en
// sert au milieu des paires fortes (RIFE fait ensuite les demi-paires), « GMFSS seul » à chaque instant.
//
// Repris de vsgmfss_fortuna 1.0.0 (HolyWu, MIT), GMFSS_union.py et softsplat.py :
//   reuse() : FeatureNet sur les deux images, GMFlow dans les deux sens au flot entier, MetricNet
//             (tanh × 10) ;
//   forward() : six softsplat « soft » (images et caractéristiques, trois niveaux, deux côtés), l'IFNet de
//               RIFE à l'instant, puis GridNet ;
//   softsplat_out : addition bilinéaire en FP32 par atomicAdd, normalisation + 1e-7.
// L'export : tools/gmfss/export_gmfss.py. Tests : tests/unit/kernels/gmfss_test.cpp.
#pragma once

#include "kernels/spec/fp_off_begin.h"  // IWYU pragma: keep

namespace kirari::kernels {

// Les images entrent comme celles de RIFE : RGB [0, 1] de la conversion de la chaîne, remplies au multiple
// de 64 et centrées (rife_geometry) ; la sortie se recadre pareil. vsgmfss étire l'image au multiple de 64
// au lieu de la remplir : la fenêtre garde ici ses pixels exacts. Le bord recopie le pixel le plus proche de
// l'image, pas le noir de RIFE : au flot réduit de moitié, le noir passait dans le flot près du bord et
// assombrissait les premières lignes de chaque image union.
//
// Le graphe « paire » (une fois par paire forte) :
//   entrée « frames », 1 × 6 × ph × pw : image a en RGB (plans 0-2), image b (plans 3-5) ;
//   sorties, pour le côté k (0 : depuis a, 1 : depuis b) et le niveau l (1, 2, 3, soit 1/2, 1/4 et 1/8 de
//   ph × pw) :
//     « half{k} » : l'image réduite de moitié (bilinéaire), 3 plans ;
//     « feature{k}_1 » : les 64 caractéristiques du niveau 1 ;
//     « source{k}_{l} », l = 2, 3 : les caractéristiques splattées, 128 et 192 plans ;
//     « motion{k}_{l} » : 3 plans, le flot fx, fy en pixels du niveau et la métrique z, tous pour t = 1
//                         (flot de a vers b au côté 0, de b vers a au côté 1).
// Le graphe « fusion » (une fois par image intermédiaire) :
//   entrées « time » (1 × 1 × 1 × 1, l'instant t), « half0 », « half1 », « warped{k}_{l} » (splattées à
//   l'instant du côté, gmfss_side_time) : au niveau 1, 3 + 64 plans, l'image splattée puis les
//   caractéristiques splattées (deux splats du même mouvement, chacun dans sa tranche, ce qui vaut le splat
//   des 67 plans : chaque plan est normalisé par le poids de son seul pixel) ; aux niveaux 2 et 3,
//   source{k}_{l} splattée ;
//   sortie « out », 1 × 3 × ph × pw, bornée à [0, 1].
// Les dimensions variables des deux graphes s'écrivent « H », « W », « H/2 », « W/8 »… : le lecteur les fixe
// à la taille remplie avant de construire le moteur.
// Multiple de la taille remplie : GMFlow divise par 64 ; une fusion dont l'IFNet est un RIFE du lecteur
// (tools/gmfss/ifnet_rife.py) le fait tourner à la demi-taille, qui doit être un multiple de 64. 1280 × 768
// et 1920 × 1152 (720p et 1080p) le sont déjà.
inline constexpr int kGmfssAlign = 128;
inline constexpr int kGmfssLevels = 3;
inline constexpr int kGmfssFramePlanes = 6;
inline constexpr int kGmfssImagePlanes = 3;
inline constexpr int kGmfssFeaturePlanes = 64;  // feature{k}_1

// Plans de warped{k}_{l} au niveau l de 1 à kGmfssLevels.
constexpr int gmfss_warped_channels(int level) {
  return level == 1 ? kGmfssImagePlanes + kGmfssFeaturePlanes : level == 2 ? 128 : 192;
}
inline constexpr int kGmfssMotionChannels = 3;

// Instant du côté k : t depuis a, 1 − t depuis b.
constexpr float gmfss_side_time(int side, float t) {
  return side == 0 ? t : 1.0F - t;
}

// Softmax splatting à l'instant s du côté (softsplat « soft ») : pour chaque pixel source (x, y), en FP32,
//   z' = s · clamp(z, −kGmfssMetricLimit, kGmfssMetricLimit) (NaN : la borne basse) ;
//   X = x + s fx, Y = y + s fy (rien si X ou Y n'est pas dans ]−1, taille[, NaN compris) ;
//   x0 = ⌊X⌋, y0 = ⌊Y⌋ ; poids nord-ouest (x0 + 1 − X)(y0 + 1 − Y), nord-est (X − x0)(y0 + 1 − Y),
//   sud-ouest (x0 + 1 − X)(Y − y0), sud-est (X − x0)(Y − y0), chacun ajouté seulement si son pixel est dans
//   le plan ;
//   accumulation de source_c × e^z' × poids dans le plan c et de e^z' × poids dans le poids.
// Sortie : plan c accumulé / (poids accumulé + kGmfssNormalizeEpsilon). Un pixel que rien n'atteint vaut 0.
// L'ordre des additions est libre (atomiques) : les ports s'écartent de la référence par les arrondis de cet
// ordre seulement.
inline constexpr float kGmfssNormalizeEpsilon = 1e-7F;

// La métrique sort de tanh × 10 : e^z' reste dans [e^-10, e^10], sans débordement en FP32.
inline constexpr float kGmfssMetricLimit = 10.0F;

// Le mouvement sur une autre grille que la source (splat d'une image d'une autre taille que le niveau du
// mouvement) : le pixel source (x, y) lit le mouvement en u = (x + 1/2) / scale − 1/2 + ox,
// v = (y + 1/2) / scale − 1/2 + oy, bilinéaire borné au plan du mouvement (spec/warp.h), le flot lu
// multiplié par scale, la métrique telle quelle. scale 1 et décalages nuls sur une grille de même taille :
// le mouvement du pixel lui-même, tel quel. Le flot du niveau 1 (moitié des plans remplis à la taille de
// RIFE) pousse l'image de RIFE (scale 2, ox = gauche / 2, oy = haut / 2 de la géométrie) ou l'image de
// travail plus grande (scale 2 × travail / RIFE, mêmes décalages).
// Couverture : avec un plan de sortie de plus que la source, ce plan vaut
// poids accumulé / (poids accumulé + kGmfssNormalizeEpsilon) : 1 où une source arrive, 0 où rien n'arrive.
struct MotionGrid {
  float scale = 1.0F;
  float x = 0.0F;
  float y = 0.0F;
};

// Le splat tiré (SplatJob::pull, ref::soft_pull_at) : au lieu de pousser les pixels de la source, ce qui
// étale chaque pixel sur quatre voisins et adoucit l'image, on pousse le flot lui-même, puis chaque pixel de
// sortie va lire la source là d'où il vient. Pour la source (x, y) : (fx, fy) son flot sur la grille de la
// source (comme ci-dessus), splatté avec sa métrique à l'instant s ; en sortie (X, Y), le flot splatté
// (Fx, Fy) et la couverture c ; valeur = Catmull-Rom de la source en (X − s Fx, Y − s Fy) (spec/warp.h,
// RifeWarp::CatmullRom : position bornée au plan, sortie bornée aux quatre points centraux) ; le plan de
// couverture vaut c. La même occultation que le splat (la métrique choisit le flot du premier plan), le
// détail de la source tel quel. La composition à la taille de travail tire ses sources ainsi.

// Composition à la taille de travail (le mélange, travail plus grand que RIFE) : l'image du milieu se
// compose des sources de travail splattées par le flot du niveau 1, corrigées du résidu de la fusion :
// out = H + up(F − L), H le mélange des sources de travail splattées, L le même mélange des sources à la
// taille de RIFE, F la sortie de la fusion dans sa fenêtre, up l'agrandissement bilinéaire. Là où rien
// n'arrive, H et L sont nuls : la fusion agrandie reste seule. Pour le pixel (X, Y) de la sortie
// (ow × oh, au moins la taille w × h de RIFE) :
//   xw = (X + 1/2) · w / ow − 1/2, yw = (Y + 1/2) · h / oh − 1/2 ;
//   chaque côté k donne (R, G, B, c) de sa source de travail en (X, Y) et de sa source de RIFE en (xw, yw)
//   (bilinéaire borné au plan) ; poids w0 = (1 − t) c0, w1 = t c1 ;
//   M = (w0 P0 + w1 P1) / (w0 + w1 + kGmfssNormalizeEpsilon), sur chaque grille (H, L) ;
//   F lu en (xw, yw) dans la fenêtre, bilinéaire borné à la fenêtre ;
//   out = clamp(H + F − L, 0, 1), par canal.
// Le détail de la source de travail reste là où GMFSS le pousse, au lieu d'une image de RIFE agrandie.

// Décor fixe (gmfss_still) : la clé K, à la taille des sources a et b, reprend leurs pixels là où rien ne
// bouge entre elles. La fusion de GMFSS rend une image plus douce que les sources, et sa composition à la
// taille de travail n'en rend pas tout le détail : sans cette reprise, chaque clé était plus molle que ses
// voisines sur toute l'image, décor immobile compris, et les traits changeaient d'épaisseur d'une image à
// l'autre en lecture. Pour chaque pixel (x, y), en FP32 :
//   d = max sur le voisinage 3 × 3 (borné au plan) et les trois canaux de |a − b| ;
//   m = clamp((kGmfssStillHigh − d) / (kGmfssStillHigh − kGmfssStillLow), 0, 1) ;
//   out = clamp(K + m · ((1 − t) a + t b − K), 0, 1), par canal.
// Sur place possible : un pixel ne lit que son K.
inline constexpr float kGmfssStillLow = 3.0F / 255.0F;
inline constexpr float kGmfssStillHigh = 8.0F / 255.0F;

}  // namespace kirari::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
