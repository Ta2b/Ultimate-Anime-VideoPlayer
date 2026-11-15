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

namespace ajp::kernels {

// Les images entrent comme celles de RIFE : RGB [0, 1] de la conversion de la chaîne, remplies au multiple
// de 64 et centrées (rife_geometry), bords noirs ; la sortie se recadre pareil. vsgmfss étire l'image au
// multiple de 64 au lieu de la remplir : la fenêtre garde ici ses pixels exacts.
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

}  // namespace ajp::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
