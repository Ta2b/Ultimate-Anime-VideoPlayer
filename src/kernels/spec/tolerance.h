// Tolérances : de combien un port (CUDA, GLSL, Metal) ou la référence peut s'écarter d'un
// résultat exact. Un golden compare chaque port à la référence CPU avec ces bornes.
// Chaque borne est vérifiée par un test de tests/unit/kernels/.
#pragma once

namespace ajp::kernels::tolerance {

// Tenseur RGB fp16 d'un port contre la référence float, pour une valeur dans [0, 2] :
// un pas de fp16 à 1,0 (arrondi fp16 et contraction FMA du port).
inline constexpr float kRgbF16 = 1.0F / 1024.0F;

// Sortie quantifiée d'un port contre la référence : un code au plus. Les égalités à 0,5
// peuvent basculer quand le port contracte a × b + c en FMA.
inline constexpr int kCode = 1;

// Aller-retour 4:4:4 YUV → RGB → YUV dans la référence : exact, en 8, 10 et 16 bits, plage
// limitée ou pleine. La luma 4:2:0 aussi ; la chroma 4:2:0 seulement si elle est uniforme.
inline constexpr int kRoundTrip = 0;

// Somme des poids d'une table de rééchantillonnage rangée en float.
inline constexpr float kWeightSum = 1e-6F;

// Image uniforme rééchantillonnée : écart relatif à la valeur d'origine.
inline constexpr float kConstantImage = 2e-6F;

// Fonctions de transfert float contre la formule en double : |f − d| ≤ rel |d| + abs × crête.
// L'EOTF PQ fixe la borne relative : près de 10 000 cd/m², le dénominateur c2 − c3 p perd
// sept bits par annulation, puis l'exposant 1/m1 multiplie l'erreur par 6,3. On mesure
// 5,8e-5 en float à e = 0,9988 ; l'inverse et les autres courbes restent sous 2e-5.
inline constexpr double kTransferRelative = 1e-4;
inline constexpr double kTransferAbsolute = 1e-6;

// Matrices de primaires rangées en float contre le double.
inline constexpr double kMatrixFloat = 1e-7;

// Score de coupe de scène : l'ordre de la réduction est libre sur GPU.
inline constexpr double kSceneScoreRelative = 1e-4;

// Warp et mélange d'un port contre la référence (lectures fp16, logf/expf approchées).
inline constexpr float kWarp = 2.0F / 1024.0F;

}  // namespace ajp::kernels::tolerance
