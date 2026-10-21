// Tolérances : de combien un port (CUDA, GLSL, Metal) ou la référence peut s'écarter d'un
// résultat exact. Un golden compare chaque port à la référence CPU avec ces bornes, le port et
// la référence recevant les mêmes entrées. Chaque borne est vérifiée par un test de
// tests/unit/kernels/ ; celles des ports par tolerance_test.cpp, contre un port simulé qui
// contracte chaque a × b + c en FMA et range ses tenseurs en fp16.
#pragma once

namespace ajp::kernels::tolerance {

// Tenseur RGB fp16 d'un port contre la référence float, pour une valeur dans [0, 2] :
// un pas de fp16 à 1,0 (arrondi fp16 et contraction FMA du port).
inline constexpr float kRgbF16 = 1.0F / 1024.0F;

// Sortie quantifiée d'un port contre la référence, à entrée float identique : un code au plus
// (en pas de 10 bits pour P010). Les égalités à 0,5 basculent quand le port contracte en FMA.
inline constexpr int kCode = 1;

// Aller-retour YUV → RGB → YUV dans la référence : exact en 4:4:4 16 bits, plage limitée ou
// pleine ; en 4:2:0 (8, 10 et 16 bits), exact en luma, et en chroma si elle est uniforme.
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

// Warp et mélange d'un port contre la référence, à tenseurs fp16 identiques : contraction FMA
// et sortie fp16, vérifiées à t = 1/2. Aux autres instants, la marge doit aussi absorber logf
// et expf approchées : à confirmer par le golden de chaque port.
inline constexpr float kWarp = 2.0F / 1024.0F;

// Le même warp lu par le filtre bilinéaire matériel (objets texture CUDA, bilinear.cuh) : la
// fraction sur 8 bits écarte chaque prise d'au plus le gradient local / 512 (/ 256 si le matériel
// tronque). Émulé sur les images du test du backend TensorRT (flot jusqu'à 1,5 px par pixel),
// l'écart atteint 2,1/1024 en troncature ; l'instant réglé des poses reste lu exactement.
inline constexpr float kWarpFiltered = 4.0F / 1024.0F;

}  // namespace ajp::kernels::tolerance
