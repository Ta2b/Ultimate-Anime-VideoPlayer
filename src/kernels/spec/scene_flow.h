// Contrat de la carte de la scène (étape 2.3 du plan du flux NVIDIA) : le flux optique matériel de NVIDIA
// (NVOF, backend TensorRT) entre deux images de travail, réduit par tuile sur le GPU, puis lu sur l'hôte en
// une carte de décisions (SceneMotion). Le flux de NVOF est imprécis sur les mouvements rapides de l'anime
// (11 à 45 pixels d'écart au-delà de 45 pixels de mouvement, mesure du 17 novembre) : la carte ne dit que
// « ça bouge ou pas », « ça correspond ou pas » et le mouvement d'ensemble, jamais un vecteur par pixel.
//
// Image du flux. L'image de travail (RGB de la chaîne, 1 × 3 × H × W) est réduite d'un diviseur entier
// d = scene_flow_divisor(H), pour que le flux ne dépasse pas kSceneFlowMaxHeight lignes (en 1080p, NVOF fait
// pire qu'en 720p : sa portée de recherche est dépassée). Chaque pixel du flux, (x, y), est la moyenne de la
// boîte d × d de travail [d x, d x + d) × [d y, d y + d) de la luma Y = kr R + (1 − kr − kb) G + kb B
// (BT.709), puis code = arrondi(255 × borne(Y, 0, 1)). Le RGB garde l'échelle de la plage de la source :
// une source limitée donne des codes de 16 à 235, sans conséquence pour le flux. Largeur et hauteur du flux :
// W / d et H / d, tronquées ; la carte refuse sous kSceneFlowMinSide.
//
// Flux. NVOF, grille kSceneFlowGrid : un vecteur (S10.5, pixels du flux) par bloc 4 × 4, de a vers b
// (a(p) ≈ b(p + v)), et un coût 8 bits par vecteur (plus haut, moins sûr ; Ampere et après).
//
// Tuiles. kSceneTileCells × kSceneTileCells vecteurs, soit 16 × 16 pixels du flux et 16 d × 16 d pixels de
// travail (32 en 1080p). Pour chaque tuile, sur ses vecteurs dans l'image (cells, sum_x, sum_y, sum_xx,
// sum_yy, cost, unmatched : coût ≥ kSceneCostHigh) et sur ses pixels du flux dans l'image (pixels ; texture :
// Σ |a(x+1, y) − a(x, y)| + |a(x, y+1) − a(x, y)|, chaque terme compté seulement si le voisin est dans
// l'image ; diff : Σ |b − a| ; signed_diff : Σ (b − a)). Tout est sommé en entiers (vecteurs en S10.5,
// codes), puis rendu en float : sum_x et sum_y ÷ 32, sum_xx et sum_yy ÷ 1024 ; le résultat ne dépend donc pas
// de l'ordre de la réduction. Référence :
// ref::scene_tile_sums ; port : gpu/cuda/backend/scene_flow_kernels.cu.
//
// Carte (ref::scene_motion_of, code du lecteur) : voir SceneMotion plus bas et kernels/ref/scene_flow.cpp.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "kernels/spec/fp_off_begin.h"  // IWYU pragma: keep

namespace kirari::kernels {

inline constexpr int kSceneFlowMaxHeight = 720;
inline constexpr int kSceneFlowMinSide = 32;
inline constexpr int kSceneFlowGrid = 4;
inline constexpr int kSceneTileCells = 4;
inline constexpr int kSceneTileFlowPixels = kSceneFlowGrid * kSceneTileCells;
// Coût d'un vecteur à partir duquel il est « sans correspondance ». Sur 255, mais NVOF n'en donne presque
// jamais plus de 32 (banc de la porte 2.3, flux 960 × 540, réglage moyen : 99e centile 20, 99,9e 30, 0,07 %
// au-dessus de 32) : à 160, aucun vecteur ne l'était, ni effet ni coupe ne se voyaient. Entre les deux.
inline constexpr std::uint8_t kSceneCostHigh = 24;

constexpr int scene_flow_divisor(int work_height) {
  return std::max(1, (work_height + kSceneFlowMaxHeight - 1) / kSceneFlowMaxHeight);
}
constexpr int scene_flow_cells(int flow_side) {
  return (flow_side + kSceneFlowGrid - 1) / kSceneFlowGrid;
}
constexpr int scene_flow_tiles(int flow_side) {
  return (scene_flow_cells(flow_side) + kSceneTileCells - 1) / kSceneTileCells;
}

// Les sommes d'une tuile, écrites par le GPU dans cet ordre (11 float, sans bourrage).
struct SceneTileSums {
  float cells = 0.0F;
  float sum_x = 0.0F;  // pixels du flux
  float sum_y = 0.0F;
  float sum_xx = 0.0F;
  float sum_yy = 0.0F;
  float cost = 0.0F;
  float unmatched = 0.0F;
  float pixels = 0.0F;
  float texture = 0.0F;  // codes
  float diff = 0.0F;
  float signed_diff = 0.0F;
};
static_assert(sizeof(SceneTileSums) == 11 * sizeof(float));

// Ce que le backend relit d'une paire : la grille de tuiles, ligne par ligne.
struct SceneFlowTiles {
  int divisor = 1;     // travail → flux
  int flow_width = 0;  // image du flux
  int flow_height = 0;
  int cols = 0;
  int rows = 0;
  std::vector<SceneTileSums> tiles;  // cols × rows
};

// Classe d'une tuile. Uncertain : le flux ne permet pas de trancher (aplat qui change ou dont le vecteur
// contredit la caméra, vecteurs sans correspondance sans effet, mouvement sans caméra ajustée). Un aplat qui
// suit la caméra et ne change guère est décor : rien n'y bouge de visible.
enum class SceneClass : std::uint8_t { Uncertain, Background, Drawing, Effect };

struct SceneTile {
  SceneClass kind = SceneClass::Uncertain;
  float confidence = 0.0F;  // [0, 1]
  float dx = 0.0F;          // déplacement moyen de la tuile, de a vers b, pixels de travail
  float dy = 0.0F;
  float change = 0.0F;  // écart moyen |b − a| de la luma du flux, codes 8 bits (grain compris)
  // Glissement moyen (b − a) de la luma du flux, codes 8 bits, brut : le grain s'y annule, un fondu ou une
  // lueur locale s'y voit (les tuiles qui bougent l'additionnent, app/play/upscale_tiles.h).
  float shift = 0.0F;
  // La tuile a changé, sans compter le grain : vecteur moyen d'au moins kStillMotion, vecteurs épars,
  // un quart de vecteurs sans correspondance, ou un écart moyen d'au moins kChangeDiff. Brut, non lissé.
  bool changed = false;
};

// La carte d'une paire (a, b) d'images de travail.
struct SceneMotion {
  // Mouvement de caméra, affine, en pixels de travail : le point (x, y) de a est en
  // (c0 + c1 x + c2 y, c3 + c4 x + c5 y) dans b. Identité si camera_fit est faux.
  std::array<double, 6> camera{0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  bool camera_fit = false;
  double residual = 0.0;  // écart moyen des tuiles décor au modèle, pixels de travail
  int tile_px = 0;        // côté d'une tuile, pixels de travail
  int cols = 0;
  int rows = 0;
  std::vector<SceneTile> tiles;  // cols × rows, lissées dans le temps (ref::SceneHistory)
  double unmatched = 0.0;        // part des vecteurs sans correspondance
  // Rien ne bouge hors de la caméra : au plus kDrawingHeldShare des tuiles en dessin ou en effet (classes
  // brutes) ; sans caméra ajustée, au plus cette part de tuiles changées. Faux sur une coupe ou un fondu.
  bool drawing_held = false;
  bool background_held = false;  // la caméra ne bouge pas
  bool cut_likely = false;       // la plupart des vecteurs sans correspondance, hors fondu
  bool fade = false;             // luminosité qui glisse partout dans le même sens (fondu, flash)

  const SceneTile& at(int col, int row) const {
    return tiles[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                 static_cast<std::size_t>(col)];
  }
};

// Seuils de la carte (ref::scene_motion_of), réglés par la porte 2.3 (classe juste sur 95 % des tuiles sûres
// des extraits du banc 2.1, tools/bench/qualite/carte_lecteur.py). Premier passage (19 novembre) : aucune
// caméra sur 4 panoramiques sur 6, faute de tuiles candidates. Il fallait 6 codes par pixel (à peu près un
// trait net entier par tuile) sur 15 % de l'image (306 tuiles en 960 × 540) ; ces scènes sombres de Re:Zero
// n'en ont que 2 à 355 au-dessus de 3 codes, et 32 à 461 au-dessus de 2 (luma moyenne 31 à 49). Le banc 2.1
// y ajuste pourtant sa caméra (résidu 0,13 à 0,36 px) sur tout bloc où la luma varie d'un code.
// D'où l'ajustement dès 1,5 code sur 3 % de l'image (62 tuiles), l'aplat sous 3.
namespace scene {
inline constexpr float kTextureFloor = 3.0F;  // texture moyenne par pixel, codes : sous elle, aplat
inline constexpr float kFitTexture = 1.5F;    // et texture d'une tuile qui peut ajuster la caméra
inline constexpr float kMatchedFloor = 0.5F;  // part de vecteurs sûrs d'une tuile qui correspond
inline constexpr float kFitMatched = 0.75F;   // et d'une tuile qui ajuste la caméra, d'un aplat décor
inline constexpr float kSpreadMax = 1.5F;     // écart type des vecteurs d'une tuile cohérente, pixels du flux
inline constexpr float kStillDiff = 3.0F;     // écart moyen |b − a| d'une tuile qui ne change pas, codes
inline constexpr float kChangeDiff = 6.0F;    // écart moyen d'une tuile qui change sans flux net
inline constexpr float kChangeUnmatched = 0.25F;  // part sans correspondance d'une tuile qui change
inline constexpr float kEffectDiff = 12.0F;       // et d'une tuile sans correspondance qui change (effet)
inline constexpr float kBackgroundFloor = 1.5F;   // écart au modèle d'une tuile décor, pixels du flux
inline constexpr float kBackgroundShare = 0.1F;   // plus cette part du mouvement de caméra
inline constexpr float kStillMotion = 0.5F;       // mouvement nul, pixels du flux (sans caméra ajustée)
inline constexpr int kFitMinTiles = 16;           // tuiles retenues au moins pour ajuster la caméra
inline constexpr float kFitMinShare = 0.03F;      // et cette part des tuiles de l'image
inline constexpr float kFitInlierScale = 2.5F;    // rejet : au-delà de 2,5 × l'écart médian
inline constexpr int kFitRounds = 4;
inline constexpr float kCutUnmatched = 0.6F;       // part de vecteurs sans correspondance d'une coupe
inline constexpr float kFadeShift = 4.0F;          // glissement moyen de luminosité d'un fondu, codes
inline constexpr float kFadeAgree = 0.8F;          // part des tuiles qui glissent dans le même sens
inline constexpr float kDrawingHeldShare = 0.01F;  // part de tuiles qui bougent au plus d'un dessin tenu
inline constexpr double kHeldCamera = 0.25;        // translation nulle, pixels de travail
inline constexpr double kHeldLinear = 0.002;       // partie linéaire égale à l'identité
inline constexpr float kContradicted = 0.4F;  // facteur de confiance d'une classe contredite (jamais sûre)
}  // namespace scene

}  // namespace kirari::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
