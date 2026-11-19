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
// suit la caméra et ne change guère est décor : rien n'y bouge de visible. Caméra affinée, c'est l'écart
// après la caméra mesuré sur l'image qui sépare le décor du dessin et de l'effet (kMovingShare). Une lumière
// qui change sur place est un effet (kLightAgree).
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
  // un quart de vecteurs sans correspondance, un écart moyen d'au moins kChangeDiff, ou (caméra affinée) des
  // pixels qui bougent après la caméra (kMovingShare). Brut, non lissé.
  bool changed = false;
};

// La carte d'une paire (a, b) d'images de travail.
struct SceneMotion {
  // Mouvement de caméra, affine, en pixels de travail : le point (x, y) de a est en
  // (c0 + c1 x + c2 y, c3 + c4 x + c5 y) dans b. Identité si camera_fit est faux.
  std::array<double, 6> camera{0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  bool camera_fit = false;
  bool camera_refined = false;  // affinée par Lucas-Kanade sur les lumas (spec/scene_camera.h)
  // Écart du décor au modèle, pixels de travail. Mesuré sur l'image (médiane des décalages qui restent sur
  // residual_tiles tuiles du décor, spec/scene_camera.h) quand la caméra est affinée et que l'image le permet
  // ; sinon (residual_tiles 0), écart moyen des vecteurs de NVOF des tuiles gardées par l'ajustement.
  double residual = 0.0;
  int residual_tiles = 0;
  int tile_px = 0;  // côté d'une tuile, pixels de travail
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
// L'ajustement du banc 2.1 (camera() de tools/bench/qualite/nvof_mesures.py), en pixels à 720p (× hauteur du
// flux / 720) : départ sur la translation médiane, kFitRounds tours, et à chaque tour, rejet des tuiles à
// plus de kFitMad fois l'écart médian de celles à moins de kFitNear, seuil borné entre kFitBoundLow et
// kFitBoundHigh. Poids d'une tuile : 1 / (1 + coût moyen / coût moyen médian), × sa texture.
inline constexpr int kFitRounds = 8;
inline constexpr float kFitMad = 3.0F * 1.4826F;
inline constexpr float kFitNear = 2.0F;
inline constexpr float kFitBoundLow = 0.75F;
inline constexpr float kFitBoundHigh = 2.0F;
inline constexpr float kCutUnmatched = 0.6F;       // part de vecteurs sans correspondance d'une coupe
inline constexpr float kFadeShift = 4.0F;          // glissement moyen de luminosité d'un fondu, codes
inline constexpr float kFadeAgree = 0.8F;          // part des tuiles qui glissent dans le même sens
inline constexpr float kDrawingHeldShare = 0.01F;  // part de tuiles qui bougent au plus d'un dessin tenu
inline constexpr double kHeldCamera = 0.25;        // translation nulle, pixels de travail
inline constexpr double kHeldLinear = 0.002;       // partie linéaire égale à l'identité
inline constexpr float kContradicted = 0.4F;  // facteur de confiance d'une classe contredite (jamais sûre)
// Un effet ne se voit qu'en changeant : sur une tuile qui ne change pas (ref::SceneHistory), il tient au plus
// kEffectIdlePairs paires, les images doublées d'une animation en trois, puis la tuile reprend sa classe
// brute. Tenu sans limite, le gros plan du 2:28 gardait ses faux effets 70 paires sur une image fixe.
inline constexpr int kEffectIdlePairs = 2;
// La séparation par l'image. Deuxième passage de la porte 2.3 (19 novembre) : caméra ajustée sur 95 à 100 %
// des paires, mais dessin rare (0 à 2 % des tuiles sur 5 extraits sur 8) et effet quasi absent ; sur les
// dialogues, les personnages finissaient en décor. Une bouche ou des yeux qui changent ne déplacent pas le
// vecteur moyen de NVOF, qui reste celui de la caméra, et une part sans correspondance n'existe guère (coût
// au-dessus de kSceneCostHigh sur 0,5 % des vecteurs). D'où l'image : quand la caméra est affinée, chaque
// tuile a son écart après la caméra (spec/scene_camera.h). Elle bouge hors de la caméra si au moins
// kMovingShare de ses pixels bougent, ou si Lucas-Kanade la recale d'au moins kShiftMoving pixels à 720p dans
// un sens résolu : effet si ses vecteurs coûtent (coût moyen d'au moins kEffectCost), qu'au moins
// kEffectShare de ses pixels bougent et qu'aucun bord net n'y bouge (ni elle ni ses voisines, kLightSharp),
// dessin sinon. Elle ne bouge pas : décor, aplat compris (rien n'y bouge de visible), quoi que disent ses
// vecteurs. Sans le bord net, un gros plan animé en trois (Re:Zero 18, 2:28, cheveux et visage d'Emilia)
// virait à l'effet sur 230 à 430 tuiles par dessin : un trait qui bouge coûte autant à NVOF qu'une lueur.
// Une tuile sans correspondance qui change (kEffectDiff) est un effet au même titre, un dessin si un bord
// net y bouge (il découvre ou recouvre le décor).
inline constexpr float kMovingShare = 0.1F;
inline constexpr float kShiftMoving = 1.0F;  // pixels à 720p
inline constexpr float kEffectShare = 0.3F;
inline constexpr float kEffectCost =
    12.0F;  // sur 255 ; 27 % des vecteurs coûtent plus de 8, 2,2 % plus de 16
// Les effets lumineux. Instantanés de Re:Zero (16:20 et 16:43, une lueur et une aura) : aucune tuile en
// effet, et les éclats HDR en dépendent (app/play/effect_book.cpp). Une lueur coûte peu à NVOF (rien de net à
// apparier) et, sur place, ne déplace pas son vecteur. Ce qui la distingue :
// - d'un dessin qui se déplace : ses pixels qui changent vont presque tous dans le même sens de luminosité
//   (l'accord |éclaircis − assombris| / qui bougent atteint kLightAgree), quand un dessin assombrit d'un côté
//   ce qu'il éclaircit de l'autre ;
// - d'un dégradé qui glisse (d'un seul sens aussi sur une tuile) : aucun recalage ne l'explique, il reste au
//   moins kLightKept de l'écart après celui de Lucas-Kanade (SceneTileCheck::settled) ;
// - d'une bouche qui s'ouvre, d'un œil qui cligne, d'un objet qui apparaît (sur place et d'un seul
//   sens) : elle change en douceur ; ni sa tuile ni ses huit voisines n'ont plus de kLightSharp de leurs
//   pixels qui bougent sur un bord net (scene::kSharpEdge, spec/scene_camera.h ; le cœur d'une grande
//   bouche change sans bord, les tuiles de son bord le trahissent) ;
// - d'un objet clair qui entre dans une tuile sombre : il entraîne ses vecteurs, ceux d'une lumière
//   suivent la caméra.
// Une tuile est donc une lumière (effet, de confiance l'accord) si au moins kMovingShare de ses pixels
// bougent après la caméra et que tout cela tient. Il y faut l'écart après la caméra (une demande d'affinage :
// presque toutes les paires où la caméra est ajustée) ; sans lui, rien ne sépare une lueur d'une bouche.
// Jamais sur un fondu (le glissement y est partout). Réglés sans mesure, à revoir sur des extraits à effets
// (carte_lecteur.py).
inline constexpr float kLightAgree = 0.6F;  // 80 % des pixels qui bougent dans le même sens
// Un dégradé qui glisse de 4 pixels du flux garde la moitié de son écart (le premier pas du recalage va à
// kCheckClip) ; une lueur sur une texture forte (cas du test), 57 à 83 %, sur un fond sombre, presque tout.
inline constexpr float kLightKept = 0.5F;
inline constexpr float kLightSharp = 0.25F;
}  // namespace scene

}  // namespace kirari::kernels

#include "kernels/spec/fp_off_end.h"  // IWYU pragma: keep
