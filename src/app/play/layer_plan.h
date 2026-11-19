// Interpolation par calques et plaque de décor (docs/calques.md) : ce que le lecteur décide d'une paire
// d'après sa carte de la scène (kernels/spec/scene_flow.h), sans rien toucher au GPU. layer_stage.h exécute.
//
// Calques : là où la carte dit « décor qui suit la caméra », l'image intermédiaire reprend a et b déplacées
// par le seul mouvement de caméra (kernels/spec/layers.h) ; RIFE garde les dessins, les effets et tout ce
// dont la carte doute. Plaque : chaque source b reprend son décor d'une image clé passée une fois par un SPAN
// plus précis, recadrée au mouvement de caméra cumulé depuis elle.
//
// Règle absolue : sans carte, ou dès qu'elle doute de la paire, rien ne change à l'image d'aujourd'hui.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <optional>
#include <string_view>
#include <vector>

#include "kernels/spec/layers.h"
#include "kernels/spec/scene_flow.h"

namespace kirari::app::play {

// Pourquoi une paire n'a pas de calques (compté au journal de fin).
enum class LayerSkip : std::uint8_t {
  NoMap,     // pas de carte : AMD, Intel, Turing, NVOF refusé, pas encore relue
  NoCamera,  // la carte n'a pas pu ajuster de caméra
  Cut,       // coupe probable ou fondu
  Residual,  // le décor s'écarte du modèle de caméra au-delà de kLayerResidualMax (porte 2.1)
  Warp,      // mouvement qui ne s'inverse pas (zoom extrême)
  FewTiles,  // trop peu de décor sûr pour qu'il y ait quelque chose à gagner
  Grid,      // grille plus grande que kernels::kLayerMaxCols × kLayerMaxRows
};
inline constexpr std::size_t kLayerSkipCount = 7;
std::string_view name(LayerSkip skip);

// Seuils (non mesurés, réglés à la séance du groupe 4) :
// - le décor suit la caméra à moins d'un demi-pixel de travail (la porte 2.1) ;
inline constexpr double kLayerResidualMax = 0.5;
// - une tuile décor compte avec une confiance d'au moins ceci ;
inline constexpr float kLayerConfidence = 0.5F;
// - une paire n'a de calques qu'avec au moins cette part de tuiles décor, après l'érosion.
inline constexpr double kLayerMinShare = 0.05;

// Les calques d'une paire : la caméra et les tuiles décor, sur la grille de la carte (pixels de travail).
struct PairLayers {
  kernels::Affine camera = kernels::kIdentityAffine;  // a vers b, pixels de travail
  int work_width = 0;
  int work_height = 0;
  int tile_px = 0;
  int cols = 0;
  int rows = 0;
  std::vector<std::uint8_t> decor;  // cols × rows, 1 : décor sûr
  double share = 0.0;               // part des tuiles décor
};

// Les tuiles décor sûres : Background à confiance suffisante, ou Uncertain sans changement (un aplat que la
// caméra déplace) ; puis tout le chemin d'un dessin, d'un effet ou d'une tuile incertaine qui change (de sa
// tuile dans a à sa tuile dans b, élargi d'une tuile) est retiré, pour qu'aucun bout de personnage ne passe
// par la caméra.
std::expected<PairLayers, LayerSkip> plan_layers(const std::optional<kernels::SceneMotion>& scene,
                                                 int work_width, int work_height);

// La grille des tuiles décor portée sur une image de out_width × out_height qui couvre le travail.
kernels::LayerGrid layer_grid(const PairLayers& layers, int out_width, int out_height);

// Les paramètres de la composition d'une image intermédiaire à l'instant t, a, b et la sortie à
// out_width × out_height. Rien si le mouvement ne s'inverse pas.
std::optional<kernels::LayerParams> pair_params(const PairLayers& layers, int out_width, int out_height,
                                                double t, bool cubic);

// La plaque de décor d'un plan. Une image clé est passée par le SPAN de la plaque ; les sources suivantes du
// plan reprennent leur décor d'elle, au mouvement de caméra cumulé. Une nouvelle clé quand la plaque ne
// couvre plus assez l'image, que le zoom a trop changé ou qu'elle a trop vieilli ; plus de plaque dès qu'une
// paire perd la caméra, jusqu'à la paire sûre suivante.
inline constexpr double kPlateMinCover = 0.75;
inline constexpr double kPlateMaxZoom = 0.03;       // |det − 1| du mouvement cumulé
inline constexpr std::uint32_t kPlateMaxAge = 240;  // paires (10 s à 24 i/s)

enum class PlateAction : std::uint8_t {
  None,     // b garde son SPAN
  Refresh,  // b devient la clé : SPAN de la plaque sur b, puis le décor de b repris d'elle, sans décalage
  Use,      // le décor de b repris de la plaque, déplacé de key_to_b
};

struct PlateStep {
  PlateAction action = PlateAction::None;
  kernels::Affine key_to_b = kernels::kIdentityAffine;  // pixels de travail
};

class PlateTrack {
 public:
  // La paire (a, b) d'un même plan, avec ses calques ou la raison de leur absence ; continues faux après un
  // saut, une nouvelle époque ou une coupe : tout repart.
  PlateStep step(const std::expected<PairLayers, LayerSkip>& pair, bool continues);
  // La clé et ses tuiles décor (la grille de la plaque, sur sa propre image).
  const PairLayers& key() const { return key_; }
  void forget() {
    has_key_ = false;
    age_ = 0;
  }

 private:
  bool has_key_ = false;
  kernels::Affine key_to_b_ = kernels::kIdentityAffine;
  std::uint32_t age_ = 0;
  PairLayers key_;
};

// La part de l'image b (w × h) que l'image clé couvre, mesurée sur une grille de 16 × 9 points.
double plate_cover(const kernels::Affine& key_to_b, int width, int height);

// Les paramètres de la composition de la plaque dans SPAN de b, tous deux à out_width × out_height ; b_layers
// les tuiles décor de b (celles de la paire qui l'amène). Rien si le mouvement ne s'inverse pas.
std::optional<kernels::LayerParams> plate_params(const PairLayers& key, const PairLayers& b_layers,
                                                 const kernels::Affine& key_to_b, int out_width,
                                                 int out_height, bool cubic);

}  // namespace kirari::app::play
