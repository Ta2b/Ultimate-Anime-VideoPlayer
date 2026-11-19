// Le rythme d'un plan, lu en avance par la pré-analyse (media/shot_probe.h) : combien de ses images
// apportent un dessin nouveau (anime en un, en deux, en trois), quelle part de l'image bouge, la caméra, le
// grain des zones immobiles. L'Auto plan par plan (app/play/shot_plan.h) en tire le moteur d'interpolation
// et la stabilisation temporelle de chaque plan. Pur calcul, sans FFmpeg ni thread.
//
// Deux sources pour une paire d'images (la précédente, celle-ci) :
// - la luma de la grille de l'analyse (kernels/spec/frame_stats.h, 128 × 72 points), toujours là : par blocs
//   de 8 × 8 points, la part des blocs qui changent et le grain des autres ;
// - le flux matériel de NVIDIA à 256 × 144 (media/probe_flow.h, quand la carte le fait) : le mouvement de la
//   caméra (translation dominante) et celui des dessins par rapport à elle. Il sépare un panoramique d'un
//   passage de sakuga, ce que la luma seule ne sait pas.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace kirari::media {

// Seuils du rythme. Non mesurés : la séance du PC (docs/auto-par-plan.md) les règle.
namespace rhythm {
inline constexpr int kBlockPoints = 8;        // côté d'un bloc, en points de la grille
inline constexpr float kMoveCodes = 6.0F;     // écart moyen |b − a| d'un bloc qui change, codes 8 bits
inline constexpr float kHeldChanged = 0.02F;  // part de blocs qui changent d'une paire sans dessin nouveau
// Avec le flux : mouvement des dessins par rapport à la caméra, et de la caméra, en part de la hauteur de
// l'image par image (0,25 pixel à 144 lignes).
inline constexpr float kHeldDrawing = 0.25F / 144.0F;
inline constexpr float kHeldCamera = 0.25F / 144.0F;
// Une paire forte : une grande part de l'image change, ou les dessins vont vite (sakuga) ; avec le flux,
// aussi une part de blocs sans correspondance (ce que la recherche par blocs ne suit plus).
inline constexpr float kStrongChanged = 0.35F;
inline constexpr float kStrongDrawing = 6.0F / 144.0F;
inline constexpr float kStrongUnmatched = 0.15F;
}  // namespace rhythm

// Une paire d'images consécutives d'un plan.
struct RhythmPair {
  float changed = 0.0F;  // part des blocs dont l'écart moyen atteint kMoveCodes ; les autres sont immobiles
  float grain = 0.0F;    // écart moyen |b − a| des blocs immobiles, codes 8 bits (grain, compression)
  // Le flux, quand il a été mesuré (flow) : en part de la hauteur de l'image, par image.
  bool flow = false;
  float camera = 0.0F;     // translation dominante
  float drawing = 0.0F;    // écart moyen au mouvement de la caméra (les dessins)
  float unmatched = 0.0F;  // part des blocs texturés sans correspondance
  friend bool operator==(const RhythmPair&, const RhythmPair&) = default;
};

// Le flux d'une paire tel que la pré-analyse le lit (les champs de media/probe_flow.h, ProbeMotion), en
// pixels de la luma de height lignes.
struct FlowSummary {
  float shift_x = 0.0F;
  float shift_y = 0.0F;
  float residual = 0.0F;
  float unmatched = 0.0F;
};
// La paire avec son flux : la caméra et les dessins ramenés à la hauteur de l'image.
RhythmPair with_flow(RhythmPair pair, const FlowSummary& flow, int height);

// La paire (previous, luma) mesurée sur la grille : deux images de width × height lumas dans [0, 1].
// Rien si les tailles diffèrent ou sont vides.
std::optional<RhythmPair> rhythm_pair_of(std::span<const float> previous, std::span<const float> luma,
                                         int width, int height);

// Le rythme d'un plan : des moyennes de ses paires, la première exclue (elle enjambe la coupe).
struct ShotRhythm {
  std::uint32_t pairs = 0;
  std::uint32_t held = 0;        // paires sans dessin nouveau
  std::uint32_t strong = 0;      // paires fortes (sakuga, grand mouvement)
  std::uint32_t flow_pairs = 0;  // paires mesurées par le flux
  float changed = 0.0F;          // part moyenne des blocs qui changent, paires à dessin nouveau
  float grain = 0.0F;            // grain moyen des blocs immobiles, codes 8 bits
  float still = 0.0F;            // part moyenne des blocs immobiles, toutes paires
  float camera = 0.0F;           // moyennes du flux (flow_pairs), part de la hauteur par image
  float drawing = 0.0F;
  float unmatched = 0.0F;
  friend bool operator==(const ShotRhythm&, const ShotRhythm&) = default;
};

// Une paire sans dessin nouveau : avec le flux, ni les dessins ni la caméra ne bougent ; sans lui, presque
// aucun bloc ne change.
bool held_pair(const RhythmPair& pair);
bool strong_pair(const RhythmPair& pair);

// Images par dessin : 1 (anime en un), 2, 3… ; 0 sans paire. Moyenne sur le plan : pairs / (pairs − held).
float drawing_cadence(const ShotRhythm& rhythm);

// Les sommes d'un plan en cours (media/shot_cutter.h).
class RhythmTotals {
 public:
  void add(const RhythmPair& pair);
  ShotRhythm rhythm() const;
  std::uint32_t pairs() const { return pairs_; }

 private:
  std::uint32_t pairs_ = 0, held_ = 0, strong_ = 0, flow_ = 0;
  double changed_ = 0.0, grain_ = 0.0, still_ = 0.0, camera_ = 0.0, drawing_ = 0.0, unmatched_ = 0.0;
};

// La luma de la grille d'une image, pour la paire suivante (media/shot_measure.cpp). Garde la précédente :
// push rend la paire (précédente, luma), rien pour la première image ou après reset (un saut).
class RhythmProbe {
 public:
  std::optional<RhythmPair> push(std::vector<float> luma, int width, int height);
  void reset() { previous_.clear(); }

 private:
  std::vector<float> previous_;
  int width_ = 0;
  int height_ = 0;
};

}  // namespace kirari::media
