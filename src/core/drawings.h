// Timeline des dessins. L'anime tient un dessin sur deux, trois ou quatre
// sources : interpoler de source à source déplace l'image en une seule période,
// puis la fige pendant les autres, et le mouvement avance par à-coups.
//
// La timeline classe chaque source (nouveau dessin ou dessin tenu), reconnaît le
// rythme du plan (en uns, en deux, en trois) et dit quoi montrer à chaque instant
// de sortie :
//   - Source et KeepPoses gardent la grille source. Entre deux dessins tenus,
//     Fast et Refine reprennent l'image (REPEATED) ; Quality ne reprend qu'un
//     doublon exact. KeepPoses ajuste ensuite t par pixel (DRBA), dans gpu ;
//   - Smooth suit la timeline des dessins : le mouvement d'un dessin au suivant
//     se répartit sur toute sa tenue, jusqu'à kMaxSpreadHold sources. En Smooth,
//     seule la première source d'un dessin sert d'extrémité.
// Une coupe de scène n'est jamais interpolée : l'image de gauche reste.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>

#include "core/profile.h"
#include "core/types.h"

namespace ajp::core {

namespace drawings {
// Un dessin est tenu quand son pire bloc 32x32 diffère d'au plus 0,70 niveau 8 bits.
// 0,40 tombe dans le bruit de compression : la bande 0,40-0,70 n'en contient que.
inline constexpr double kHeldThreshold = 0.70;
inline constexpr double kMaxDiff = 255.0;
// Smooth répartit le mouvement sur une tenue de 4 sources au plus ; au-delà, le plan est fixe.
inline constexpr std::uint64_t kMaxSpreadHold = 4;
// Préchargement du rythme : sources classées au-delà de la source droite de la paire.
inline constexpr int kSmoothPreload = 4;
inline constexpr int kKeepPosesPreload = 1;
// Le rythme d'un plan est le mode de ses dernières tenues achevées.
inline constexpr std::size_t kRhythmWindow = 8;
// Sources gardées : la paire à décider et son préchargement doivent y tenir.
inline constexpr std::size_t kWindow = 32;
}  // namespace drawings

constexpr int preload(Pacing pacing) {
  switch (pacing) {
    case Pacing::Source:
      return 0;
    case Pacing::Smooth:
      return drawings::kSmoothPreload;
    case Pacing::KeepPoses:
      return drawings::kKeepPosesPreload;
  }
  return 0;
}

// Mesure d'une source contre la précédente, faite par gpu sur la luma.
struct PairMeasure {
  double worst_block_diff = 0.0;  // différence absolue moyenne du pire bloc 32x32, niveaux 8 bits
  bool scene_cut = false;
  bool identical = false;  // égales au bit : exige une différence nulle et aucune coupe
};

enum class DrawingClass : std::uint8_t {
  New,   // nouveau dessin : première source, coupe, ou pire bloc au-dessus du seuil
  Held,  // dessin tenu : le même dessin que la source précédente
};

struct SourceDrawing {
  DrawingClass kind = DrawingClass::New;
  bool scene_cut = false;  // New : une coupe la sépare de la précédente, un plan s'ouvre
  bool identical = false;  // Held : doublon exact de la source précédente
  SourceId first;          // première source du dessin
};

enum class Rhythm : std::uint8_t {
  Unknown,  // aucun dessin achevé dans le plan
  Ones,     // en uns : chaque source porte un dessin
  Twos,     // en deux : un dessin tient deux sources
  Threes,   // en trois
  Longer,   // quatre sources ou plus
};

enum class Show : std::uint8_t {
  Source,       // la source a elle-même
  Interpolate,  // entre a et b, à la position u dans ]0, 1[
  Repeated,     // l'image de a, déjà calculée : rien à calculer (REPEATED)
};

struct InstantDecision {
  Show show = Show::Source;
  SourceId a;
  SourceId b;     // Interpolate seulement
  Ratio u{0, 1};  // Interpolate seulement, réduit
  friend bool operator==(const InstantDecision&, const InstantDecision&) = default;
};

class DrawingsTimeline {
 public:
  // Première source d'une séquence : démarrage, Flush, ou grille recommencée.
  void start(SourceId first);
  // Source suivante, mesurée contre la précédente.
  std::expected<SourceDrawing, Refusal> push(SourceId id, const PairMeasure& measure);
  // Fin du flux : l'anticipation qui manque vaut « aucun dessin suivant ».
  void finish() { finished_ = true; }

  // Rythme du plan de la dernière source classée.
  Rhythm rhythm() const;
  // Les sources dont a besoin la paire (left, suivante) sont-elles classées ?
  bool ready(SourceId left, Pacing pacing) const;
  // Quoi montrer à l'instant t (0 < t <= 1) de la paire (left, suivante).
  std::expected<InstantDecision, Refusal> decide(Policy policy, Pacing pacing, SourceId left, Ratio t) const;

 private:
  struct Entry {
    SourceId id;
    SourceDrawing drawing;
    std::uint64_t first_at = 0;  // position de la première source du dessin
  };

  // La source à cette position ; elle doit être dans la fenêtre (find(), ou plus récente).
  const Entry& at(std::uint64_t position) const;
  std::optional<std::uint64_t> find(SourceId id) const;
  void store(const Entry& entry);
  void record_hold(std::uint64_t hold);
  InstantDecision on_grid(Policy policy, const Entry& left, const Entry& right, Ratio t) const;
  InstantDecision on_drawings(std::uint64_t left, Ratio t) const;

  std::array<Entry, drawings::kWindow> ring_{};
  std::uint64_t count_ = 0;  // sources classées depuis start()
  bool started_ = false;
  bool finished_ = false;
  std::array<std::uint64_t, drawings::kRhythmWindow> holds_{};  // tenues achevées du plan
  std::size_t hold_count_ = 0;
};

}  // namespace ajp::core
