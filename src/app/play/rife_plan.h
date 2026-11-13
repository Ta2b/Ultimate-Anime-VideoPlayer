// Plan d'interpolation d'une paire de sources (a, b) dans l'étage gpu (upscale_stage.h) : si RIFE (ou un
// autre moteur d'interpolation) y tourne. Les instants de sortie viennent de core/cadence.h.
//
// RIFE ne tourne pas sur une paire qui ne bouge pas ou qui coupe :
//   - coupe de scène (kernels/spec/detect.h, score > 0,120) : mélanger deux plans ferait un
//     fantôme ;
//   - dessin tenu (pire bloc 16 × 16 sous 0,70 niveau, kHeldBlock) : interpoler une image avec elle-même
//     redonne la même image, au prix de RIFE ;
//   - écart de pts nul, négatif ou plus long que kMaxPairGapUs (trou, reprise) ;
//   - pas de source a dans la même époque (début, seek).
// b sort alors seul, et l'écran garde a jusqu'à son pts. En animation sur deux ou trois images,
// la plupart des paires sont tenues : RIFE n'y coûte rien.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "core/cadence.h"
#include "gpu/kernel_ops.h"

namespace ajp::app::play {

// Le facteur que tient une paire de l'accumulateur de cadence (et le ring, RingStage::kMaxOutputs).
inline constexpr auto kMaxRifeFactor = static_cast<std::uint32_t>(core::kMaxInstantsPerPair);
inline constexpr std::int64_t kMaxPairGapUs = 250'000;

struct PairFacts {
  std::uint32_t factor = 1;               // ×N demandé, arrondi au-dessus ; 1 : pas d'interpolation
  bool same_epoch = false;                // a existe et appartient à l'époque de b
  std::int64_t gap_us = 0;                // pts(b) − pts(a)
  std::optional<gpu::PairScores> scores;  // mesures de la paire ; absentes : pas mesurée
  // La coupe selon les plans de la pré-analyse (cut_book.h, --shot-cuts) ; rien : la mesure décide.
  std::optional<bool> known_cut;
  // P3.32 : la part de l'image hors bandes noires (analyse du grade) ; le score de RIFE, mesuré sur toute
  // l'image, y est rapporté. 1 : pas de bandes connues.
  double useful_area = 1.0;
  // Une coupe mesurée par RIFE moins de kCutBurstUs avant la paire (CutBurst).
  bool cut_burst = false;
};

// Coupes de RIFE en rafale. Une vraie coupe que la pré-analyse manque est une paire isolée ; un mouvement
// rapide ou un effet (fumée, éclairs, panoramique) fait passer le seuil à RIFE sur presque chaque paire
// qui bouge, pendant des secondes, dans un plan que la pré-analyse dit continu. Chaque fausse coupe laisse
// b seul et fait repartir N de l'a priori : la cadence tombe vers celle de la source. Quand les plans disent
// « même plan », seule la première coupe mesurée d'une rafale compte (la garde des coupes manquées) ; les
// suivantes, à moins de kCutBurstUs de la précédente, sont du mouvement. Les dessins tenus entre deux
// paires qui bougent ne rompent pas la rafale : seule l'heure de la dernière coupe mesurée compte.
inline constexpr std::int64_t kCutBurstUs = 500'000;

class CutBurst {
 public:
  // Vrai si une coupe mesurée a commencé avant left_us, à kCutBurstUs au plus. Strictement avant : relire
  // la même paire (ou une paire mesurée en avance) ne la compte pas contre elle-même, et un retour en arrière
  // (seek, rechargement) oublie la rafale ; un saut en avant de moins de kCutBurstUs la garde, sans mal.
  bool after_cut(std::int64_t left_us) const {
    return last_us_ && *last_us_ < left_us && left_us - *last_us_ <= kCutBurstUs;
  }
  // La paire qui commence à left_us, coupée ou non selon la seule mesure (measured_cut).
  void note(std::int64_t left_us, bool measured) {
    if (measured) last_us_ = left_us;
  }

 private:
  std::optional<std::int64_t> last_us_;
};

// Une coupe des plans n'est gardée que si la mesure de RIFE, rapportée à la zone hors bandes noires, dépasse
// ce plancher. Relevé sur 30 désaccords
// jugés à l'image (Re:Zero) : les vraies coupes que RIFE manquait avaient 0,123 à 0,145, les fausses
// coupes des plans sous 0,08 étaient toutes des effets.
inline constexpr double kPlanCutFloor = 0.09;

// Vrai si la paire enjambe une coupe : une coupe des plans au-dessus du plancher, ou la mesure de RIFE (l'une
// ou l'autre, P3.31), sauf la mesure en rafale dans un plan que les plans disent continu (CutBurst).
bool pair_cut(const PairFacts& facts);
// La coupe selon la seule mesure, le score rapporté à la zone hors bandes noires : ce que le journal des
// désaccords compare aux plans (sur le score brut, toute vidéo à bandes comptait de faux désaccords, A09-01).
bool measured_cut(const PairFacts& facts);

// Vrai si l'interpolation tourne sur la paire ; sinon b sort seul.
bool rife_runs(const PairFacts& facts);

// Smooth (« Lisser les dessins tenus », --smooth-drawings, chantier 7.9) : un dessin tenu n sources (a, puis
// n - 1 copies, puis b nouveau), de 2 à kSmoothMaxHold, se fond en un seul mouvement de a vers b. La paire de
// rang k de la tenue (0 : (a, première copie) ; n - 1 : (dernière copie, b)) donne à son instant t la
// position u = (k + t) / n entre a et b : linéaire sur toute la tenue, continue d'une paire à la suivante.
// Une tenue plus longue garde le rythme des sources (une pose voulue, pas un dessin en deux ou en trois).
inline constexpr std::uint32_t kSmoothMaxHold = 6;
constexpr double smooth_hold_u(double t, std::uint32_t rank, std::uint32_t hold) {
  return (static_cast<double>(rank) + t) / static_cast<double>(hold);
}
// Les images en vol que l'étage tient en plus pour le Smooth (RingStage, extra_frames) : la tête attend
// kSmoothMaxHold - 1 suivantes, là où kMaxFrames en prévoit une.
constexpr std::size_t smooth_extra_frames(bool smooth) {
  return smooth ? std::size_t{kSmoothMaxHold} - 2 : 0;
}

// Dessin tenu avant la paire (la paire précédente n'a pas bougé) : RIFE avec poses n'a rien à calculer
// (mouvement précédent nul, kernels/spec/poses.h : a tient jusqu'à 1/2, puis b paraît). Avant 1/2 :
// l'instant est a, déjà à l'écran (Before) ; à 1/2 et après : b (After).
enum class PoseCopy : std::uint8_t { None, Before, After };
constexpr PoseCopy poses_copy(bool keep_poses, bool before_still, float t) {
  if (!keep_poses || !before_still) return PoseCopy::None;
  return t < 0.5F ? PoseCopy::Before : PoseCopy::After;
}

}  // namespace ajp::app::play
