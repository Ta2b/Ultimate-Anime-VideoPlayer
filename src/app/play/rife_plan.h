// Plan d'interpolation d'une paire de sources (a, b) dans la tranche verticale : si RIFE (ou un
// autre moteur d'interpolation) y tourne. Les instants de sortie viennent de core/cadence.h.
//
// RIFE ne tourne pas sur une paire qui ne bouge pas ou qui coupe :
//   - coupe de scène (kernels/spec/detect.h, score > 0,150) : mélanger deux plans ferait un
//     fantôme ;
//   - dessin tenu (pire bloc 32 × 32 sous 0,70 niveau) : interpoler une image avec elle-même
//     redonne la même image, au prix de RIFE ;
//   - écart de pts nul, négatif ou plus long que kMaxPairGapUs (trou, reprise) ;
//   - pas de source a dans la même époque (début, seek).
// b sort alors seul, et l'écran garde a jusqu'à son pts. En animation sur deux ou trois images,
// la plupart des paires sont tenues : RIFE n'y coûte rien.
#pragma once

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
};

// Vrai si l'interpolation tourne sur la paire ; sinon b sort seul.
bool rife_runs(const PairFacts& facts);

}  // namespace ajp::app::play
