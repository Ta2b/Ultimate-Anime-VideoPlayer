// Plan d'interpolation d'une paire de sources (a, b) dans la tranche verticale : combien de
// sorties b donne (les images intermédiaires de RIFE, puis b), et l'instant de chacune.
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

#include "gpu/kernel_ops.h"

namespace ajp::app::play {

inline constexpr std::uint32_t kMaxRifeFactor = 8;
inline constexpr std::int64_t kMaxPairGapUs = 250'000;

struct PairFacts {
  std::uint32_t factor = 1;               // ×N demandé ; 1 : pas d'interpolation
  bool same_epoch = false;                // a existe et appartient à l'époque de b
  std::int64_t gap_us = 0;                // pts(b) − pts(a)
  std::optional<gpu::PairScores> scores;  // mesures de la paire ; absentes : pas mesurée
};

// Nombre de sorties de b : factor si RIFE tourne, 1 sinon.
std::uint32_t rife_outputs(const PairFacts& facts);

// Instant de la sortie k (0 ≤ k < outputs) : a + (b − a) × (k + 1) / outputs ; la dernière est b.
std::int64_t rife_output_pts(std::int64_t pts_a, std::int64_t pts_b, std::uint32_t k, std::uint32_t outputs);

}  // namespace ajp::app::play
