// Un GPU bloqué sans être perdu (A31-14) : une file qui a du travail en attente et n'en finit aucun. Le
// pilote ne rend aucune erreur (cudaStreamQuery dit cudaErrorNotReady, le sémaphore du slot n'avance pas),
// l'image reste figée et le son continue. La boucle de main suit la plus vieille image de l'époque : écrite
// par le GPU ou non, et laquelle. La même image attendue plus de kHangUs de suite, c'est un GPU bloqué :
// refus DeviceLost, qui arrête la lecture et relance le lecteur une fois (A31-02).
//
// Deux tours de boucle séparés de plus de kHangGapUs (veille, gel du processus, débogueur) ne comptent pas
// comme une attente : le compte repart, la veille a son propre traitement (sleep_watch.h).
#pragma once

#include <cstdint>
#include <optional>

namespace ajp::app::play {

// Plus long que toute écriture d'image en lecture (une passe SPAN 4K sur une petite carte : quelques
// centaines de ms ; le premier appel d'un moteur TensorRT qui charge ses noyaux paresseusement : moins d'une
// seconde), assez court pour que l'image figée ne dure pas. La mesure qui le fixe : A-CONFIRMER, A31-14.
inline constexpr std::int64_t kHangUs = 5'000'000;
inline constexpr std::int64_t kHangGapUs = 1'000'000;

class HangWatch {
 public:
  // Un tour : waiting si une image attend son écriture, mark l'identifie (une autre image, c'est que le GPU
  // avance). Vrai quand la même image attend depuis plus de kHangUs.
  bool hung(bool waiting, std::uint64_t mark, std::int64_t now_us) {
    const std::optional<std::int64_t> last = last_us_;
    last_us_ = now_us;
    if (!waiting || !since_us_ || mark != mark_ || (last && now_us - *last > kHangGapUs)) {
      since_us_ = waiting ? std::optional(now_us) : std::nullopt;
      mark_ = mark;
      return false;
    }
    return now_us - *since_us_ > kHangUs;
  }

 private:
  std::optional<std::int64_t> since_us_;  // depuis quand la même image attend
  std::optional<std::int64_t> last_us_;   // le tour d'avant
  std::uint64_t mark_ = 0;
};

}  // namespace ajp::app::play
