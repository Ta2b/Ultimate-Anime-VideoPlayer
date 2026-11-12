// Les décisions de la pré-analyse des plans (media/shot_probe.h), sans fichier ni thread : où repartir après
// un saut de la lecture, et à quel rythme décoder quand l'avance est prise. Exposées pour les tests (C42-11).
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include "media/shot_timeline.h"

namespace ajp::media {

// Un saut de la lecture vers `to` : l'instant d'où l'analyse doit repartir, rien si elle continue où elle en
// est. Déjà analysé assez loin : rien. B42-09 : connu jusqu'à la fin du fichier (les dernières secondes),
// rien ; une cible connue devant l'analyse la fait partir du bout de la zone connue, sans redécoder la zone.
inline std::optional<std::int64_t> restart_for_seek(const ShotTimeline& timeline, std::int64_t to,
                                                    std::int64_t frontier, std::int64_t file_end_us,
                                                    std::int64_t skip_known_us) {
  constexpr std::int64_t kEndSlackUs = 500'000;
  const std::int64_t known = timeline.analyzed_until(to);
  if (file_end_us > 0 && known >= file_end_us - kEndSlackUs) return std::nullopt;
  if (known - to < skip_known_us) return to;
  if (to > frontier) return known;
  return std::nullopt;
}

// Le rythme de l'analyse (C42-01) : à pleine vitesse tant que son avance sur la lecture est courte, puis à
// `factor` fois le temps réel ; au même rythme quand elle comble un trou derrière la lecture (le début laissé
// par une reprise, pour le cache), qui ne sert pas l'image en cours. Le décodage logiciel d'un Hi10P, que
// NVDEC ne prend pas, occupait sinon deux à quatre cœurs jusqu'à 180 s d'avance, à priorité normale sous
// Windows (les threads de FFmpeg n'héritent pas du mode de fond), pendant que la lecture du même Hi10P décode
// elle aussi en logiciel.
class ProbePace {
 public:
  ProbePace(double factor, std::int64_t free_lead_us) : factor_(factor), free_lead_us_(free_lead_us) {}
  // L'attente, en µs, avant de décoder plus loin que media_us (la dernière image mesurée), avance `lead_us`
  // sur la lecture (négative : derrière elle), à l'instant wall_us d'une horloge monotone. 0 : continuer.
  std::int64_t delay_us(std::int64_t media_us, std::int64_t lead_us, std::int64_t wall_us) {
    constexpr std::int64_t kBehindUs = 2'000'000;  // la lecture a pris un peu d'avance : encore utile
    const bool urgent = lead_us >= -kBehindUs && lead_us < free_lead_us_;
    if (!(factor_ > 0.0) || urgent || !anchored_ || media_us < anchor_media_) {
      anchored_ = true;
      anchor_media_ = media_us;  // pleine vitesse, ou un saut : le rythme repart de cette image
      anchor_wall_ = wall_us;
      return 0;
    }
    const auto due =
        anchor_wall_ + static_cast<std::int64_t>(static_cast<double>(media_us - anchor_media_) / factor_);
    return std::max<std::int64_t>(due - wall_us, 0);
  }

 private:
  double factor_;
  std::int64_t free_lead_us_;
  bool anchored_ = false;
  std::int64_t anchor_media_ = 0;
  std::int64_t anchor_wall_ = 0;
};

}  // namespace ajp::media
