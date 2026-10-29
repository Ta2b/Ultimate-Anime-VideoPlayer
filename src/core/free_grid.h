// Grille « libre » des instants de sortie d'une paire de sources (mode Auto libre, docs/auto-libre.md).
//
// Pas de facteur, pas de grille uniforme : pour chaque paire (g, d), les instants t_k de (g.pts, d.pts] sont
// choisis à partir de l'horloge prédite et du temps de service mesuré :
//   t_0 = max(échéance suivante, dernier instant émis + écart minimal, horloge prédite à la fin du calcul)
//   t_{k+1} = t_k + max(écart minimal, temps de service × marge)         tant que t < d.pts.
// Un instant n'est jamais avant l'horloge prédite (il serait déjà passé quand l'image paraît) ; la dernière
// image de la paire est la source droite elle-même, qui remplace l'instant trop proche d'elle. Tout est en
// µs de temps de contenu (pts) : la phase se reporte d'une paire à l'autre (échéance suivante), sans dérive
// puisque chaque instant est un pts exact et non un compte d'images.
//
// Les instants sortent sous la forme de CadenceStep (t = (t_k − g)/(d − g) exact en µs), comme
// CadenceAccumulator : l'étage gpu n'a qu'une grille à lire.
#pragma once

#include <cstdint>
#include <expected>
#include <optional>

#include "core/cadence.h"

namespace ajp::core {

// Ce que le lecteur sait au moment de préparer une paire, en µs de temps de contenu.
struct FreeInputs {
  // Le pts qu'aura l'image qui serait soumise maintenant quand elle paraîtra à l'écran (horloge maître
  // prédite à l'instant de fin de calcul + retard vidéo mesuré). INT64_MIN : horloge inconnue, rien ne borne.
  std::int64_t ready_us = INT64_MIN;
  std::int64_t min_gap_us = 1;  // écart minimal entre deux images (période de l'écran ou du plafond)
  std::int64_t step_us = 0;     // temps de service × marge ; 0 : inconnu (deux images par paire)
  // Le nombre maximal d'instants de la paire que le moteur accepte (1 à kMaxInstantsPerPair).
  std::size_t max_instants = kMaxInstantsPerPair;
  // Écran à fréquence fixe (sans VRR) : les instants tombent sur la grille des vsyncs, vsync_origin_us + k ×
  // vsync_period_us (temps de contenu, le pts où une image doit être pour paraître à un vsync). Une image ne
  // paraît ainsi jamais entre deux vsyncs : l'écart minimal devient la période, et un GPU trop lent laisse
  // l'image tenue un vsync de plus. 0 : VRR, aucune grille.
  std::int64_t vsync_period_us = 0;
  std::int64_t vsync_origin_us = 0;
};

class FreeGrid {
 public:
  // La source suivante, dans l'ordre de présentation.
  CadenceStep push(const SourceStamp& source, const FreeInputs& inputs);

  // La paire n'a pas été interpolée (coupe, dessin tenu, moteur absent) : la grille repart de la source
  // droite.
  void resync();

  // Une source sautée avant calcul, seek ou changement de plan : la prochaine source redevient l'origine.
  void reset();
  std::expected<void, Refusal> flush(std::uint32_t epoch);

  std::optional<std::int64_t> next_due_us() const { return next_due_; }

 private:
  CadenceStep origin(const SourceStamp& source, GridEvent event);

  std::optional<SourceStamp> left_;
  std::optional<std::int64_t> next_due_;      // où l'instant suivant tomberait, au pas courant
  std::optional<std::int64_t> last_emitted_;  // dernier pts émis
  std::uint32_t epoch_ = 0;
  bool started_ = false;
};

}  // namespace ajp::core
