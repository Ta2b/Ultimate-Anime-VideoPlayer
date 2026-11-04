// Une seule source de vérité des coupes (chantier 8.7, essai --shot-cuts) : les plans de la pré-analyse
// (media/shot_timeline.h), que main donne au livre, disent si une paire d'images (a, b) enjambe une coupe.
// Quand la chronologie couvre la paire, l'interpolation prend sa réponse et la mesure de RIFE ne fait plus
// que signaler les désaccords ; hors de la chronologie (pas encore analysée, départ d'analyse sans coupe
// connue), rien : la mesure décide comme avant. Le grade de l'anneau y lit aussi le début de chaque plan, où
// sa mémoire temporelle repart. Livre vide (sans l'essai) : rien ne change.
#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "media/shot_cutter.h"

namespace ajp::app::play {

// La réponse des plans pour la paire (a, b], a < b : vrai si b est dans un plan qui commence à une vraie
// coupe après a, faux si a et b sont dans le même plan, rien si l'un des deux n'est dans aucun plan ou si le
// plan de b commence sans coupe connue (départ d'analyse).
std::optional<bool> known_cut(std::span<const media::Shot> shots, std::int64_t a_us, std::int64_t b_us);

class CutBook {
 public:
  void set_shots(std::span<const media::Shot> shots);
  std::optional<bool> cut_between(std::int64_t a_us, std::int64_t b_us) const;
  bool active() const;

 private:
  mutable std::mutex mutex_;
  std::vector<media::Shot> shots_;
};

}  // namespace ajp::app::play
