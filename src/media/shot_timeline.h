// La chronologie des plans d'une vidéo : ce que la pré-analyse a trouvé (media/shot_probe.h), lu par le
// grade à l'horodatage de l'image. Écrite par le thread d'analyse, lue par le thread principal :
// toutes les méthodes sont sûres entre threads.
#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "media/shot_cutter.h"

namespace ajp::media {

// Le plan d'une image : son rang dans la chronologie (il change à chaque coupe) et ses valeurs à cet
// instant (celles d'un fondu, interpolées entre son début et sa fin).
struct ShotAt {
  std::int64_t start_us = 0;
  std::int64_t end_us = 0;
  std::uint64_t serial = 0;  // propre au plan : deux images du même plan ont le même
  ShotStats stats;
  bool fade = false;
  bool cut_before = true;
};

class ShotTimeline {
 public:
  void add(const Shot& found);
  // Une séquence de flashs dangereux (media/flash_guard.h), fusionnée à celles qu'elle touche.
  void add_flash(FlashSpan span);
  // pts_us est dans une séquence de flashs dangereux connue.
  bool flashing(std::int64_t pts_us) const;
  std::vector<FlashSpan> flashes() const;
  // Change à chaque ajout qui modifie la chronologie : un plan remplacé à nombre égal se voit (A42-11).
  std::uint64_t version() const;
  // Le plan qui contient pts_us, s'il est déjà analysé.
  std::optional<ShotAt> at(std::int64_t pts_us) const;
  // Durée analysée, en microsecondes : la somme des plans.
  std::int64_t covered_us() const;
  // L'instant jusqu'où l'analyse va sans trou depuis pts_us (pts_us lui-même si rien n'y est).
  std::int64_t analyzed_until(std::int64_t pts_us) const;
  // Le premier trou depuis le début de la vidéo (les plans se suivent à 0,1 s près) ; la fin du dernier plan
  // s'il n'y en a pas, 0 sans plan.
  std::int64_t first_gap_us() const;
  // La fin du dernier plan, 0 sans plan.
  std::int64_t end_us() const;
  std::size_t size() const;
  // Tous les plans, dans l'ordre (le cache, media/shot_cache.h).
  std::vector<Shot> shots() const;
  void clear();

 private:
  mutable std::mutex mutex_;
  std::vector<Shot> shots_;  // triés par début, sans chevauchement
  std::vector<std::uint64_t> serials_;
  std::vector<FlashSpan> flashes_;  // triées, sans chevauchement
  std::uint64_t next_serial_ = 1;
  std::uint64_t version_ = 0;
};

}  // namespace ajp::media
