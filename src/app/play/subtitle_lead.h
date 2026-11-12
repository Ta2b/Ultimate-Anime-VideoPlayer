// La piste de sous-titres du conteneur lue en avance par un second démultiplexeur (A17-05, A17-06).
//
// La source ne livre les sous-titres qu'au fil de la vidéo, à partir du point clé d'un seek : une réplique
// commencée avant lui n'arrivait jamais (jusqu'à 10 s perdues avec des points clés espacés), et un délai
// négatif lisait des instants que la source n'avait pas encore atteints. SubtitleLead ouvre le fichier une
// seconde fois, n'y garde que la piste choisie et l'envoie à subs :
// - à chaque seek, de kSubtitlePrerollUs avant la cible jusqu'à kSubtitleLeadUs après elle (plus |délai|),
//   puis, en second, de kSubtitleBackfillUs avant la cible jusqu'au début de cette première plage : les
//   répliques commencées bien avant la cible et encore à l'écran (générique défilant, pancarte d'une scène,
//   B17-07). Une plage déjà lue n'est pas relue ;
// - tant que le délai est négatif, jusqu'à |délai| + kSubtitleLeadUs après l'instant demandé.
// Les paquets que la source livre aussi sont des doublons que la piste écarte (SubtitleTrack::add_packet).
// Un fichier externe n'en a pas besoin : il est lu en entier.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "subs/subs_messages.h"

namespace ajp::app::play {

inline constexpr std::int64_t kSubtitlePrerollUs = 10'000'000;    // relu avant la cible d'un seek
inline constexpr std::int64_t kSubtitleBackfillUs = 120'000'000;  // relu en second, avant la préroulée
inline constexpr std::int64_t kSubtitleLeadUs = 1'000'000;        // lu au-delà de l'instant voulu
inline constexpr std::chrono::milliseconds kSubtitleStopLimit{2000};

struct SubtitleLeadConfig {
  std::filesystem::path file;
  int reference = 0;          // la piste vidéo : son index place le fichier
  std::vector<int> mov_text;  // pistes mov_text, dont le texte suit une longueur (A18-15)
  std::shared_ptr<subs::SubsInbox> inbox;
};

class SubtitleLead {
 public:
  // Le fichier s'ouvre dans le thread, au premier besoin : l'ouverture de la lecture n'attend pas.
  explicit SubtitleLead(SubtitleLeadConfig config);
  SubtitleLead(const SubtitleLead&) = delete;
  SubtitleLead& operator=(const SubtitleLead&) = delete;
  // Interrompt une lecture en cours, puis attend le thread au plus kSubtitleStopLimit : au-delà, il est
  // laissé à son sort (abandon_thread), sans plus toucher qu'à l'état qu'il partage.
  ~SubtitleLead();

  // Depuis main. La piste du conteneur dont les paquets partent ; rien : aucune (piste externe).
  void select(std::optional<int> track);
  // Un seek vers target_us : relit la piste de kSubtitlePrerollUs avant lui.
  void seek(std::int64_t target_us);
  // Une demande de rendu à pts_us, avec le délai des sous-titres : un délai négatif lit en avance.
  void follow(std::int64_t pts_us, std::int64_t delay_us);

  // Pour les tests : paquets envoyés à subs depuis le départ.
  std::uint64_t sent() const;
  // Pour les tests : vrai quand le thread n'a plus rien à lire.
  bool idle() const;

 private:
  // L'état que le thread partage : il survit à la SubtitleLead quand le thread est laissé à son sort
  // (B30-01 : une lecture suspendue à un partage réseau muet ne fige plus la fenêtre à la fermeture).
  struct Core;
  std::shared_ptr<Core> core_;
  std::jthread thread_;  // en dernier : il part quand tout le reste existe
};

}  // namespace ajp::app::play
