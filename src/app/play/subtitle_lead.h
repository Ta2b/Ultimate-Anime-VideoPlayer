// La piste de sous-titres du conteneur lue en avance par un second démultiplexeur (A17-05, A17-06).
//
// La source ne livre les sous-titres qu'au fil de la vidéo, à partir du point clé d'un seek : une réplique
// commencée avant lui n'arrivait jamais (jusqu'à 10 s perdues avec des points clés espacés), et un délai
// négatif lisait des instants que la source n'avait pas encore atteints. SubtitleLead ouvre le fichier une
// seconde fois, n'y garde que la piste choisie et l'envoie à subs :
// - à chaque seek, de kSubtitlePrerollUs avant la cible jusqu'à kSubtitleLeadUs après elle (plus |délai|) ;
// - tant que le délai est négatif, jusqu'à |délai| + kSubtitleLeadUs après l'instant demandé.
// Les paquets que la source livre aussi sont des doublons que la piste écarte (SubtitleTrack::add_packet).
// Un fichier externe n'en a pas besoin : il est lu en entier.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <vector>

#include "media/demuxer.h"
#include "subs/subs_messages.h"

namespace ajp::app::play {

inline constexpr std::int64_t kSubtitlePrerollUs = 10'000'000;  // relu avant la cible d'un seek
inline constexpr std::int64_t kSubtitleLeadUs = 1'000'000;      // lu au-delà de l'instant voulu

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
  ~SubtitleLead();  // interrompt une lecture en cours, puis attend le thread

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
  static constexpr std::int64_t kNothingRead = std::numeric_limits<std::int64_t>::min();

  struct Job {
    int track = 0;
    std::optional<std::int64_t> place;  // où replacer le fichier avant de lire
    std::uint64_t generation = 0;
  };

  void run(const std::stop_token& stop);
  bool wanted() const;  // sous mutex_
  bool prepare(const Job& job);
  void read_one(const Job& job, const std::stop_token& stop);
  bool push(subs::TrackPacket packet, std::uint64_t generation, const std::stop_token& stop);

  SubtitleLeadConfig config_;
  std::unique_ptr<media::Demuxer> demuxer_;  // écrit par le thread sous mutex_, interrompu par le destructeur
  std::optional<int> kept_;                  // thread : la piste que garde le démultiplexeur
  bool unusable_ = false;                    // thread : le fichier ne s'est pas ouvert une seconde fois

  mutable std::mutex mutex_;
  std::condition_variable_any wake_;
  std::optional<int> track_;
  std::optional<std::int64_t> place_;  // seek reçu, pas encore fait
  std::int64_t want_us_ = kNothingRead;
  std::int64_t read_us_ = kNothingRead;  // pts du dernier paquet lu depuis le dernier placement
  std::int64_t delay_us_ = 0;
  std::uint64_t generation_ = 0;  // change à chaque seek ou piste : un paquet d'avant est jeté
  std::uint64_t sent_ = 0;
  bool ended_ = false;  // fin du fichier ou lecture refusée : rien jusqu'au prochain seek
  bool busy_ = false;

  std::jthread thread_;  // en dernier : il part quand tout le reste existe
};

}  // namespace ajp::app::play
