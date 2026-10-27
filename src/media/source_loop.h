// Boucle du thread source. Interne au module : SourceThread la lance.
#pragma once

#include <cstdint>
#include <optional>
#include <stop_token>

#include "media/packet_reserve.h"
#include "media/source.h"

namespace ajp::media {

class SourceLoop {
 public:
  SourceLoop(SourceMedia media, SourceLinks links, SourceInbox& inbox);

  // Tourne jusqu'à l'arrêt ou cmd::Quit. N'attend que la boîte.
  void run(const std::stop_token& stop);

 private:
  enum class Progress : std::uint8_t {
    Moved,    // un pas de fait : on continue sans attendre
    Blocked,  // un puits est plein : on attend un Credit
    Idle,     // fin de flux, échec ou Flush : on attend un Seek
  };

  bool has_mail() const;
  void handle(SourceInbox::Message message);
  void on_flush(const Flush& flush);
  void on_seek(const cmd::Seek& seek);
  void forget_position();
  void recover_position();  // un seek refusé après avoir déplacé le fichier : retour à la dernière image
  Progress advance();
  Progress decode();
  Progress route(Packet packet);
  void keep(DecodedFrame frame);
  void fail(const Refusal& refusal);

  SourceMedia media_;
  SourceLinks links_;
  SourceInbox& inbox_;
  PacketReserve reserve_;  // paquets annexes refusés, voir media/source.h

  std::optional<DecodedFrame> held_frame_;    // décodée, pas encore acceptée
  std::optional<std::int64_t> drop_before_;   // seek exact en cours : cible en µs
  std::optional<std::int64_t> last_pts_us_;  // pts de la dernière image tenue : où revenir si un seek échoue
  std::optional<DecodedFrame> last_dropped_;  // seek exact : livrée si la fin vient avant la cible
  std::uint32_t epoch_ = 0;
  std::uint32_t next_id_ = 1;   // 0 reste « aucune source »
  bool primed_ = false;         // gpu a accepté une image de l'époque à cette position
  bool idle_ = false;           // fin de flux ou échec
  bool awaiting_seek_ = false;  // Flush reçu : rien ne part avant le Seek
  bool quit_ = false;
};

}  // namespace ajp::media
