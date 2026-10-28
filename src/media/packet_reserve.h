// Réserve des paquets audio et sous-titres que leur puits a refusés pendant
// l'amorçage d'une époque. Interne au module : SourceLoop la possède.
//
// Une file par piste, dans l'ordre du fichier : un paquet ne part qu'après
// ceux de sa piste déjà tenus. Le plafond porte sur la somme des footprint de
// toutes les pistes ; le dépasser est un refus PacketBacklog, jamais une
// attente. Les paquets portent l'époque de leur livraison, pas de leur lecture.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <span>
#include <vector>

#include "core/types.h"
#include "media/packet.h"

namespace ajp::media {

class PacketReserve {
 public:
  // sinks[i] reçoit les paquets de la piste routée i ; max_bytes borne la réserve.
  PacketReserve(std::span<PacketSink* const> sinks, std::size_t max_bytes);

  // Livre ce qui peut l'être, piste par piste et dans l'ordre, sous `epoch`.
  // true : la réserve est vide.
  bool offer(std::uint32_t epoch);

  // Livre `packet` à la piste `route` sous `epoch`, ou le garde derrière ceux
  // qu'elle tient déjà. PacketBacklog si le plafond est dépassé : la réserve
  // est alors vidée, et la source s'arrête jusqu'au Seek suivant.
  std::expected<void, Refusal> route(std::size_t route, Packet packet, std::uint32_t epoch);

  // Au Flush et au Seek : ce qui est tenu appartient à l'ancienne position.
  void clear();

  bool empty() const { return bytes_ == 0; }
  std::size_t bytes() const { return bytes_; }
  std::size_t held(std::size_t route) const { return lanes_[route].packets.size(); }

 private:
  struct Lane {
    PacketSink* sink;
    std::deque<Packet> packets;
  };

  std::vector<Lane> lanes_;
  std::size_t max_bytes_;
  std::size_t bytes_ = 0;
};

}  // namespace ajp::media
