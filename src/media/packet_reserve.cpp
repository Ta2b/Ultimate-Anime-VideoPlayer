#include "media/packet_reserve.h"

#include <format>
#include <utility>

namespace ajp::media {
namespace {

// Une copie : un refus ne consomme rien, et l'époque se pose à la livraison.
bool deliver(PacketSink& sink, const Packet& packet, std::uint32_t epoch) {
  Packet copy = packet;
  copy.epoch = epoch;
  return sink.try_push(std::move(copy));
}

}  // namespace

PacketReserve::PacketReserve(std::span<PacketSink* const> sinks, std::size_t max_bytes)
    : max_bytes_(max_bytes) {
  lanes_.reserve(sinks.size());
  for (PacketSink* sink : sinks) lanes_.push_back(Lane{.sink = sink, .packets = {}});
}

bool PacketReserve::offer(std::uint32_t epoch) {
  for (Lane& lane : lanes_) {
    while (!lane.packets.empty() && deliver(*lane.sink, lane.packets.front(), epoch)) {
      bytes_ -= lane.packets.front().footprint();
      lane.packets.pop_front();
    }
  }
  return empty();
}

std::expected<void, Refusal> PacketReserve::route(std::size_t route, Packet packet, std::uint32_t epoch) {
  Lane& lane = lanes_[route];
  if (lane.packets.empty() && deliver(*lane.sink, packet, epoch)) return {};
  bytes_ += packet.footprint();
  const int track = packet.track;
  lane.packets.push_back(std::move(packet));
  if (bytes_ <= max_bytes_) return {};
  const std::size_t total = bytes_;
  clear();
  return std::unexpected(
      Refusal{RefusalCode::PacketBacklog,
              std::format("réserve des pistes annexes : {} octets au-delà du plafond de {} (piste {}) ; un "
                          "consommateur ne retire plus ses paquets avant la première image de l'époque",
                          total, max_bytes_, track)});
}

void PacketReserve::clear() {
  for (Lane& lane : lanes_) lane.packets.clear();
  bytes_ = 0;
}

}  // namespace ajp::media
