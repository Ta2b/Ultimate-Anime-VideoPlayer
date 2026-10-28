#include "media/packet_reserve.h"

#include <array>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using ajp::media::kPacketOverhead;
using ajp::media::Packet;
using ajp::media::PacketReserve;
using ajp::media::PacketSink;

namespace {

// Puits qui accepte tant qu'on lui laisse de la place, et garde tout.
class Gate final : public PacketSink {
 public:
  bool try_push(Packet&& packet) override {
    if (room == 0) return false;
    --room;
    got.push_back(std::move(packet));
    return true;
  }
  int room = 0;
  std::vector<Packet> got;
};

Packet at(int track, std::int64_t pts_us) {
  return Packet{.track = track, .pts_us = pts_us, .epoch = 99};
}

}  // namespace

TEST_CASE("PacketReserve : un puits libre reçoit directement, l'époque posée à la livraison",
          "[media][packet]") {
  Gate audio;
  audio.room = 1;
  const std::array<PacketSink*, 1> sinks{&audio};
  PacketReserve reserve(sinks, 16 * kPacketOverhead);
  REQUIRE(reserve.route(0, at(1, 0), 3));
  CHECK(reserve.empty());
  REQUIRE(audio.got.size() == 1);
  CHECK(audio.got[0].epoch == 3);
}

TEST_CASE("PacketReserve : chaque piste garde son ordre, sans attendre les autres", "[media][packet]") {
  Gate audio;
  Gate subs;
  const std::array<PacketSink*, 2> sinks{&audio, &subs};
  PacketReserve reserve(sinks, 16 * kPacketOverhead);
  for (int k = 0; k < 3; ++k) REQUIRE(reserve.route(0, at(1, k), 0));
  REQUIRE(reserve.route(1, at(2, 10), 0));
  CHECK(reserve.held(0) == 3);
  CHECK(reserve.held(1) == 1);
  CHECK(reserve.bytes() == 4 * kPacketOverhead);

  subs.room = 8;  // les sous-titres partent, l'audio plein ne les retient pas
  CHECK_FALSE(reserve.offer(1));
  CHECK(reserve.held(1) == 0);
  REQUIRE(subs.got.size() == 1);
  CHECK(subs.got[0].epoch == 1);

  audio.room = 1;
  CHECK_FALSE(reserve.offer(1));
  REQUIRE(reserve.route(0, at(1, 3), 1));  // derrière ceux déjà tenus, même avec de la place
  audio.room = 8;
  CHECK(reserve.offer(2));
  REQUIRE(audio.got.size() == 4);
  for (std::size_t k = 0; k < audio.got.size(); ++k) CHECK(audio.got[k].pts_us == static_cast<int>(k));
  CHECK(audio.got[0].epoch == 1);
  CHECK(audio.got[1].epoch == 2);
}

TEST_CASE("PacketReserve : au-delà du plafond, refus PacketBacklog et réserve vidée", "[media][packet]") {
  Gate audio;
  const std::array<PacketSink*, 1> sinks{&audio};
  PacketReserve reserve(sinks, 3 * kPacketOverhead);
  for (int k = 0; k < 3; ++k) REQUIRE(reserve.route(0, at(1, k), 0));
  auto over = reserve.route(0, at(1, 3), 0);
  REQUIRE_FALSE(over);
  CHECK(over.error().code == RefusalCode::PacketBacklog);
  CHECK(over.error().detail.find("piste 1") != std::string::npos);
  CHECK(reserve.empty());
  audio.room = 8;
  CHECK(reserve.offer(0));
  CHECK(audio.got.empty());
}

TEST_CASE("PacketReserve : clear oublie l'ancienne position", "[media][packet]") {
  Gate audio;
  const std::array<PacketSink*, 1> sinks{&audio};
  PacketReserve reserve(sinks, 16 * kPacketOverhead);
  REQUIRE(reserve.route(0, at(1, 0), 0));
  reserve.clear();
  CHECK(reserve.empty());
  CHECK(reserve.held(0) == 0);
  audio.room = 1;
  CHECK(reserve.offer(1));
  CHECK(audio.got.empty());
}
