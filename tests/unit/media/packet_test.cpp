#include "media/packet.h"

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using ajp::media::kMaxPacketQueue;
using ajp::media::Packet;
using ajp::media::PacketQueue;

TEST_CASE("PacketQueue : borne hors plage refusée", "[media][packet]") {
  auto empty = PacketQueue::create(0, [] {});
  REQUIRE_FALSE(empty);
  CHECK(empty.error().code == RefusalCode::FieldOutOfRange);
  CHECK_FALSE(PacketQueue::create(kMaxPacketQueue + 1, [] {}));
  CHECK(PacketQueue::create(kMaxPacketQueue, [] {}));
}

TEST_CASE("PacketQueue : refuse au-delà de la borne, réveille le producteur une fois", "[media][packet]") {
  int wakeups = 0;
  auto queue = PacketQueue::create(2, [&] { ++wakeups; });
  REQUIRE(queue);
  PacketQueue& q = **queue;
  CHECK(q.try_push(Packet{.track = 1, .pts_us = 0}));
  CHECK(q.try_pop().has_value());
  CHECK(wakeups == 0);  // aucun refus : personne n'attend

  CHECK(q.try_push(Packet{.track = 1, .pts_us = 1}));
  CHECK(q.try_push(Packet{.track = 1, .pts_us = 2}));
  CHECK_FALSE(q.try_push(Packet{.track = 1, .pts_us = 3}));
  CHECK_FALSE(q.try_push(Packet{.track = 1, .pts_us = 3}));
  CHECK(q.size() == 2);
  auto first = q.try_pop();
  REQUIRE(first);
  CHECK(first->pts_us == 1);  // ordre d'arrivée
  CHECK(wakeups == 1);
  CHECK(q.try_pop()->pts_us == 2);
  CHECK(wakeups == 1);
  CHECK_FALSE(q.try_pop().has_value());
}

TEST_CASE("PacketQueue : discard_before jette les époques passées", "[media][packet]") {
  int wakeups = 0;
  auto queue = PacketQueue::create(3, [&] { ++wakeups; });
  REQUIRE(queue);
  PacketQueue& q = **queue;
  CHECK(q.try_push(Packet{.pts_us = 1, .epoch = 0}));
  CHECK(q.try_push(Packet{.pts_us = 2, .epoch = 0}));
  CHECK(q.try_push(Packet{.pts_us = 3, .epoch = 1}));
  CHECK_FALSE(q.try_push(Packet{.pts_us = 4, .epoch = 1}));
  q.discard_before(1);
  CHECK(wakeups == 1);
  CHECK(q.size() == 1);
  CHECK(q.try_pop()->pts_us == 3);
  CHECK(Packet{}.data().empty());
}
