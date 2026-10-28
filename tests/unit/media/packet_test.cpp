#include "media/packet.h"

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using ajp::media::kMaxQueueBytes;
using ajp::media::kMaxQueueDurationUs;
using ajp::media::kMinQueueBytes;
using ajp::media::kPacketOverhead;
using ajp::media::Packet;
using ajp::media::PacketQueue;
using ajp::media::PacketQueueLimits;

namespace {

// 20 ms sans données : footprint kPacketOverhead.
Packet audio(std::int64_t pts_us, std::uint32_t epoch = 0) {
  return Packet{.track = 1, .pts_us = pts_us, .duration_us = 20'000, .epoch = epoch};
}

}  // namespace

TEST_CASE("PacketQueue : bornes hors plage refusées", "[media][packet]") {
  const auto refused = [](PacketQueueLimits limits) {
    auto queue = PacketQueue::create(limits, [] {});
    return !queue && queue.error().code == RefusalCode::FieldOutOfRange;
  };
  CHECK(refused({.max_duration_us = 0}));
  CHECK(refused({.max_duration_us = -1}));
  CHECK(refused({.max_duration_us = kMaxQueueDurationUs + 1}));
  CHECK(refused({.max_bytes = kMinQueueBytes - 1}));
  CHECK(refused({.max_bytes = kMaxQueueBytes + 1}));
  CHECK(PacketQueue::create({.max_duration_us = kMaxQueueDurationUs, .max_bytes = kMinQueueBytes}, [] {}));
  CHECK(PacketQueue::create({.max_duration_us = std::nullopt, .max_bytes = kMaxQueueBytes}, [] {}));
  CHECK(PacketQueue::create({}, [] {}));
}

TEST_CASE("PacketQueue : bornée en durée, réveille le producteur une fois", "[media][packet]") {
  int wakeups = 0;
  auto queue = PacketQueue::create({.max_duration_us = 60'000}, [&] { ++wakeups; });
  REQUIRE(queue);
  PacketQueue& q = **queue;
  CHECK(q.try_push(audio(0)));
  CHECK(q.try_pop().has_value());
  CHECK(wakeups == 0);  // aucun refus : personne n'attend

  CHECK(q.try_push(audio(20'000)));
  CHECK(q.try_push(audio(40'000)));
  CHECK(q.try_push(audio(60'000)));  // 60 ms tout juste
  CHECK(q.duration_us() == 60'000);
  CHECK_FALSE(q.try_push(audio(80'000)));
  CHECK_FALSE(q.try_push(audio(80'000)));
  CHECK(q.size() == 3);
  auto first = q.try_pop();
  REQUIRE(first);
  CHECK(first->pts_us == 20'000);  // ordre d'arrivée
  CHECK(wakeups == 1);
  CHECK(q.duration_us() == 40'000);
  CHECK(q.try_pop()->pts_us == 40'000);
  CHECK(q.try_pop()->pts_us == 60'000);
  CHECK(wakeups == 1);
  CHECK_FALSE(q.try_pop().has_value());
  CHECK(q.duration_us() == 0);
  CHECK(q.bytes() == 0);
}

TEST_CASE("PacketQueue : bornée en octets, une durée inconnue ne compte pas", "[media][packet]") {
  // Sous-titres : sans borne de durée, des centaines d'évènements au même instant.
  auto queue =
      PacketQueue::create({.max_duration_us = std::nullopt, .max_bytes = 4 * kPacketOverhead}, [] {});
  REQUIRE(queue);
  PacketQueue& q = **queue;
  CHECK(q.try_push(Packet{.track = 2, .pts_us = 0, .duration_us = 3'600'000'000}));
  CHECK(q.try_push(Packet{.track = 2, .pts_us = 0, .duration_us = -5}));
  CHECK(q.try_push(Packet{.track = 2, .pts_us = 0}));
  CHECK(q.try_push(Packet{.track = 2, .pts_us = 0}));
  CHECK(q.bytes() == 4 * kPacketOverhead);
  CHECK_FALSE(q.try_push(Packet{.track = 2, .pts_us = 0}));

  auto timed = PacketQueue::create({.max_duration_us = 40'000}, [] {});
  REQUIRE(timed);
  CHECK((*timed)->try_push(Packet{.duration_us = -1}));
  CHECK((*timed)->try_push(Packet{.duration_us = 0}));
  CHECK((*timed)->duration_us() == 0);
}

TEST_CASE("PacketQueue : une file vide accepte toujours un paquet", "[media][packet]") {
  auto queue = PacketQueue::create({.max_duration_us = 20'000, .max_bytes = kMinQueueBytes}, [] {});
  REQUIRE(queue);
  PacketQueue& q = **queue;
  CHECK(q.try_push(Packet{.duration_us = 5'000'000}));  // plus long que la borne
  CHECK_FALSE(q.try_push(audio(0)));
  CHECK(q.try_pop().has_value());
  CHECK(q.try_push(audio(0)));
}

TEST_CASE("PacketQueue : discard_before jette les époques passées", "[media][packet]") {
  int wakeups = 0;
  auto queue = PacketQueue::create({.max_duration_us = 60'000}, [&] { ++wakeups; });
  REQUIRE(queue);
  PacketQueue& q = **queue;
  CHECK(q.try_push(audio(1, 0)));
  CHECK(q.try_push(audio(2, 0)));
  CHECK(q.try_push(audio(3, 1)));
  CHECK_FALSE(q.try_push(audio(4, 1)));
  q.discard_before(1);
  CHECK(wakeups == 1);
  CHECK(q.size() == 1);
  CHECK(q.duration_us() == 20'000);
  CHECK(q.bytes() == kPacketOverhead);
  CHECK(q.try_pop()->pts_us == 3);
  CHECK(Packet{}.data().empty());
  CHECK(Packet{}.footprint() == kPacketOverhead);
}
