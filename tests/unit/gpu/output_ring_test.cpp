// Ring de sortie côté gpu : états des slots et valeurs des sémaphores (protocole de core/ring.h).
#include "gpu/output_ring.h"

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu_test_support.h"

using ajp::RefusalCode;
using ajp::RingFrame;
using ajp::RingSlotDesc;
using ajp::SlotFree;
using namespace ajp::gpu;

namespace {

std::vector<RingSlotDesc> slots(std::size_t n, std::uint32_t generation = 3) {
  std::vector<RingSlotDesc> ring;
  for (std::size_t i = 0; i < n; ++i) {
    ring.push_back(
        {.index = static_cast<std::uint32_t>(i), .generation = generation, .width = 8, .height = 4});
  }
  return ring;
}

OutputRing make(std::size_t n) {
  auto ring = OutputRing::create(slots(n));
  REQUIRE(ring);
  return *ring;
}

}  // namespace

TEST_CASE("Ring : chaque écriture attend la valeur rendue par main et signale l'impaire suivante", "[ring]") {
  OutputRing ring = make(2);
  test::CollectingSink sink;

  auto first = ring.acquire();
  REQUIRE(first);
  CHECK(first->slot == 0);
  CHECK(first->sync.wait_value == 0);
  CHECK(first->sync.signal_value == 1);
  ring.commit(first->slot, RingFrame{.pts_us = 100, .epoch = 0});
  CHECK(ring.state(0) == SlotState::Ready);
  CHECK(ring.deliver(sink) == 1);
  CHECK(ring.state(0) == SlotState::Shown);
  REQUIRE(sink.frames.size() == 1);
  CHECK(sink.frames[0].slot == 0);
  CHECK(sink.frames[0].generation == 3);
  CHECK(sink.frames[0].ready_value == 1);
  CHECK(sink.frames[0].pts_us == 100);

  auto second = ring.acquire();
  REQUIRE(second);
  CHECK(second->slot == 1);
  ring.commit(second->slot, RingFrame{});
  CHECK_FALSE(ring.acquire());  // tous pris : le thread attend un SlotFree

  for (const SlotFree& free : sink.free_all()) CHECK(ring.release(free) == Release::Freed);
  ring.deliver(sink);
  for (const SlotFree& free : sink.free_all()) CHECK(ring.release(free) == Release::Freed);
  auto again = ring.acquire();
  REQUIRE(again);
  CHECK(again->slot == 0);
  CHECK(again->sync.wait_value == 2);
  CHECK(again->sync.signal_value == 3);
}

TEST_CASE("Ring : main plein, une écriture reste prête et part dans l'ordre", "[ring]") {
  OutputRing ring = make(3);
  test::CollectingSink sink(1);
  for (int i = 0; i < 3; ++i) {
    auto lease = ring.acquire();
    REQUIRE(lease);
    ring.commit(lease->slot, RingFrame{.pts_us = i});
  }
  CHECK(ring.deliver(sink) == 1);
  CHECK(ring.count(SlotState::Ready) == 2);
  sink.free_all();
  CHECK(ring.deliver(sink) == 1);
  sink.free_all();
  CHECK(ring.deliver(sink) == 1);
  REQUIRE(sink.frames.size() == 3);
  CHECK(sink.frames[1].pts_us == 1);
  CHECK(sink.frames[2].pts_us == 2);
}

TEST_CASE("Ring : un Flush rend les slots prêts sans passer par main", "[ring]") {
  OutputRing ring = make(2);
  auto old = ring.acquire();
  REQUIRE(old);
  ring.commit(old->slot, RingFrame{.epoch = 0});
  auto kept = ring.acquire();
  REQUIRE(kept);
  ring.commit(kept->slot, RingFrame{.epoch = 1});
  CHECK(ring.drop_before(1) == 1);
  CHECK(ring.state(0) == SlotState::Free);
  CHECK(ring.state(1) == SlotState::Ready);
  // L'écriture abandonnée a signalé 1 : la suivante l'attend et signale 3, toujours impaire.
  auto next = ring.acquire();
  REQUIRE(next);
  CHECK(next->slot == 0);
  CHECK(next->sync.wait_value == 1);
  CHECK(next->sync.signal_value == 3);
  ring.abandon(next->slot);
  CHECK(ring.state(0) == SlotState::Free);
}

TEST_CASE("Ring : SlotFree hors protocole", "[ring]") {
  OutputRing ring = make(2);
  CHECK(ring.release({.slot = 0, .semaphore_value = 2, .generation = 9}) == Release::OtherGeneration);
  CHECK(ring.release({.slot = 5, .semaphore_value = 2, .generation = 3}).error().code ==
        RefusalCode::SequenceOrder);
  CHECK(ring.release({.slot = 0, .semaphore_value = 2, .generation = 3}).error().code ==
        RefusalCode::SequenceOrder);

  test::CollectingSink sink;
  auto lease = ring.acquire();
  REQUIRE(lease);
  ring.commit(lease->slot, RingFrame{});
  ring.deliver(sink);
  CHECK(ring.release({.slot = 0, .semaphore_value = 4, .generation = 3}).error().code ==
        RefusalCode::SequenceOrder);
  CHECK(ring.release({.slot = 0, .semaphore_value = 2, .generation = 3}) == Release::Freed);
}

TEST_CASE("Ring : refus à la création", "[ring]") {
  CHECK(OutputRing::create({}).error().code == RefusalCode::FieldOutOfRange);
  CHECK(OutputRing::create(slots(kMaxRingSlots + 1)).error().code == RefusalCode::FieldOutOfRange);
  auto mixed = slots(2);
  mixed[1].generation = 4;
  CHECK(OutputRing::create(mixed).error().code == RefusalCode::FieldOutOfRange);
  auto shuffled = slots(2);
  shuffled[0].index = 1;
  CHECK(OutputRing::create(shuffled).error().code == RefusalCode::FieldOutOfRange);
  auto sizes = slots(2);
  sizes[1].width = 16;
  CHECK(OutputRing::create(sizes).error().code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("Ring : il reçoit la sortie d'un Plan de sa taille et de son format", "[ring]") {
  const OutputRing ring = make(2);
  ajp::core::Plan plan;
  plan.output_width = 8;
  plan.output_height = 4;
  CHECK(ring.fits(plan));
  plan.output_height = 8;
  CHECK_FALSE(ring.fits(plan));
  plan.output_height = 4;
  plan.ring = ajp::RingFormat::Yuv444p16;
  CHECK_FALSE(ring.fits(plan));
}
