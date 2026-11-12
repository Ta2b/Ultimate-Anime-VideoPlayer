// La file d'images de la session (frame_queue.h) : choix par l'horloge ou par numéro de vblank, images pas
// encore écrites, époques. Sortis d'options_test.cpp (C13-10), par le chemin du produit : pick avec ready, et
// pick_by.
#include "app/play/frame_queue.h"

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "doubles.h"

namespace ajp::app::play {
namespace {

const auto kWritten = [](const RingFrame&) { return true; };

TEST_CASE("File d'images : choix par l'horloge, retards, époques", "[app][frames]") {
  FrameQueue queue;
  for (std::uint32_t i = 0; i < 4; ++i) queue.push(test::ring_frame(i, i * 40'000));
  std::vector<RingFrame> late;
  CHECK_FALSE(queue.pick(-10'000, 0, late, kWritten));  // la tête n'est pas atteinte
  const auto picked = queue.pick(85'000, 0, late, kWritten);
  REQUIRE(picked);
  CHECK(picked->pts_us == 80'000);
  CHECK(late.size() == 2);
  CHECK(queue.size() == 1);

  queue.push(test::ring_frame(5, kNoPts, 1));
  queue.push(test::ring_frame(6, 0, 1));
  CHECK(queue.first_of(1)->slot == 5);
  const auto stale = queue.take_stale(1);
  REQUIRE(stale.size() == 1);
  CHECK(stale[0].slot == 3);
  late.clear();
  // Une image sans pts passe dès qu'elle est en tête, et seule.
  CHECK(queue.pick(-1'000'000, 0, late, kWritten)->slot == 5);
  CHECK(queue.take_all().size() == 1);
}

TEST_CASE("File d'images : l'image sans pts arrête le choix, bloc de deque après bloc", "[app][frames]") {
  // Assez d'images pour que pop_front libère plusieurs blocs de la deque : le choix ne relit
  // jamais l'image qu'il vient de retirer (ASan le verrait).
  FrameQueue queue;
  constexpr std::uint32_t kCount = 999;
  for (std::uint32_t i = 0; i < kCount; ++i)
    queue.push(test::ring_frame(i, i % 3 == 2 ? kNoPts : std::int64_t{i} * 1'000));
  std::vector<RingFrame> late;
  for (std::uint32_t i = 2; i < kCount; i += 3) {
    late.clear();
    const auto picked = queue.pick(std::int64_t{kCount} * 1'000, 0, late, kWritten);
    REQUIRE(picked);
    CHECK(picked->slot == i);
    CHECK(late.size() == 2);
  }
  CHECK(queue.empty());
}

TEST_CASE("File d'images : une image pas encore écrite arrête le choix, et reste en file", "[app][frames]") {
  FrameQueue queue;
  for (std::uint32_t i = 0; i < 4; ++i) queue.push(test::ring_frame(i, i * 40'000));
  std::vector<RingFrame> late;
  // Les slots 0 et 1 sont écrits, le 2 non : l'écran garde le 1 tant que le 2 se calcule.
  const auto written_below_two = [](const RingFrame& frame) { return frame.slot < 2; };
  const auto picked = queue.pick(200'000, 0, late, written_below_two);
  REQUIRE(picked);
  CHECK(picked->slot == 1);
  CHECK(late.size() == 1);
  CHECK(queue.size() == 2);
  CHECK(queue.first_of(0)->slot == 2);
}

TEST_CASE("File d'images : le choix par échéance (pick_by) suit le prédicat, pas le pts", "[app][frames]") {
  FrameQueue queue;
  for (std::uint32_t i = 0; i < 4; ++i) queue.push(test::ring_frame(i, i * 40'000));
  std::vector<RingFrame> late;
  // Par numéro de vblank : dues, les images jusqu'au slot 2 ; le seuil de pts n'y est pour rien.
  const auto picked = queue.pick_by([](const RingFrame& frame) { return frame.slot <= 2; }, late, kWritten);
  REQUIRE(picked);
  CHECK(picked->slot == 2);
  CHECK(late.size() == 2);
  CHECK_FALSE(queue.pick_by([](const RingFrame&) { return false; }, late, kWritten));
  CHECK(queue.size() == 1);
}

}  // namespace
}  // namespace ajp::app::play
