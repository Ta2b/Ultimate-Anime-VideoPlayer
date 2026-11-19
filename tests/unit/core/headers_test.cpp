// Chaque en-tête de core_types se compile seul.
#include <catch2/catch_test_macros.hpp>

#include "core/frame.h"
#include "core/gpu_device.h"
#include "core/inbox.h"
#include "core/messages.h"
#include "core/ring.h"
#include "core/thread.h"
#include "core/types.h"

TEST_CASE("core_types : une boîte gpu porte images, slots, commandes, Program et Flush", "[core_types]") {
  using GpuInbox =
      kirari::Inbox<kirari::DecodedFrame, kirari::SlotFree, kirari::Command, kirari::Install, kirari::Flush>;
  GpuInbox inbox({4, 8, 16, 1, 1});
  CHECK(inbox.try_push(kirari::DecodedFrame{.id = {1}, .width = 1920, .height = 1080}));
  CHECK(inbox.try_push(kirari::Command{kirari::cmd::Seek{.target_us = 5'000'000}}));
  CHECK(inbox.try_push(kirari::Flush{.epoch = 1}));
  CHECK_FALSE(inbox.try_push(kirari::Flush{.epoch = 2}));
}
