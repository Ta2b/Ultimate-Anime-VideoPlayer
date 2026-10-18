// Chaque en-tête de core_types se compile seul.
#include "core/frame.h"
#include "core/inbox.h"
#include "core/messages.h"
#include "core/types.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("core_types : une boîte gpu porte images, slots, commandes, Program et Flush", "[core_types]") {
  using GpuInbox = ajp::Inbox<ajp::DecodedFrame, ajp::SlotFree, ajp::Command, ajp::Install, ajp::Flush>;
  GpuInbox inbox({4, 8, 16, 1, 1});
  CHECK(inbox.try_push(ajp::DecodedFrame{.id = {1}, .width = 1920, .height = 1080}));
  CHECK(inbox.try_push(ajp::Command{ajp::cmd::Seek{.target_us = 5'000'000}}));
  CHECK(inbox.try_push(ajp::Flush{.epoch = 1}));
  CHECK_FALSE(inbox.try_push(ajp::Flush{.epoch = 2}));
}
