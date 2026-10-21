#include "core/inbox.h"

#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {

struct Frame {
  int index;
};
struct Command {
  std::string text;
};
using TestInbox = ajp::Inbox<Frame, Command>;

}  // namespace

TEST_CASE("Inbox : chaque type a sa propre borne", "[Inbox]") {
  TestInbox inbox({2, 1});
  CHECK(inbox.try_push(Frame{1}));
  CHECK(inbox.try_push(Frame{2}));
  CHECK_FALSE(inbox.try_push(Frame{3}));
  CHECK(inbox.try_push(Command{"pause"}));
  CHECK_FALSE(inbox.try_push(Command{"seek"}));
  CHECK(inbox.pending<Frame>() == 2);
  CHECK(inbox.pending<Command>() == 1);
}

TEST_CASE("Inbox : wait rend les messages dans l'ordre d'arrivée et libère la borne", "[Inbox]") {
  TestInbox inbox({1, 1});
  std::stop_source stop;
  REQUIRE(inbox.try_push(Frame{7}));
  REQUIRE(inbox.try_push(Command{"pause"}));

  auto first = inbox.wait(stop.get_token());
  REQUIRE(first);
  CHECK(std::get<Frame>(*first).index == 7);
  CHECK(inbox.try_push(Frame{8}));  // la place est revenue

  auto second = inbox.wait(stop.get_token());
  REQUIRE(second);
  CHECK(std::get<Command>(*second).text == "pause");
}

TEST_CASE("Inbox : l'arrêt réveille un thread qui attend", "[Inbox]") {
  TestInbox inbox({1, 1});
  bool got_message = true;
  std::jthread consumer([&](std::stop_token stop) { got_message = inbox.wait(stop).has_value(); });
  consumer.request_stop();
  consumer.join();
  CHECK_FALSE(got_message);
}

TEST_CASE("Inbox : plusieurs producteurs, un consommateur, rien de perdu", "[Inbox]") {
  constexpr int kProducers = 4;
  constexpr int kPerProducer = 1000;
  TestInbox inbox({8, 8});
  std::vector<int> seen(kProducers * kPerProducer, 0);

  std::jthread consumer([&](std::stop_token stop) {
    for (int received = 0; received < kProducers * kPerProducer; ++received) {
      auto message = inbox.wait(stop);
      if (!message) return;
      ++seen[static_cast<std::size_t>(std::get<Frame>(*message).index)];
    }
  });
  {
    std::vector<std::jthread> producers;
    for (int p = 0; p < kProducers; ++p) {
      producers.emplace_back([&inbox, p] {
        for (int i = 0; i < kPerProducer; ++i) {
          while (!inbox.try_push(Frame{p * kPerProducer + i})) std::this_thread::yield();
        }
      });
    }
  }
  consumer.join();
  for (int count : seen) CHECK(count == 1);
}
