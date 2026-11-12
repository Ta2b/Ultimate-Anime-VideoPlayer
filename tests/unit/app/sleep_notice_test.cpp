// B31-01, B08-10 : la veille annoncée passe au thread main, qui met en pause avant qu'elle n'arrive.
#include "app/play/sleep_notice.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

using namespace std::chrono_literals;

void count(void* context) {
  ++*static_cast<std::atomic<int>*>(context);
}

}  // namespace

TEST_CASE("B31-01 : la veille annoncée réveille la boucle et attend sa pause", "[sleep]") {
  std::atomic<int> wakes{0};
  SleepGate gate(&count, &wakes);
  CHECK_FALSE(gate.take());
  // Le fil du système annonce la veille et attend.
  auto announced = std::async(std::launch::async, [&] { return gate.announce(2000ms); });
  while (wakes.load() == 0) std::this_thread::sleep_for(1ms);
  // La boucle, réveillée, prend l'annonce une fois, se met en pause, puis rend la main.
  CHECK(gate.take());
  CHECK_FALSE(gate.take());
  CHECK(announced.wait_for(20ms) == std::future_status::timeout);  // pas avant la pause
  gate.ready();
  CHECK(announced.get());
}

TEST_CASE("B31-01 : une boucle qui ne répond pas ne retient pas la veille", "[sleep]") {
  SleepGate gate(nullptr, nullptr);
  const auto start = std::chrono::steady_clock::now();
  CHECK_FALSE(gate.announce(30ms));
  CHECK(std::chrono::steady_clock::now() - start < 1s);
  CHECK_FALSE(gate.take());  // l'annonce échue ne vaut plus au réveil
}

TEST_CASE("C29-06 : un réveil dit par le système se prend une fois et réveille la boucle", "[sleep]") {
  std::atomic<int> wakes{0};
  SleepGate gate(&count, &wakes);
  CHECK_FALSE(gate.take_resumed());
  gate.resumed();  // la veille annulée : couvercle rouvert, un programme l'a empêchée
  CHECK(wakes.load() == 1);
  CHECK(gate.take_resumed());
  CHECK_FALSE(gate.take_resumed());
}

}  // namespace ajp::app::play
