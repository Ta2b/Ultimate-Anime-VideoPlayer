#include "app/play/pumped_task.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace ajp;
using namespace ajp::app::play;

TEST_CASE("run_pumped_for : une lecture qui ne répond plus rend la main au bout du délai (P2.15)",
          "[app][pumped]") {
  const auto start = std::chrono::steady_clock::now();
  const auto slow = run_pumped_for<int>("lent", std::chrono::milliseconds(200), false, [] {
    std::this_thread::sleep_for(std::chrono::seconds(2));  // un partage réseau muet
    return 1;
  });
  REQUIRE_FALSE(slow);
  CHECK(slow.error().code == RefusalCode::IoFailed);
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
  const auto quick = run_pumped_for<int>("rapide", std::chrono::seconds(5), false, [] { return 7; });
  REQUIRE(quick);
  CHECK(*quick == 7);
}

TEST_CASE("run_pumped_for : les sondes laissées à leur sort sont gardées, bornées, puis attendues (A30-05)",
          "[app][pumped]") {
  REQUIRE(settle_abandoned_threads(std::chrono::seconds(5)));  // celle du cas précédent a fini
  std::atomic<bool> release = false;
  const auto blocked = [&release] {
    while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return 1;
  };
  for (std::size_t i = 0; i < kMaxAbandonedThreads; ++i)
    CHECK_FALSE(run_pumped_for<int>("muet", std::chrono::milliseconds(20), false, blocked));
  CHECK(abandoned_running() == kMaxAbandonedThreads);
  // Une de plus : refusée tout de suite, sans thread ni descripteur de plus (A33-07).
  const auto start = std::chrono::steady_clock::now();
  const auto refused = run_pumped_for<int>("muet", std::chrono::seconds(5), false, blocked);
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::IoFailed);
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds(1));
  CHECK_FALSE(settle_abandoned_threads(std::chrono::milliseconds(50)));  // la sortie ne les attend pas
  release = true;
  CHECK(settle_abandoned_threads(std::chrono::seconds(5)));  // rendus : joints
  CHECK(abandoned_running() == 0);
}

TEST_CASE("C30-02, C33-05 : un fil de fond bloqué ne fait pas refuser la sonde d'un fichier",
          "[app][pumped]") {
  REQUIRE(settle_abandoned_threads(std::chrono::seconds(5)));
  std::atomic<bool> release = false;
  // Sous-titres en avance et pré-analyse bloqués sur un NAS muet : abandonnés comme fils de fond.
  for (int i = 0; i < 3; ++i) {
    std::jthread stuck([&release] {
      while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    });
    abandon_thread(std::move(stuck), [&release] { return release.load(); });
  }
  CHECK(abandoned_running(Abandoned::Background) == 3);
  CHECK(abandoned_running(Abandoned::Probe) == 0);
  // La vidéo locale suivante se sonde (avant : « 3 essais précédents ne répondent toujours pas »).
  const auto probe = run_pumped_for<int>("probe", std::chrono::seconds(5), false, [] { return 4; });
  REQUIRE(probe);
  CHECK(*probe == 4);
  release = true;
  CHECK(settle_abandoned_threads(std::chrono::seconds(5)));
}

TEST_CASE("C18-01 : une sonde lancée tôt s'attend plus tard, son délai compté depuis le lancement",
          "[app][pumped]") {
  auto task = PumpedTask<int>::start("probe", true, [] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return 9;
  });
  REQUIRE(task);
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // l'ouverture de l'écran
  const auto start = std::chrono::steady_clock::now();
  const auto got = task->wait(std::chrono::seconds(5), false, false);
  REQUIRE(got);
  CHECK(*got == 9);
  CHECK(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(500));
  // Jamais attendue : laissée à son sort comme une sonde, son résultat détruit sur son propre thread.
  {
    auto dropped = PumpedTask<std::vector<int>>::start("probe", true, [] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return std::vector<int>(1024, 1);
    });
    REQUIRE(dropped);
  }
  CHECK(settle_abandoned_threads(std::chrono::seconds(5)));
}

TEST_CASE("C30-01 : la fenêtre fermée pendant un travail sans délai attend sa fin", "[app][pumped]") {
  std::atomic<bool> done = false;
  std::jthread work([&done] {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    done = true;
  });
  const auto start = std::chrono::steady_clock::now();
  finish_after_close("engine-load", [&done] { return done.load(); });
  CHECK(done.load());  // le travail a fini avant que les objets qu'il lit ne meurent
  CHECK(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(150));
}
