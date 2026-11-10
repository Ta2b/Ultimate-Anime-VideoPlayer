// Le garde de la fermeture (app/play/close_watch.h, A10-04).
#include "app/play/close_watch.h"

#include <atomic>
#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {

using namespace std::chrono_literals;

TEST_CASE("Garde de fermeture : une attente qui finit à temps ne déclenche rien (A10-04)", "[app][close]") {
  std::atomic<int> stuck{0};
  const auto since = std::chrono::steady_clock::now();
  {
    const CloseWatch watch(30s, [&] { ++stuck; });
    std::this_thread::sleep_for(5ms);  // le démontage du GPU, rapide
  }
  CHECK(stuck == 0);
  CHECK(std::chrono::steady_clock::now() - since < 5s);  // le garde ne retient pas la fermeture
}

TEST_CASE("Garde de fermeture : un GPU qui ne répond pas déclenche la sortie, une seule fois (A10-04)",
          "[app][close]") {
  std::atomic<int> stuck{0};
  {
    const CloseWatch watch(20ms, [&] { ++stuck; });
    // Le GPU bloqué : avant, cette attente n'avait pas de fin et rien ne se passait.
    for (int k = 0; k < 400 && stuck == 0; ++k) std::this_thread::sleep_for(5ms);
    CHECK(stuck == 1);
    std::this_thread::sleep_for(60ms);
  }
  CHECK(stuck == 1);
}

}  // namespace ajp::app::play
