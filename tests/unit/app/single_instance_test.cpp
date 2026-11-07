#include "app/play/single_instance.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <catch2/catch_test_macros.hpp>

using namespace ajp::app::play;

TEST_CASE("Instance unique : un second lancement passe sa vidéo au premier, qui la reçoit comme glissée",
          "[app][instance]") {
  REQUIRE(SDL_Init(SDL_INIT_EVENTS));
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-instance-test";
  std::filesystem::create_directories(dir);
  const std::string key = instance_key(dir);
  auto first = claim_instance(dir, key, {});
  REQUIRE(first.listener != nullptr);
  CHECK_FALSE(first.forwarded);
  const auto second = claim_instance(dir, key, {"/vidéos/épisode 1.mkv"});
  CHECK(second.forwarded);
  std::optional<std::string> received;
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!received && std::chrono::steady_clock::now() < until) {
    SDL_Event event;
    if (SDL_WaitEventTimeout(&event, 50) && event.type == SDL_EVENT_DROP_FILE &&
        std::string_view(event.drop.source) == kForwardedDrop)
      received = event.drop.data;
  }
  CHECK(received == "/vidéos/épisode 1.mkv");
  first.listener.reset();
  SDL_Quit();
  const std::array<std::string_view, 1> plain{"film.mkv"};
  const std::array<std::string_view, 2> scripted{"film.mkv", "--keys=1:p"};
  CHECK(shares_instance(plain));
  CHECK_FALSE(shares_instance(scripted));
}
