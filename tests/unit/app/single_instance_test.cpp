#include "app/play/single_instance.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <catch2/catch_test_macros.hpp>

#ifndef _WIN32
#include <cstring>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

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

TEST_CASE("Instance unique : deux lancements simultanés, un seul prend la place (A19-13)",
          "[app][instance]") {
  REQUIRE(SDL_Init(SDL_INIT_EVENTS));
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-instance-race";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string key = instance_key(dir);
  auto a = std::async(std::launch::async, [&] { return claim_instance(dir, key, {"a.mkv"}); });
  auto b = std::async(std::launch::async, [&] { return claim_instance(dir, key, {"b.mkv"}); });
  InstanceClaim first = a.get();
  InstanceClaim second = b.get();
  const int listeners = (first.listener != nullptr ? 1 : 0) + (second.listener != nullptr ? 1 : 0);
  CHECK(listeners == 1);
  CHECK(first.forwarded != second.forwarded);
  first.listener.reset();
  second.listener.reset();
  SDL_Quit();
  std::filesystem::remove_all(dir);
}

#ifndef _WIN32
TEST_CASE("Instance unique : un socket resté d'un lecteur arrêté net est repris (A19-13)",
          "[app][instance]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-instance-stale";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  {
    // Un socket lié puis abandonné sans être retiré, comme celui d'un lecteur tué.
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    const std::string text = (dir / "instance.sock").string();
    REQUIRE(text.size() < sizeof(address.sun_path));
    std::memcpy(static_cast<char*>(address.sun_path), text.c_str(), text.size() + 1);
    REQUIRE(::bind(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0);
    ::close(fd);
  }
  const std::string key = instance_key(dir);
  auto claim = claim_instance(dir, key, {"film.mkv"});
  CHECK(claim.listener != nullptr);
  CHECK_FALSE(claim.forwarded);
  claim.listener.reset();
  std::filesystem::remove_all(dir);
}
#endif

TEST_CASE("C31-03 : un lecteur relancé ne passe pas sa vidéo à l'ancien qui tient encore le canal",
          "[app][instance]") {
  REQUIRE(SDL_Init(SDL_INIT_EVENTS));
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-instance-takeover";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string key = instance_key(dir);
  auto old = claim_instance(dir, key, {});
  REQUIRE(old.listener != nullptr);
  // L'ancien lecteur, figé sur le GPU, sort un peu après avoir relancé celui-ci.
  auto leaving = std::async(std::launch::async, [&old] {
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    old.listener.reset();
  });
  const auto start = std::chrono::steady_clock::now();
  auto relaunched = claim_instance(dir, key, {"/vidéos/épisode 1.mkv"}, /*take_over=*/true);
  leaving.get();
  CHECK_FALSE(relaunched.forwarded);  // avant : la vidéo partait vers l'ancien, qui mourait avec elle
  CHECK(relaunched.listener != nullptr);
  CHECK(std::chrono::steady_clock::now() - start < kTakeOverWait);
  relaunched.listener.reset();
  SDL_Quit();
}
