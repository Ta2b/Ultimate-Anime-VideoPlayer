// Le rythme de l'accueil au repos (app/play/home_idle.h, A24-17).
#include "app/play/home_idle.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_init.h>
#include <catch2/catch_test_macros.hpp>

#include "app/play/home_screen.h"

namespace ajp::app::play {
namespace {

// Les images d'une minute d'accueil, un événement à chaque instant de `events` (µs, croissants).
int frames_in_a_minute(std::initializer_list<std::int64_t> events, bool busy) {
  int frames = 0;
  std::int64_t last_event = 0;
  const auto* next = events.begin();
  for (std::int64_t now = 0; now < 60'000'000;) {
    ++frames;
    std::int64_t wake = now + home_wait_ms(now, last_event, busy) * 1'000LL;
    if (next != events.end() && *next < wake) {  // un événement réveille l'attente
      wake = *next;
      last_event = *next++;
    }
    now = wake;
  }
  return frames;
}

}  // namespace

TEST_CASE("Accueil : au repos, une image toutes les 100 ms au lieu de 16 (A24-17)", "[app][home]") {
  // Avant : 3 750 images par minute, quoi qu'il arrive (60 000 / 16).
  // Après : deux secondes au rythme plein, puis dix images par seconde.
  CHECK(frames_in_a_minute({}, false) == 125 + 580);
  // Quelque chose avance sans événement (message, boîte de fichiers, séance scriptée) : rythme plein.
  CHECK(frames_in_a_minute({}, true) == 3'750);
}

TEST_CASE("Accueil : un événement rend le rythme plein aussitôt", "[app][home]") {
  CHECK(home_wait_ms(10'000'000, 0, false) == kHomeIdleMs);
  CHECK(home_wait_ms(10'000'000, 9'999'000, false) == kHomeFrameMs);
  CHECK(home_wait_ms(10'000'000, 8'000'000, false) == kHomeIdleMs);
  CHECK(home_wait_ms(10'000'000, 8'000'001, false) == kHomeFrameMs);
  // Un mouvement de souris à 30 s : l'attente de 100 ms est coupée, puis deux secondes de rythme plein.
  const int moved = frames_in_a_minute({30'000'000}, false);
  CHECK(moved > 705);
  CHECK(moved < 705 + 130);
}

TEST_CASE("Accueil : un fichier glissé se trie par son extension, sans lire le disque (A24-15)",
          "[app][home]") {
  CHECK(drop_kind("/nulle/part/Épisode 01.MKV") == DropKind::Video);
  CHECK(drop_kind("episode.mp4") == DropKind::Video);
  CHECK(drop_kind("episode.fr.ASS") == DropKind::Subtitles);
  CHECK(drop_kind("episode.srt") == DropKind::Subtitles);
  CHECK(drop_kind("affiche.png") == DropKind::Other);
  CHECK(drop_kind("dossier") == DropKind::Other);
  // Un autre événement qu'un fichier glissé : rien, et la fenêtre n'est pas touchée.
  SDL_Event other{};
  other.type = SDL_EVENT_KEY_DOWN;
  CHECK_FALSE(dropped(other));
  raise_for_forwarded(other);
}

TEST_CASE(
    "Attente : une vidéo reçue pendant l'assistant, l'installation ou la calibration n'est pas perdue "
    "(B29-04)",
    "[app][home]") {
  REQUIRE(SDL_Init(SDL_INIT_EVENTS));
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-deferred-drops";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string video = (dir / "ep02.mkv").string();
  const std::string text = (dir / "notes.txt").string();
  std::ofstream(video) << "v";
  std::ofstream(text) << "t";
  const auto drop = [](const std::string& path) {
    SDL_Event event{};
    event.type = SDL_EVENT_DROP_FILE;
    event.drop.data = path.c_str();
    return event;
  };
  SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
  {
    DeferredDrops drops;
    SDL_Event key{};
    key.type = SDL_EVENT_KEY_DOWN;
    CHECK_FALSE(drops.keep(key));   // un autre événement suit son chemin
    CHECK(drops.keep(drop(text)));  // un dépôt qui n'est pas une vidéo est écarté, sans rien garder
    CHECK(drops.files().empty());
    CHECK(drops.keep(drop(video)));
    REQUIRE(drops.files().size() == 1);
  }  // la fin de l'attente rend la vidéo à la file
  std::optional<std::string> back;
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!back && std::chrono::steady_clock::now() < until) {
    SDL_Event event;
    if (SDL_WaitEventTimeout(&event, 50) && event.type == SDL_EVENT_DROP_FILE && event.drop.data != nullptr)
      back = event.drop.data;
  }
  CHECK(back == video);
  SDL_Quit();
  std::filesystem::remove_all(dir);
}

}  // namespace ajp::app::play
