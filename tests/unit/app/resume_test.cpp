// Mémoire de lecture : position retenue et rendue, oubliée près du début et de la fin, vidéo
// remplacée, plusieurs vidéos, fichier abîmé.
#include "app/play/resume.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/utf8_path.h"
#include "prepare/key_lock.h"

using namespace ajp::app::play;

namespace {

constexpr std::int64_t kMinute = 60'000'000;
constexpr std::int64_t kEpisode = 24 * kMinute;

struct Place {
  std::filesystem::path dir;
  std::filesystem::path store;
  explicit Place(const std::string& name)
      : dir(std::filesystem::temp_directory_path() / ("ajp_resume_" + name)), store(dir / "resume.tsv") {
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
  }
  Place(const Place&) = delete;
  Place& operator=(const Place&) = delete;
  Place(Place&&) = delete;
  Place& operator=(Place&&) = delete;
  ~Place() { std::filesystem::remove_all(dir); }

  std::filesystem::path video(const std::string& name, std::size_t bytes = 16) const {
    const std::filesystem::path path = dir / ajp::core::path_from_utf8(name);
    std::ofstream(path) << std::string(bytes, 'x');
    return path;
  }
};

}  // namespace

TEST_CASE("reprise : la position retenue est rendue", "[app][resume]") {
  const Place place("kept");
  const auto a = place.video("a.mkv");
  CHECK_FALSE(resume_position(place.store, a));
  REQUIRE(remember_position(place.store, a, 7 * kMinute, kEpisode));
  CHECK(resume_position(place.store, a) == 7 * kMinute);
  // Une autre façon d'écrire le même chemin désigne la même vidéo.
  CHECK(resume_position(place.store, place.dir / "." / "a.mkv") == 7 * kMinute);
  REQUIRE(remember_position(place.store, a, 9 * kMinute, kEpisode));
  CHECK(resume_position(place.store, a) == 9 * kMinute);
}

TEST_CASE("reprise : oubliée près du début et vers la fin", "[app][resume]") {
  const Place place("ends");
  const auto a = place.video("a.mkv");
  REQUIRE(remember_position(place.store, a, 7 * kMinute, kEpisode));
  REQUIRE(remember_position(place.store, a, 10'000'000, kEpisode));  // à peine commencée
  CHECK_FALSE(resume_position(place.store, a));
  REQUIRE(remember_position(place.store, a, 7 * kMinute, kEpisode));
  REQUIRE(remember_position(place.store, a, kEpisode - 2 * kMinute, kEpisode));  // pendant l'ending (P3.83)
  CHECK_FALSE(resume_position(place.store, a));
  // Durée inconnue : seul le début compte.
  REQUIRE(remember_position(place.store, a, 50 * kMinute, 0));
  CHECK(resume_position(place.store, a) == 50 * kMinute);
}

TEST_CASE("reprise : une vidéo déplacée se retrouve à son nom et sa taille (P3.83)", "[app][resume]") {
  const Place place("moved");
  const auto a = place.video("a.mkv");
  REQUIRE(remember_position(place.store, a, 7 * kMinute, kEpisode));
  std::filesystem::create_directories(place.dir / "range");
  std::filesystem::rename(a, place.dir / "range" / "a.mkv");
  CHECK(resume_position(place.store, place.dir / "range" / "a.mkv") == 7 * kMinute);
}

TEST_CASE("reprise : une vidéo remplacée repart du début", "[app][resume]") {
  const Place place("replaced");
  const auto a = place.video("a.mkv", 16);
  REQUIRE(remember_position(place.store, a, 7 * kMinute, kEpisode));
  place.video("a.mkv", 32);
  CHECK_FALSE(resume_position(place.store, a));
}

TEST_CASE("reprise : plusieurs vidéos, lignes abîmées ignorées", "[app][resume]") {
  const Place place("many");
  const auto a = place.video("a.mkv");
  const auto b = place.video("b.mkv");
  REQUIRE(remember_position(place.store, a, 3 * kMinute, kEpisode));
  REQUIRE(remember_position(place.store, b, 5 * kMinute, kEpisode));
  std::ofstream(place.store, std::ios::app) << "pas un nombre\tnon plus\t/x\n\n12\n";
  CHECK(resume_position(place.store, a) == 3 * kMinute);
  CHECK(resume_position(place.store, b) == 5 * kMinute);
  REQUIRE(remember_position(place.store, a, 4 * kMinute, kEpisode));
  CHECK(resume_position(place.store, a) == 4 * kMinute);
  CHECK(resume_position(place.store, b) == 5 * kMinute);
}

TEST_CASE("reprise : un nom non ASCII se retient et se retrouve (A13-11, A13-13)", "[app][resume]") {
  const Place place("utf8");
  const auto a = place.video("進撃の巨人 01.mkv");
  const auto b = place.video("Épisode 02.mkv");
  REQUIRE(remember_position(place.store, a, 7 * kMinute, kEpisode));
  REQUIRE(remember_position(place.store, b, 8 * kMinute, kEpisode));
  CHECK(resume_position(place.store, a) == 7 * kMinute);
  CHECK(resume_position(place.store, b) == 8 * kMinute);
}

TEST_CASE("reprise : au-delà de 500 vidéos, la plus ancienne est oubliée (A13-13)", "[app][resume]") {
  const Place place("bound");
  const auto first = place.video("first.mkv");
  REQUIRE(remember_position(place.store, first, 7 * kMinute, kEpisode));
  for (std::size_t i = 0; i < kResumeEntries; ++i) {
    const auto v = place.video("v" + std::to_string(i) + ".mkv");
    REQUIRE(remember_position(place.store, v, 7 * kMinute, kEpisode));
  }
  CHECK_FALSE(resume_position(place.store, first));
  CHECK(resume_position(place.store, place.dir / "v0.mkv") == 7 * kMinute);
  std::ifstream in(place.store);
  std::size_t lines = 0;
  for (std::string line; std::getline(in, line);) ++lines;
  CHECK(lines == kResumeEntries);
}

TEST_CASE("reprise : deux lecteurs qui écrivent ensemble gardent chacun leur vidéo (A13-10)",
          "[app][resume]") {
  const Place place("writers");
  std::vector<std::filesystem::path> videos;
  for (int i = 0; i < 8; ++i) videos.push_back(place.video("w" + std::to_string(i) + ".mkv"));
  // Deux lecteurs, chacun ses quatre vidéos : sans verrou, l'un réécrivait le fichier relu avant l'écriture
  // de l'autre, et une position se perdait.
  std::atomic<int> refused = 0;  // les assertions de Catch2 restent sur ce thread
  const auto writer = [&](std::size_t first) {
    for (int round = 0; round < 20; ++round)
      for (std::size_t i = first; i < first + 4; ++i)
        if (!remember_position(place.store, videos[i], (7 + round) * kMinute, 4 * kEpisode)) ++refused;
  };
  {
    const std::jthread one(writer, 0);
    const std::jthread two(writer, 4);
  }
  CHECK(refused == 0);
  for (const auto& video : videos) CHECK(resume_position(place.store, video) == 26 * kMinute);
}

TEST_CASE("reprise : un verrou tenu par un autre lecteur fait refuser sans écrire à l'aveugle (A13-10)",
          "[app][resume]") {
  const Place place("held");
  const auto first = place.video("h0.mkv");
  const auto second = place.video("h1.mkv");
  REQUIRE(remember_position(place.store, first, 7 * kMinute, 4 * kEpisode));
  std::filesystem::path lock_file = place.store;
  lock_file += ".lock";
  {
    // Un autre lecteur (un autre processus) tient le verrou au-delà du délai : écrire sans lui perdrait ce
    // qu'il écrit, la position est refusée et le fichier reste tel quel.
    auto held = ajp::prepare::KeyLock::acquire(lock_file);
    REQUIRE(held);
    const auto refused = remember_position(place.store, second, 9 * kMinute, 4 * kEpisode);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == ajp::RefusalCode::CacheBusy);
  }
  CHECK(resume_position(place.store, first) == 7 * kMinute);
  CHECK_FALSE(resume_position(place.store, second));
  REQUIRE(remember_position(place.store, second, 9 * kMinute, 4 * kEpisode));  // verrou rendu
  CHECK(resume_position(place.store, second) == 9 * kMinute);
}
