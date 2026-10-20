// Mémoire de lecture : position retenue et rendue, oubliée près du début et de la fin, vidéo
// remplacée, plusieurs vidéos, fichier abîmé.
#include "app/play/resume.h"

#include <filesystem>
#include <fstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

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
    const std::filesystem::path path = dir / name;
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
  REQUIRE(remember_position(place.store, a, kEpisode - kMinute, kEpisode));  // dans les 5 % de la fin
  CHECK_FALSE(resume_position(place.store, a));
  // Durée inconnue : seul le début compte.
  REQUIRE(remember_position(place.store, a, 50 * kMinute, 0));
  CHECK(resume_position(place.store, a) == 50 * kMinute);
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
