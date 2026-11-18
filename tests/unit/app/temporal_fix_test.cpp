// La stabilisation temporelle hors GPU (temporal_fix.h, temporal_window.h) : le choix des sept images, le
// reflet aux bords comme vs_temporalfix, l'oubli d'une époque, et sa clé de play.toml.
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/settings.h"
#include "app/play/temporal_fix_model.h"
#include "app/play/temporal_window.h"
#include "stage_rig.h"

namespace ajp::app::play {
namespace {

DecodedFrame frame(std::uint32_t id, std::uint32_t epoch = 0) {
  return test::host_frame(id, 16, 16, static_cast<std::int64_t>(id) * 1000, epoch);
}

// Les images (leurs id) de la fenêtre autour de l'image center, dans l'ordre du temps.
std::vector<std::uint32_t> ids(const TemporalWindow& w, std::size_t center, bool* mirrored = nullptr) {
  std::vector<std::uint32_t> out;
  for (const std::size_t i : w.window(center, mirrored)) out.push_back(w.raw(i).id.value);
  return out;
}

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST_CASE("Stabilisation : sept images autour de l'image, dans l'ordre du temps", "[app][temporal]") {
  TemporalWindow w(10);
  std::array<std::size_t, 9> at{};
  for (std::uint32_t id = 1; id <= 9; ++id) at[id - 1] = w.claim(frame(id), {}, false);
  bool mirrored = true;
  CHECK(ids(w, at[4], &mirrored) == std::vector<std::uint32_t>{2, 3, 4, 5, 6, 7, 8});
  CHECK_FALSE(mirrored);
}

TEST_CASE("Stabilisation : la fenêtre se reflète au début et à la fin, comme vs_temporalfix",
          "[app][temporal]") {
  TemporalWindow w(10);
  std::array<std::size_t, 5> at{};
  for (std::uint32_t id = 1; id <= 5; ++id) at[id - 1] = w.claim(frame(id), {}, false);
  bool mirrored = false;
  CHECK(ids(w, at[0], &mirrored) == std::vector<std::uint32_t>{4, 3, 2, 1, 2, 3, 4});
  CHECK(mirrored);
  CHECK(ids(w, at[1]) == std::vector<std::uint32_t>{5, 4, 1, 2, 3, 4, 5});
  CHECK(ids(w, at[4]) == std::vector<std::uint32_t>{2, 3, 4, 5, 4, 3, 2});
  // Une seule image : elle-même partout.
  TemporalWindow alone(4);
  const std::size_t only = alone.claim(frame(7), {}, false);
  CHECK(ids(alone, only) == std::vector<std::uint32_t>{7, 7, 7, 7, 7, 7, 7});
  // Deux images : la voisine qui manque des deux côtés reprend la plus proche déjà prise.
  TemporalWindow two(4);
  const std::size_t first = two.claim(frame(1), {}, false);
  two.claim(frame(2), {}, false);
  CHECK(ids(two, first) == std::vector<std::uint32_t>{2, 2, 2, 1, 2, 2, 2});
}

TEST_CASE("Stabilisation : un seek oublie l'époque d'avant, le plus petit pts part d'abord",
          "[app][temporal]") {
  TemporalWindow w(3);
  w.claim(frame(1), {}, false);
  w.claim(frame(2), {}, false);
  w.claim(frame(3), {}, false);
  const std::size_t fourth = w.claim(frame(4), {}, false);  // plein : l'image 1 cède sa place
  CHECK_FALSE(w.find(frame(1), {}));
  CHECK(w.find(frame(2), {}));
  CHECK(*w.find(frame(4), {}) == fourth);
  const std::size_t after_seek = w.claim(frame(20, 1), {}, false);
  CHECK(ids(w, after_seek) == std::vector<std::uint32_t>{20, 20, 20, 20, 20, 20, 20});
  CHECK_FALSE(w.find(frame(2), {}));
  // Une autre réduction n'est pas la même image : elle se réingère, et l'ancienne cède sa place.
  const gpu::Reduction other{.kind = gpu::ReduceKind::Cubic};
  CHECK_FALSE(w.find(frame(20, 1), other));
}

TEST_CASE("play.toml : la stabilisation temporelle, absente d'un ancien fichier, s'enregistre",
          "[app][settings][temporal]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-temporal.toml";
  REQUIRE(save_settings(path, PlaySettings{}));
  std::string text = read_text(path);
  const std::string key = "temporal_fix = 'off'\n";
  const std::size_t line = text.find(key);
  REQUIRE(line != std::string::npos);
  text.erase(line, key.size());
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
  }
  auto old = load_settings(path);
  REQUIRE(old);
  CHECK(old->temporal_fix == TemporalFix::Off);  // absente : éteinte
  old->temporal_fix = TemporalFix::Strong;
  REQUIRE(save_settings(path, *old));
  CHECK(read_text(path).find("temporal_fix = 'strong'") != std::string::npos);
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(reread->temporal_fix == TemporalFix::Strong);
  CHECK_FALSE(reread->temporal_fix_full);  // absente : à demi-taille
  PlaySettings full = *reread;
  full.temporal_fix_full = true;
  REQUIRE(save_settings(path, full));
  const auto full_read = load_settings(path);
  REQUIRE(full_read);
  CHECK(full_read->temporal_fix_full);
  std::filesystem::remove(path);
}

TEST_CASE("Stabilisation : la taille de travail doit se diviser en deux, puis encore en deux à demi-taille",
          "[app][temporal]") {
  CHECK(temporal_fix_fits(1920, 1080, true));
  CHECK(temporal_fix_fits(1280, 720, true));
  CHECK_FALSE(temporal_fix_fits(1918, 1078, true));  // 959 × 539 à demi-taille
  CHECK(temporal_fix_fits(1918, 1078, false));
  CHECK_FALSE(temporal_fix_fits(1919, 1080, false));
  CHECK_FALSE(temporal_fix_fits(0, 1080, false));
}

TEST_CASE("Stabilisation : l'allumer, l'éteindre ou changer sa taille recharge la vidéo", "[app][temporal]") {
  PlaySettings running;
  PlaySettings on = running;
  on.temporal_fix = TemporalFix::Normal;
  CHECK(change_between(running, on, true).cost == ChangeCost::Reopen);
  CHECK(change_between(on, running, true).cost == ChangeCost::Reopen);
  PlaySettings full = on;
  full.temporal_fix_full = true;
  CHECK(change_between(on, full, true).cost == ChangeCost::Reopen);
  // Éteinte, sa taille ne recharge rien : le réglage attend qu'on l'allume.
  PlaySettings off_full = running;
  off_full.temporal_fix_full = true;
  CHECK(change_between(running, off_full, true).cost != ChangeCost::Reopen);
}

}  // namespace ajp::app::play
