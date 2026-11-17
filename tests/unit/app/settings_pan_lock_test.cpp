// Le mouvement d'ensemble (PlaySettings::pan_lock, [interpolation] de play.toml) : absent d'un ancien
// fichier, il se lit éteint ; coché, il se relit.
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/settings.h"

namespace ajp::app::play {
namespace {

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}

}  // namespace

TEST_CASE("play.toml : mouvement d'ensemble, absent d'un ancien fichier, s'enregistre",
          "[app][settings][pan_lock]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-pan-lock.toml";
  REQUIRE(save_settings(path, PlaySettings{}));
  std::string text = read_text(path);
  const std::size_t line = text.find("pan_lock = false\n");
  REQUIRE(line != std::string::npos);
  text.erase(line, std::string("pan_lock = false\n").size());
  write_text(path, text);
  auto old = load_settings(path);
  REQUIRE(old);
  CHECK_FALSE(old->pan_lock);
  old->pan_lock = true;
  REQUIRE(save_settings(path, *old));
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(reread->pan_lock);
  std::filesystem::remove(path);
}

}  // namespace ajp::app::play
