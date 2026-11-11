// Anime4K sans GPU (render/anime4k.h) : le mode choisi d'après la source, et les shaders embarqués.
#include "render/anime4k.h"

#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

namespace ajp::render {
namespace {

TEST_CASE("Anime4K : le mode suit la source et sa réduction", "[render][anime4k]") {
  // Gardée telle quelle : A pour une 1080p (souvent agrandie d'une 720p, floue) et les SD, B pour une 720p.
  CHECK(anime4k_mode(1080, 0) == Anime4kMode::A);
  CHECK(anime4k_mode(1080, 1080) == Anime4kMode::A);
  CHECK(anime4k_mode(2160, 0) == Anime4kMode::A);
  CHECK(anime4k_mode(720, 720) == Anime4kMode::B);
  CHECK(anime4k_mode(720, 1080) == Anime4kMode::B);  // une hauteur de travail plus haute ne réduit rien
  CHECK(anime4k_mode(480, 0) == Anime4kMode::A);
  CHECK(anime4k_mode(576, 720) == Anime4kMode::A);
  // Réduite : 1080p vers 720p (1,5) en B, vers 540p ou 480p (2 et plus) en C.
  CHECK(anime4k_mode(1080, 720) == Anime4kMode::B);
  CHECK(anime4k_mode(1080, 900) == Anime4kMode::B);
  CHECK(anime4k_mode(1080, 540) == Anime4kMode::C);
  CHECK(anime4k_mode(1080, 480) == Anime4kMode::C);
  CHECK(anime4k_mode(2160, 720) == Anime4kMode::C);
  CHECK(anime4k_mode(720, 480) == Anime4kMode::B);
  // Inconnue : A.
  CHECK(anime4k_mode(0, 540) == Anime4kMode::A);
}

TEST_CASE("Anime4K : les shaders embarqués sont ceux de l'amont, dans l'ordre de mpv", "[render][anime4k]") {
  for (const Anime4kSize size : {Anime4kSize::Fast, Anime4kSize::Quality}) {
    for (const Anime4kMode mode : {Anime4kMode::A, Anime4kMode::B, Anime4kMode::C}) {
      CAPTURE(name(size), name(mode));
      const auto texts = anime4k_texts({.size = size, .mode = mode});
      REQUIRE(texts.size() == (mode == Anime4kMode::C ? 2U : 3U));
      for (const std::string_view text : texts) {
        CHECK(text.starts_with("// MIT License"));  // la licence de l'amont, embarquée avec chaque shader
        CHECK(text.find('\r') == std::string_view::npos);
      }
      CHECK(texts.front().contains("//!DESC Anime4K-v4.0-De-Ring-Compute-Statistics"));
      const std::string_view net = name(size);
      const std::string_view last = texts.back();
      CHECK(last.contains(std::string("-x2-(") + std::string(net) + ")"));
      if (mode == Anime4kMode::C) {
        CHECK(last.contains("Upscale-Denoise-CNN-x2"));
      } else {
        CHECK(texts[1].contains(mode == Anime4kMode::B ? "Restore-CNN-Soft-(" : "Restore-CNN-("));
        CHECK(texts[1].contains(std::string("(") + std::string(net) + ")"));
        CHECK(last.contains("Upscale-CNN-x2"));
      }
    }
  }
}

TEST_CASE("Anime4K : ses passes se reconnaissent à leur description", "[render][anime4k]") {
  CHECK(is_anime4k_pass("Anime4K-v3.2-Upscale-CNN-x2-(M)-Conv-4x3x3x8"));
  CHECK(is_anime4k_pass("(hook) Anime4K-v4.0-De-Ring-Clamp"));
  CHECK_FALSE(is_anime4k_pass("AnimeJaNai grade"));
  CHECK_FALSE(is_anime4k_pass(""));
}

}  // namespace
}  // namespace ajp::render
