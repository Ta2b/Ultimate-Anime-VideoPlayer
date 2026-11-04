// Méga plan A5 : les mesures d'émetteurs v2 (render/emitters.h) sur les quatre cas du plan : halo,
// étincelle, chemise blanche, visage ; plus le néon, une lumière colorée.
#include "render/emitters.h"

#include <functional>

#include <catch2/catch_test_macros.hpp>

namespace ajp::render {
namespace {

constexpr int kW = 256;
constexpr int kH = 128;

EmitterImage image_of(const std::function<float(int, int)>& nits,
                      const std::function<float(int, int)>& chroma) {
  EmitterImage img{.width = kW, .height = kH};
  img.nits.resize(static_cast<std::size_t>(kW) * kH);
  img.chroma.resize(img.nits.size());
  for (int y = 0; y < kH; ++y) {
    for (int x = 0; x < kW; ++x) {
      img.nits[img.at(x, y)] = nits(x, y);
      img.chroma[img.at(x, y)] = chroma(x, y);
    }
  }
  return img;
}

bool inside(int x, int y, int x0, int y0, int x1, int y1) {
  return x >= x0 && x < x1 && y >= y0 && y < y1;
}

}  // namespace

TEST_CASE("émetteurs v2 : un halo sur un fond sombre est une lumière, le fond non", "[render][emitters]") {
  const auto img = image_of([](int x, int y) { return inside(x, y, 112, 48, 144, 80) ? 195.0F : 3.0F; },
                            [](int, int) { return 0.05F; });
  const auto map = emitter_map_v2(img);
  REQUIRE(map.size() == img.nits.size());
  CHECK(map[img.at(128, 64)] > 0.9F);
  CHECK(map[img.at(16, 16)] < 0.01F);
}

TEST_CASE("émetteurs v2 : une étincelle d'un pixel est une lumière", "[render][emitters]") {
  const auto img = image_of([](int x, int y) { return x == 100 && y == 60 ? 203.0F : 40.0F; },
                            [](int, int) { return 0.1F; });
  const auto map = emitter_map_v2(img);
  CHECK(map[img.at(100, 60)] > 0.9F);
  CHECK(map[img.at(101, 60)] < 0.05F);
}

TEST_CASE("émetteurs v2 : une chemise blanche qui remplit la moitié de l'image n'est pas une lumière",
          "[render][emitters]") {
  const auto img =
      image_of([](int x, int) { return x < kW / 2 ? 190.0F : 30.0F; }, [](int, int) { return 0.03F; });
  const auto map = emitter_map_v2(img);
  // Au cœur de la chemise, loin du bord : son entourage est aussi clair qu'elle.
  CHECK(map[img.at(40, 64)] < 0.05F);
}

TEST_CASE("émetteurs v2 : un visage clair sur un décor moyen n'est pas une lumière", "[render][emitters]") {
  // Une peau à 60 % du blanc, peu saturée, sur un décor à 30 %.
  const auto img = image_of([](int x, int y) { return inside(x, y, 96, 32, 160, 96) ? 122.0F : 61.0F; },
                            [](int x, int y) { return inside(x, y, 96, 32, 160, 96) ? 0.30F : 0.20F; });
  const auto map = emitter_map_v2(img);
  CHECK(map[img.at(128, 64)] < 0.1F);
}

TEST_CASE("émetteurs v2 : un néon saturé moins clair qu'un blanc est une lumière", "[render][emitters]") {
  const auto img = image_of([](int x, int y) { return inside(x, y, 112, 56, 144, 72) ? 110.0F : 4.0F; },
                            [](int x, int y) { return inside(x, y, 112, 56, 144, 72) ? 0.9F : 0.1F; });
  const auto map = emitter_map_v2(img);
  CHECK(map[img.at(128, 64)] > 0.8F);
  // Le même niveau sans couleur (un mur gris clair) : pas une lumière.
  const auto grey = image_of([](int x, int y) { return inside(x, y, 112, 56, 144, 72) ? 110.0F : 4.0F; },
                             [](int, int) { return 0.05F; });
  CHECK(emitter_map_v2(grey)[img.at(128, 64)] < 0.05F);
}

TEST_CASE("émetteurs v2 : une image mal formée ne rend rien", "[render][emitters]") {
  EmitterImage bad{.width = 4, .height = 4};
  bad.nits.resize(16);
  CHECK(emitter_map_v2(bad).empty());
}

}  // namespace ajp::render
