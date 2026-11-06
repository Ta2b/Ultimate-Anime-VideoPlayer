// Le look du contrat v2 (chantier 8.4) : ver9-couleur.cube remplace la pile par défaut, et ses gris restent
// gris (la courbe est celle du grade, tools/look/split_hdr_lut.py).
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/look_catalog.h"
#include "app/play/settings.h"
#include "render/cube_lut.h"
#include "render/looks.h"

using namespace ajp::app::play;

TEST_CASE("le look est la recette HDR anime ; --no-looks : rien", "[app][look]") {
  PlaySettings s;
  REQUIRE(look_slots(s).size() == 1);
  CHECK(look_slots(s).front().key == anime_look_key(0, 0));
  s.look_contrast = -40;
  s.look_highlight = 25;
  CHECK(look_slots(s).front().key == anime_look_key(-40, 25));
  CHECK(ajp::render::look_is_pq(look_slots(s).front().key, ""));
  set_no_looks(true);
  CHECK(look_slots(s).empty());
  set_no_looks(false);
}

TEST_CASE("look HDR anime : les intensités en LUT entières puis une partielle", "[app][look]") {
  CHECK(anime_look_mixes(3.0) == std::vector<double>{1.0, 1.0, 1.0});
  CHECK(anime_look_mixes(1.5) == std::vector<double>{1.0, 0.5});
  CHECK(anime_look_mixes(0.0).empty());
  const auto c = anime_look_mixes(7.0 * 1.5 * 4.0 / 7.0);  // contraste +50, la part d'après : 6
  CHECK(c.size() == 6);
}

TEST_CASE("look HDR anime à 0 et 0 : la pile de Tatoub du 04/11", "[app][look]") {
  const std::filesystem::path luts = std::string(AJP_APP_SOURCE_DIR) + "/../../data/luts";
  const auto recipe = anime_look(0, 0, luts);
  REQUIRE(recipe);
  std::vector<ajp::render::CubeLut> files;
  const std::vector<std::pair<std::string, double>> stack{
      {std::string(kAnimeLookBase), 1.0},      {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookContrast), 1.0},  {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookHighlight), 1.0}, {std::string(kAnimeLookHighlight), 0.5},
      {std::string(kAnimeLookSkin), 1.0},      {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookContrast), 1.0},  {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookContrast), 1.0}};
  std::vector<double> mixes;
  for (const auto& [name, mix] : stack) {
    auto lut = load_look(name, luts);
    REQUIRE(lut);
    files.push_back(std::move(*lut));
    mixes.push_back(mix);
  }
  std::vector<const ajp::render::CubeLut*> pointers;
  for (const auto& f : files) pointers.push_back(&f);
  const ajp::render::CubeLut direct = ajp::render::compose_luts(pointers, mixes);
  REQUIRE(direct.rgb.size() == recipe->rgb.size());
  CHECK(direct.rgb == recipe->rgb);
}
