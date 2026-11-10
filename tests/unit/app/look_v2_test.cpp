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

using namespace ajp::app::play;

TEST_CASE("le look est la pile HDR anime, une LUT par emplacement", "[app][look]") {
  PlaySettings s;
  const std::vector<LookLayer> pile = look_slots(s);
  const std::vector<LookLayer> reference{
      {std::string(kAnimeLookBase), 1.0},      {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookContrast), 1.0},  {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookHighlight), 1.0}, {std::string(kAnimeLookHighlight), 0.5},
      {std::string(kAnimeLookSkin), 1.0},      {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookContrast), 1.0},  {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookContrast), 1.0}};
  CHECK(pile == reference);
  CHECK(ajp::render::look_is_pq(pile.front().key, ""));
  // Les curseurs au maximum tiennent dans les emplacements du grade ; au minimum, la base et « skin fix ».
  s.look_contrast = 100;
  s.look_highlight = 100;
  CHECK(look_slots(s).size() == 19);
  CHECK(look_slots(s).size() <= ajp::render::kLookSlots);
  s.look_contrast = -100;
  s.look_highlight = -100;
  CHECK(look_slots(s).size() == 2);
  // Un curseur qui ne change que des intensités garde les mêmes fichiers : rien n'est relu.
  PlaySettings a;
  PlaySettings b;
  a.look_highlight = 10;
  b.look_highlight = 20;
  CHECK(same_look_files(a, b));
  b.look_highlight = 100;
  CHECK_FALSE(same_look_files(a, b));
}

TEST_CASE("look HDR anime : les intensités en LUT entières puis une partielle", "[app][look]") {
  CHECK(anime_look_mixes(3.0) == std::vector<double>{1.0, 1.0, 1.0});
  CHECK(anime_look_mixes(1.5) == std::vector<double>{1.0, 0.5});
  CHECK(anime_look_mixes(0.0).empty());
  const auto c = anime_look_mixes(7.0 * 1.5 * 4.0 / 7.0);  // contraste +50, la part d'après : 6
  CHECK(c.size() == 6);
}

TEST_CASE("look HDR anime : chaque fichier de la pile se lit", "[app][look]") {
  const std::filesystem::path luts = std::string(AJP_APP_SOURCE_DIR) + "/../../data/luts";
  for (const LookLayer& layer : look_slots(PlaySettings{})) {
    const auto lut = load_look(layer.key, luts);
    INFO(layer.key);
    REQUIRE(lut);
    CHECK(lut->size >= ajp::render::kLutMinSize);
  }
}
