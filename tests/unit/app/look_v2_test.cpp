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

namespace {

// Les LUT appliquées, dans l'ordre : les emplacements à intensité 0 sont sautés par le shader.
std::vector<LookLayer> applied(const std::vector<LookLayer>& pile) {
  std::vector<LookLayer> out;
  for (const LookLayer& layer : pile)
    if (layer.mix > 0.0) out.push_back(layer);
  return out;
}

}  // namespace

TEST_CASE("le look est la pile HDR anime, une LUT par emplacement", "[app][look]") {
  PlaySettings s;
  const std::vector<LookLayer> reference{
      {std::string(kAnimeLookBase), 1.0},      {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookContrast), 1.0},  {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookHighlight), 1.0}, {std::string(kAnimeLookHighlight), 0.5},
      {std::string(kAnimeLookSkin), 1.0},      {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookContrast), 1.0},  {std::string(kAnimeLookContrast), 1.0},
      {std::string(kAnimeLookContrast), 1.0}};
  CHECK(applied(look_slots(s)) == reference);
  CHECK(ajp::render::look_is_pq(look_slots(s).front().key, ""));
  // Toujours 19 emplacements, dans les 20 du grade : aux curseurs à 100, tous servent ; au minimum, la base
  // et « skin fix » seuls.
  CHECK(look_slots(s).size() == 19);
  s.look_contrast = 100;
  s.look_highlight = 100;
  CHECK(applied(look_slots(s)).size() == 19);
  s.look_contrast = 50;
  s.look_highlight = 0;
  CHECK(applied(look_slots(s)).size() == 15);  // base, 4,5 + 1,5 + skin + 6 : 1 + 5 + 2 + 1 + 6
  s.look_contrast = -100;
  s.look_highlight = -100;
  const std::vector<LookLayer> bare{{std::string(kAnimeLookBase), 1.0}, {std::string(kAnimeLookSkin), 1.0}};
  CHECK(applied(look_slots(s)) == bare);
  CHECK(look_slots(s).size() == 19);
}

TEST_CASE("look HDR anime : un curseur ne change que des intensités, jamais les fichiers", "[app][look]") {
  // Bouger un curseur ne refait ni le hook ni ses textures (chain.cpp ne relit les LUT que si les fichiers
  // changent) : de -100 à 100, la pile garde les mêmes fichiers aux mêmes places.
  PlaySettings a;
  for (const int contrast : {-100, -30, 0, 45, 100}) {
    for (const int highlight : {-100, 0, 100}) {
      PlaySettings b;
      b.look_contrast = contrast;
      b.look_highlight = highlight;
      CAPTURE(contrast, highlight);
      CHECK(same_look_files(a, b));
    }
  }
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
