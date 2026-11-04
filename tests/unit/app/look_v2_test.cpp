// Le look du contrat v2 (chantier 8.4) : ver9-couleur.cube remplace la pile par défaut, et ses gris restent
// gris (la courbe est celle du grade, tools/look/split_hdr_lut.py).
#include <cmath>
#include <fstream>
#include <iterator>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "app/play/look_catalog.h"
#include "app/play/settings.h"
#include "render/cube_lut.h"

using namespace ajp::app::play;

TEST_CASE("contrat v2 : la pile par défaut devient ver9-couleur.cube, une pile choisie reste",
          "[app][look]") {
  const PlaySettings defaults;
  CHECK(look_slots(defaults).size() == 1 + defaults.look_layers.size());
  set_contract_v2_looks(true);
  const auto v2 = look_slots(defaults);
  REQUIRE(v2.size() == 1);
  CHECK(v2.front().key == kContractV2Look);
  CHECK(v2.front().mix == 1.0);
  PlaySettings chosen;
  chosen.look_layers.push_back({"HDR-contrast-Enhancement-L.cube", 1.0});
  CHECK(look_slots(chosen).size() == 1 + defaults.look_layers.size() + 1);
  set_contract_v2_looks(false);
  CHECK(look_slots(defaults).size() == 1 + defaults.look_layers.size());
}

TEST_CASE("métadonnées : la pile SDR ne change pas les gris de plus de 2 %", "[app][look]") {
  std::ifstream file(std::string(AJP_APP_SOURCE_DIR) + "/../../data/luts-v2/ver9-couleur.cube");
  REQUIRE(file);
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  const auto lut = ajp::render::parse_cube(text);
  REQUIRE(lut);
  const int n = lut->size;
  for (int i = 1; i < n; ++i) {
    const double in = static_cast<double>(i) / (n - 1);
    const std::size_t at = 3 * static_cast<std::size_t>(i + n * i + n * n * i);
    for (int c = 0; c < 3; ++c) {
      // En lumière (gamma 2,4 du domaine SDR) : au plus 2 % d'écart sur chaque composante.
      const double ratio = std::pow(lut->rgb[at + static_cast<std::size_t>(c)] / in, 2.4);
      CHECK(std::abs(ratio - 1.0) <= 0.02);
    }
  }
}
