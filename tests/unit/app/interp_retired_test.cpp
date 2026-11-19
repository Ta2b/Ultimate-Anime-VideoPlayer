// Les choix d'interpolation retirés du menu (interp_retired.h) : un play.toml d'avant novembre 2026 se relit
// et tombe sur le plus proche des trois modèles du menu.
#include "app/play/interp_retired.h"

#include <catch2/catch_test_macros.hpp>

#include "app/play/settings.h"

namespace kirari::app::play {

TEST_CASE("play.toml d'avant le menu par modèles : chaque ancien choix tombe sur le plus proche",
          "[app][settings][interp_retired]") {
  struct Old {
    const char* toml;
    MotionEngine engine;
    const char* model;
  };
  const Old cases[] = {
      // Équilibré (4.26) et Très rapide (4.25 lite).
      {"[interpolation]\nengine = \"rife\"\nmodel = \"rife_v4.26.onnx\"\n", MotionEngine::Rife,
       "rife_v4.25.onnx"},
      {"[interpolation]\nengine = \"rife\"\nmodel = \"rife_v4.25_lite.onnx\"\n", MotionEngine::Rife,
       "rife_v4.22_lite.onnx"},
      // Élevé (4.26 et GMFSS au milieu des paires fortes) et GMFSS union seul.
      {"[interpolation]\nengine = \"rife\"\nmodel = \"rife_v4.26.onnx\"\ngmfss_middle = true\n",
       MotionEngine::Rife, "rife_v4.25.onnx"},
      {"[interpolation]\nengine = \"gmfss_union\"\n", MotionEngine::Rife, "rife_v4.25.onnx"},
      // Les trois du menu, et le défaut (nom vide), ne bougent pas.
      {"[interpolation]\nengine = \"rife\"\nmodel = \"rife_v4.25.onnx\"\n", MotionEngine::Rife,
       "rife_v4.25.onnx"},
      {"[interpolation]\nengine = \"rife\"\nmodel = \"rife_v4.22_lite.onnx\"\n", MotionEngine::Rife,
       "rife_v4.22_lite.onnx"},
      {"[interpolation]\nengine = \"rife\"\n", MotionEngine::Rife, ""},
      {"[interpolation]\nengine = \"distil_drba\"\n", MotionEngine::DistilDrba, ""},
  };
  for (const Old& old : cases) {
    CAPTURE(old.toml);
    const auto read = parse_settings(old.toml);
    REQUIRE(read);
    CHECK(read->motion_engine == old.engine);
    CHECK(read->rife_model == old.model);
    CHECK_FALSE(read->gmfss_middle);
    // Relu puis réécrit : le fichier garde le choix du menu.
    const auto again = parse_settings(format_settings(*read));
    REQUIRE(again);
    CHECK(*again == *read);
  }
  CHECK(nearest_rife_model("rife_v4.26.onnx") == "rife_v4.25.onnx");
  CHECK(nearest_rife_model("rife_v4.25_lite.onnx") == "rife_v4.22_lite.onnx");
  CHECK(nearest_rife_model("rife_v4.18.onnx") == "rife_v4.18.onnx");
}

}  // namespace kirari::app::play
