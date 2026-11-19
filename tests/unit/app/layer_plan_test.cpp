#include "app/play/layer_plan.h"

#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "app/play/chain_setup.h"
#include "app/play/settings.h"
#include "doubles.h"
#include "upscale_rig.h"

using namespace kirari;
using namespace kirari::app::play;
using Catch::Matchers::WithinAbs;
using kirari::kernels::SceneClass;
using kirari::kernels::SceneMotion;

namespace {

// Une carte de 64 × 36 pixels de travail en tuiles de 8 (8 × 5 tuiles, la dernière ligne tronquée), toute en
// décor sûr, caméra de dx pixels vers la droite.
SceneMotion pan(double dx) {
  SceneMotion s;
  s.camera = {dx, 1.0, 0.0, 0.0, 0.0, 1.0};
  s.camera_fit = true;
  s.camera_refined = true;
  s.residual = 0.2;
  s.tile_px = 8;
  s.cols = 8;
  s.rows = 5;
  s.tiles.assign(40, {.kind = SceneClass::Background, .confidence = 0.9F});
  return s;
}

}  // namespace

// Un seul test pour les décisions : la carte absente ou douteuse ne change rien, l'érosion autour d'un
// dessin, la grille portée à la sortie, la plaque qui suit la caméra et se renouvelle.
TEST_CASE("calques : décisions d'une paire et suivi de la plaque", "[layers]") {
  // Sans carte, ou carte douteuse : pas de calques, pour une raison nommée.
  CHECK(plan_layers(std::nullopt, 64, 36).error() == LayerSkip::NoMap);
  SceneMotion cut = pan(4.0);
  cut.cut_likely = true;
  CHECK(plan_layers(cut, 64, 36).error() == LayerSkip::Cut);
  SceneMotion loose = pan(4.0);
  loose.residual = 0.8;
  CHECK(plan_layers(loose, 64, 36).error() == LayerSkip::Residual);
  SceneMotion unfit = pan(4.0);
  unfit.camera_fit = false;
  CHECK(plan_layers(unfit, 64, 36).error() == LayerSkip::NoCamera);
  SceneMotion raw = pan(4.0);
  raw.camera_refined = false;  // caméra de NVOF seule : son cisaillement sur les panoramiques lents
  CHECK(plan_layers(raw, 64, 36).error() == LayerSkip::Unrefined);
  // Un zoom de 2 % autour du coin : la caméra passe des centres continus aux indices, M(x + ½) − ½.
  SceneMotion zoom = pan(0.0);
  zoom.camera = {0.0, 1.02, 0.0, 0.0, 0.0, 1.02};
  const auto zoomed = plan_layers(zoom, 64, 36);
  REQUIRE(zoomed);
  CHECK_THAT(zoomed->camera[0], WithinAbs(0.01, 1e-12));
  CHECK_THAT(zoomed->camera[1], WithinAbs(1.02, 1e-12));
  SceneMotion crowd = pan(4.0);
  for (auto& tile : crowd.tiles) tile.kind = SceneClass::Drawing;
  CHECK(plan_layers(crowd, 64, 36).error() == LayerSkip::FewTiles);
  for (const auto skip : {LayerSkip::NoMap, LayerSkip::Unrefined, LayerSkip::Grid})
    CHECK_FALSE(name(skip).empty());

  // Un dessin en (3, 2) : lui et ses huit voisines gardent RIFE ; un aplat qui ne change pas reste décor.
  SceneMotion scene = pan(4.0);
  scene.tiles[2 * 8 + 3].kind = SceneClass::Drawing;
  scene.tiles[0].kind = SceneClass::Uncertain;
  const auto layers = plan_layers(scene, 64, 36);
  REQUIRE(layers);
  CHECK(layers->decor[2 * 8 + 3] == 0);
  CHECK(layers->decor[1 * 8 + 2] == 0);
  CHECK(layers->decor[3 * 8 + 4] == 0);
  CHECK(layers->decor[2 * 8 + 5] == 1);
  CHECK(layers->decor[0] == 1);
  CHECK_THAT(layers->share, WithinAbs(31.0 / 40.0, 1e-12));

  // Un personnage rapide : cinq tuiles vers la droite d'une image à l'autre, tout son chemin garde RIFE.
  SceneMotion fast = pan(4.0);
  fast.tiles[2 * 8].kind = SceneClass::Drawing;
  fast.tiles[2 * 8].dx = 40.0F;
  const auto swept = plan_layers(fast, 64, 36);
  REQUIRE(swept);
  CHECK(swept->decor[1 * 8 + 3] == 0);
  CHECK(swept->decor[3 * 8 + 6] == 0);
  CHECK(swept->decor[2 * 8 + 7] == 1);
  CHECK(swept->decor[0] == 1);
  // Une tuile dessin sans vecteur (déplacement non fini) : seule sa zone proche garde RIFE.
  SceneMotion lost = pan(4.0);
  lost.tiles[2 * 8 + 3].kind = SceneClass::Drawing;
  lost.tiles[2 * 8 + 3].dx = std::numeric_limits<float>::quiet_NaN();
  const auto near = plan_layers(lost, 64, 36);
  REQUIRE(near);
  CHECK(near->decor == layers->decor);  // la même zone que le dessin immobile plus haut

  // Sortie ×2 : tuiles de 16, déplacements de ±4 pixels de sortie à t = ½ (8 de caméra en sortie).
  const kirari::kernels::LayerGrid grid = layer_grid(*layers, 128, 72);
  CHECK(grid.tile_w == 16.0F);
  CHECK_FALSE(grid.get(3, 2));
  CHECK(grid.get(5, 2));
  const auto params = pair_params(*layers, 128, 72, 0.5, false);
  REQUIRE(params);
  CHECK(params->count == 2);
  CHECK_THAT(params->sources[0].shift[0], WithinAbs(-4.0, 1e-5));
  CHECK_THAT(params->sources[1].shift[0], WithinAbs(4.0, 1e-5));
  CHECK_THAT(params->sources[1].weight, WithinAbs(0.5, 1e-7));

  // Plaque : la première paire sûre fait la clé ; les suivantes s'en servent au mouvement cumulé ; quand la
  // clé ne couvre plus les trois quarts de l'image, une nouvelle clé ; une paire sans caméra coupe tout.
  PlateTrack track;
  const auto step4 = plan_layers(pan(4.0), 64, 36);
  CHECK(track.step(step4, true).action == PlateAction::Refresh);
  const PlateStep used = track.step(step4, true);
  CHECK(used.action == PlateAction::Use);
  CHECK_THAT(used.key_to_b[0], WithinAbs(4.0, 1e-12));
  CHECK(track.step(step4, true).action == PlateAction::Use);  // 8 pixels sur 64 : la clé couvre encore
  CHECK(track.step(step4, true).action == PlateAction::Use);  // 12
  CHECK(track.step(step4, true).action == PlateAction::Use);  // 16 : un quart, encore couvert
  CHECK(track.step(step4, true).action == PlateAction::Refresh);
  CHECK(track.step(plan_layers(std::nullopt, 64, 36), true).action == PlateAction::None);
  CHECK(track.step(step4, true).action == PlateAction::Refresh);
  CHECK(track.step(step4, false).action == PlateAction::Refresh);  // saut ou coupe : nouvelle clé
  CHECK_THAT(plate_cover(kirari::kernels::kIdentityAffine, 64, 36), WithinAbs(1.0, 1e-12));
  const auto plate = plate_params(track.key(), *step4, {8.0, 1.0, 0.0, 0.0, 0.0, 1.0}, 128, 72, false);
  REQUIRE(plate);
  CHECK(plate->count == 1);
  CHECK(plate->source_grid.cols == 8);
  CHECK_THAT(plate->sources[0].shift[0], WithinAbs(-16.0, 1e-5));

  // La chaîne : en lecture, avec la carte ; les calques avec RIFE mouvement seul ; rien à l'export.
  ChainEngines c;
  c.layers = true;
  const test::FakeClock clock;
  CHECK_FALSE(layer_setup(c, &clock).layers);  // sans --scene-map
  c.scene_map = true;
  CHECK_FALSE(layer_setup(c, &clock).layers);  // sans RIFE mouvement
  CHECK_FALSE(layer_setup(c, nullptr).layers);

  // play.toml : les deux clés se relisent ; un ancien fichier sans elles les laisse éteintes.
  PlaySettings on;
  on.layers = true;
  on.background_plate = true;
  const auto reread = parse_settings(format_settings(on));
  REQUIRE(reread);
  CHECK(reread->layers);
  CHECK(reread->background_plate);
  const auto old = parse_settings("[interpolation]\nsmooth_drawings = true\n");
  REQUIRE(old);
  CHECK_FALSE(old->layers);
  CHECK_FALSE(old->background_plate);
}

// La règle absolue : calques et plaque allumés sans carte de la scène (AMD, Intel, NVOF refusé) donnent les
// mêmes passes que sans eux ; ni la plaque ni la composition ne tournent.
TEST_CASE("calques : sans carte, l'étage fait exactement ce qu'il faisait", "[layers][app][stage]") {
  const bool layers = GENERATE(false, true);
  INFO("calques " << layers);
  upscale_test::Rig rig;
  test::FakeEngine plate{"plate", {"input"}, {"output"}};
  rig.backend.score(SourceId{2}, SourceId{3}, 0.0, 10);
  rig.backend.motion_max(500.0F);
  UpscaleSetup s = rig.setup(true, Factor{.num = 3, .den = 1});
  s.rife = nullptr;
  s.rife_motion = &rig.motion;
  s.warp_after_span = true;
  s.layers = {.layers = layers, .plate = layers ? &plate : nullptr};
  rig.start(std::move(s));
  rig.feed(1, 6);
  REQUIRE(rig.collect(14));
  rig.stage.reset();
  using Pair = std::pair<std::uint32_t, std::uint32_t>;
  CHECK(rig.backend.warp_pairs() ==
        std::vector<Pair>{{1, 2}, {1, 2}, {2, 4}, {2, 4}, {4, 5}, {4, 5}, {5, 6}, {5, 6}});
  CHECK(rig.backend.count("engine span") == 5);
  CHECK(rig.backend.count("engine plate") == 0);
}
