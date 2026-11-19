// Les tuiles qui bougent (upscale_tiles.h, kernels/spec/span_tiles.h) : la dilatation et les masques du
// moteur, puis la décision sur des cartes synthétiques (sans carte, tuile qui change, dérive lente,
// panoramique entier, coupe, limite d'âge) et la sorte Tiles de la reprise.
#include "app/play/upscale_tiles.h"

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/upscale_reuse.h"
#include "kernels/spec/scene_flow.h"
#include "kernels/spec/span_tiles.h"

namespace kirari::app::play {
namespace {

// Une image de travail de 512 × 256 : tuiles du moteur de 64 × 16 (8 × 16), halo d'une colonne et de quatre
// lignes (SPAN à huit blocs : portée 26 pixels de demi-résolution) ; carte en tuiles de 32 (16 × 8).
constexpr kernels::SpanTileGrid kGrid{.tile_w = 64,
                                      .tile_h = 16,
                                      .cols = 8,
                                      .rows = 16,
                                      .reach_cols = kernels::span_reach_tiles(26, 32),
                                      .reach_rows = kernels::span_reach_tiles(26, 8),
                                      .shift_step = 2,
                                      .scale = 2};

kernels::SceneMotion still_scene() {
  kernels::SceneMotion s;
  s.camera_fit = true;
  s.tile_px = 32;
  s.cols = 16;
  s.rows = 8;
  s.tiles.assign(16 * 8, kernels::SceneTile{.kind = kernels::SceneClass::Background, .confidence = 1.0F});
  s.background_held = true;
  s.drawing_held = true;
  return s;
}

kernels::SceneTile& tile(kernels::SceneMotion& s, int c, int r) {
  return s.tiles[static_cast<std::size_t>(r * s.cols + c)];
}

std::uint8_t core_at(const TilePlan& p, int c, int r) {
  return p.core[static_cast<std::size_t>(r * kGrid.cols + c)];
}

}  // namespace

TEST_CASE("Tuiles qui bougent : masques, décision sur la carte et reprise", "[app][tiles]") {
  // La portée en tuiles et la dilatation : une tuile au milieu donne 3 × 9 tuiles, une au coin 2 × 5.
  REQUIRE(kGrid.reach_cols == 1);
  REQUIRE(kGrid.reach_rows == 4);
  std::vector<std::uint8_t> one(static_cast<std::size_t>(kGrid.count()), 0);
  one[static_cast<std::size_t>(8 * kGrid.cols + 4)] = 1;
  CHECK(kernels::count_span_tiles(kernels::dilate_span_tiles(one, 8, 16, 1, 4)) == 27);
  std::vector<std::uint8_t> corner(static_cast<std::size_t>(kGrid.count()), 0);
  corner[0] = 1;
  CHECK(kernels::count_span_tiles(kernels::dilate_span_tiles(corner, 8, 16, 1, 4)) == 10);
  std::vector<std::uint32_t> words(kernels::kSpanTileMaskWords, 0xFFFFFFFFU);
  kernels::pack_span_tiles(one, words);
  CHECK(words[2] == (1U << 4U));  // tuile 68 : mot 2, bit 4
  CHECK(words[0] == 0U);

  SpanTiles t;
  t.set_grid(kGrid);
  REQUIRE(t.active());
  kernels::SceneMotion scene = still_scene();
  const TileFacts fresh{.fresh = true, .can_shift = true, .scene = &scene};

  // La première image se calcule entière (l'état ne suit encore rien), comme une image sans carte.
  CHECK_FALSE(t.plan(fresh).tiled);
  CHECK_FALSE(t.plan(TileFacts{.fresh = true, .can_shift = true}).tiled);

  // Une tuile de la carte change (pixels 160 à 192, 96 à 128) : seules les tuiles du moteur dessous.
  tile(scene, 5, 3).changed = true;
  TilePlan p = t.plan(fresh);
  REQUIRE(p.tiled);
  CHECK(p.shift_x == 0);
  CHECK(core_at(p, 2, 6) == 1);
  CHECK(core_at(p, 2, 7) == 1);
  CHECK(kernels::count_span_tiles(p.core) == 2);
  tile(scene, 5, 3).changed = false;

  // Une image saine mais pas fraîche (saut, image abîmée derrière) : entière.
  CHECK_FALSE(t.plan(TileFacts{.fresh = false, .can_shift = true, .scene = &scene}).tiled);

  // Dérive lente (nuages) : 0,2 pixel par paire, sous le seuil du flux ; à la troisième, plus de 0,5 pixel
  // reçu depuis le calcul : tout est à refaire, la région dépasse kMaxShare, l'image est entière.
  for (auto& s : scene.tiles) s.dx = 0.2F;
  CHECK(t.plan(fresh).tiled);
  CHECK(t.plan(fresh).tiled);
  CHECK_FALSE(t.plan(fresh).tiled);
  for (auto& s : scene.tiles) s.dx = 0.0F;

  // Panoramique de 4 pixels vers la droite (deux pas), décor sûr qui suit : la sortie se décale, seules la
  // bande qui entre (première colonne) se recalcule.
  kernels::SceneMotion pan = still_scene();
  pan.camera[0] = 4.05;
  pan.background_held = false;
  for (auto& s : pan.tiles) {
    s.dx = 4.0F;
    s.changed = true;
  }
  p = t.plan(TileFacts{.fresh = true, .can_shift = true, .scene = &pan});
  REQUIRE(p.tiled);
  CHECK(p.shift_x == 4);
  CHECK(p.shift_y == 0);
  CHECK(core_at(p, 0, 5) == 1);
  CHECK(core_at(p, 3, 5) == 0);
  CHECK(core_at(p, 7, 5) == 0);
  CHECK(kernels::count_span_tiles(p.core) == kGrid.rows);
  // Le même panoramique sans second tampon (sortie sur place) : entière.
  CHECK_FALSE(t.plan(TileFacts{.fresh = true, .can_shift = false, .scene = &pan}).tiled);
  // Un panoramique d'un pixel et demi : pas de multiple du pas, entière.
  pan.camera[0] = 1.5;
  CHECK_FALSE(t.plan(TileFacts{.fresh = true, .can_shift = true, .scene = &pan}).tiled);

  // Une coupe : entière, même si le reste dit « rien ne change ».
  scene.cut_likely = true;
  CHECK_FALSE(t.plan(fresh).tiled);
  scene.cut_likely = false;

  // Limite d'âge : une image fixe se reprend au plus kMaxAge fois par tuile, puis tout se recalcule.
  int tiled = 0;
  for (int i = 0; i < SpanTiles::kMaxAge + 1; ++i) tiled += t.plan(fresh).tiled ? 1 : 0;
  CHECK(tiled == SpanTiles::kMaxAge);
  CHECK_FALSE(t.summary().empty());

  // Dans la reprise : une image non tenue mais lisible par tuiles rend Tiles et le plan ; --no-span-tiles,
  // Compute.
  SpanReuse r;
  r.tiles.set_grid(kGrid);
  ReuseFacts facts{.same_epoch = true,
                   .has_output = true,
                   .pair_held = false,
                   .tiles_ok = true,
                   .can_shift = true,
                   .scene = &scene};
  CHECK(r.decide(facts) == ReuseKind::Compute);  // la première suit l'état
  tile(scene, 1, 1).changed = true;
  CHECK(r.decide(facts) == ReuseKind::Tiles);
  CHECK(r.tile_plan.tiled);
  facts.tiles_ok = false;
  CHECK(r.decide(facts) == ReuseKind::Compute);
  CHECK_FALSE(r.tile_plan.tiled);
  SpanReuse off;
  off.tiles.enabled = false;
  off.tiles.set_grid(kGrid);
  facts.tiles_ok = true;
  CHECK(off.decide(facts) == ReuseKind::Compute);
  CHECK(off.decide(facts) == ReuseKind::Compute);
  // Un moteur sans tuiles (TensorRT) : jamais actif.
  SpanTiles none;
  none.set_grid(kernels::SpanTileGrid{});
  CHECK_FALSE(none.active());
}

}  // namespace kirari::app::play
