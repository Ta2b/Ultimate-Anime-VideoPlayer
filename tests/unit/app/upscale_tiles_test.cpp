// Les tuiles qui bougent (upscale_tiles.h, kernels/spec/span_tiles.h) : la dilatation et les masques du
// moteur, puis la décision sur des cartes synthétiques (sans carte, tuile qui change et ce qui la voit,
// débandage, dérive lente, lueur locale, recette de l'entrée, panoramique entier et ses deux bords, coupe,
// limite d'âge, refus) et la sorte Tiles de la reprise.
#include "app/play/upscale_tiles.h"

#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/upscale_reuse.h"
#include "kernels/spec/scene_flow.h"
#include "kernels/spec/span_tiles.h"

namespace kirari::app::play {
namespace {

// Une image de travail de 1024 × 512 : tuiles du moteur de 64 × 16 (16 × 32), halo d'une colonne et de quatre
// lignes (SPAN à huit blocs : portée 26 pixels de demi-résolution) ; carte en tuiles de 32 (32 × 16).
constexpr kernels::SpanTileGrid kGrid{.tile_w = 64,
                                      .tile_h = 16,
                                      .cols = 16,
                                      .rows = 32,
                                      .reach_cols = kernels::span_reach_tiles(26, 32),
                                      .reach_rows = kernels::span_reach_tiles(26, 8),
                                      .shift_step = 2,
                                      .scale = 2};

kernels::SceneMotion still_scene() {
  kernels::SceneMotion s;
  s.camera_fit = true;
  s.tile_px = 32;
  s.cols = 32;
  s.rows = 16;
  s.tiles.assign(32 * 16, kernels::SceneTile{.kind = kernels::SceneClass::Background, .confidence = 1.0F});
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
  std::vector<std::uint8_t> one(8 * 16, 0);
  one[static_cast<std::size_t>(8 * 8 + 4)] = 1;
  CHECK(kernels::count_span_tiles(kernels::dilate_span_tiles(one, 8, 16, 1, 4)) == 27);
  std::vector<std::uint8_t> corner(8 * 16, 0);
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

  // Une tuile de la carte change (pixels 160 à 192, 96 à 128) : les tuiles du moteur dessous, (2, 6) et
  // (2, 7), et toute sortie qui les voit, à une colonne et quatre lignes : 3 × 10 tuiles.
  tile(scene, 5, 3).changed = true;
  TilePlan p = t.plan(fresh);
  REQUIRE(p.tiled);
  CHECK(p.shift_x == 0);
  CHECK(core_at(p, 2, 6) == 1);
  CHECK(core_at(p, 1, 2) == 1);
  CHECK(core_at(p, 3, 11) == 1);
  CHECK(core_at(p, 0, 6) == 0);
  CHECK(core_at(p, 2, 12) == 0);
  CHECK(kernels::count_span_tiles(p.core) == 30);
  // Débandée, l'entrée voit la source jusqu'à 16 pixels de plus : une colonne et une ligne de plus.
  TileFacts debanded = fresh;
  debanded.spread_px = 16;
  p = t.plan(debanded);
  REQUIRE(p.tiled);
  CHECK(kernels::count_span_tiles(p.core) == 5 * 12);
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

  // Lueur locale : 0,6 code par paire sur une tuile de la carte, sous le seuil de « a changé » ; à la
  // deuxième, plus d'un code reçu : ses tuiles se recalculent.
  tile(scene, 5, 3).shift = 0.6F;
  p = t.plan(fresh);
  REQUIRE(p.tiled);
  CHECK(kernels::count_span_tiles(p.core) == 0);
  p = t.plan(fresh);
  REQUIRE(p.tiled);
  CHECK(core_at(p, 2, 6) == 1);
  CHECK(kernels::count_span_tiles(p.core) == 30);
  tile(scene, 5, 3).shift = 0.0F;

  // Une autre recette de l'entrée (débandage allumé en cours de lecture) : entière, puis par tuiles.
  TileFacts other = fresh;
  other.input_key = 7;
  CHECK_FALSE(t.plan(other).tiled);
  CHECK(t.plan(other).tiled);
  CHECK_FALSE(t.plan(fresh).tiled);

  // Panoramique de 4 pixels vers la droite (deux pas), décor sûr qui suit : la sortie se décale ; la bande
  // qui entre (première colonne) et le bord d'où le contenu sort (dernière colonne) se recalculent, avec ce
  // qui les voit.
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
  CHECK(core_at(p, 1, 5) == 1);
  CHECK(core_at(p, 2, 5) == 0);
  CHECK(core_at(p, 7, 5) == 0);
  CHECK(core_at(p, 14, 5) == 1);
  CHECK(core_at(p, 15, 5) == 1);
  CHECK(kernels::count_span_tiles(p.core) == 4 * kGrid.rows);
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
  // Un refus du moteur : l'image redevient entière, et ne compte plus parmi celles par tuiles.
  SpanTiles once;
  once.set_grid(kGrid);
  CHECK_FALSE(once.plan(fresh).tiled);
  CHECK(once.plan(fresh).tiled);
  once.refused();
  CHECK(once.summary().empty());

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
