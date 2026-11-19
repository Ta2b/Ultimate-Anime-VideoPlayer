// Les tuiles qui bougent (upscale_tiles.h, kernels/spec/span_tiles.h) : la dilatation et les masques du
// moteur, puis la décision sur des cartes synthétiques (sans carte, tuile qui change et ce qui la voit,
// débandage, dérive lente, lueur locale, recette de l'entrée, panoramique entier et ses deux bords, coupe,
// limite d'âge, refus) et la sorte Tiles de la reprise ; puis le filet de l'écart des blocs sous la carte.
#include "app/play/upscale_tiles.h"

#include <cstdint>
#include <string>
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
  const TileFacts fresh{.stale = TileStale::None, .can_shift = true, .scene = &scene};

  // La première image se calcule entière (l'état ne suit encore rien), comme une image sans carte.
  CHECK_FALSE(t.plan(fresh).tiled);
  CHECK_FALSE(t.plan(TileFacts{.stale = TileStale::None, .can_shift = true}).tiled);

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
  CHECK_FALSE(t.plan(TileFacts{.stale = TileStale::Restart, .can_shift = true, .scene = &scene}).tiled);

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
  p = t.plan(TileFacts{.stale = TileStale::None, .can_shift = true, .scene = &pan});
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
  CHECK_FALSE(t.plan(TileFacts{.stale = TileStale::None, .can_shift = false, .scene = &pan}).tiled);
  // Un panoramique d'un pixel et demi : pas de multiple du pas, entière.
  pan.camera[0] = 1.5;
  CHECK_FALSE(t.plan(TileFacts{.stale = TileStale::None, .can_shift = true, .scene = &pan}).tiled);

  // Une coupe : entière, même si le reste dit « rien ne change ».
  scene.cut_likely = true;
  CHECK_FALSE(t.plan(fresh).tiled);
  scene.cut_likely = false;

  // Limite d'âge : une image fixe se reprend au plus kMaxAge fois par tuile, puis tout se recalcule.
  int tiled = 0;
  for (int i = 0; i < SpanTiles::kMaxAge + 1; ++i) tiled += t.plan(fresh).tiled ? 1 : 0;
  CHECK(tiled == SpanTiles::kMaxAge);
  CHECK_FALSE(t.summary().empty());
  // Un refus du moteur : l'image redevient entière, et ne compte plus parmi celles par tuiles. La ligne du
  // bilan sort quand même, avec la raison de chaque image entière.
  SpanTiles once;
  once.set_grid(kGrid);
  CHECK_FALSE(once.plan(fresh).tiled);
  CHECK(once.plan(fresh).tiled);
  once.refused();
  CHECK_FALSE(once.plan(TileFacts{.stale = TileStale::Graded, .can_shift = true, .scene = &scene}).tiled);
  CHECK_FALSE(once.plan(TileFacts{.stale = TileStale::None, .can_shift = true}).tiled);
  CHECK(once.summary() ==
        "SPAN par tuiles : 0 images sur 4 calculées (0.0 %) ; images entières : 1 sans carte, 1 refus du "
        "moteur, 1 sortie d'avant gradée sur place, 1 début, saut ou image abîmée");
  // Rien de calculé : la ligne le dit aussi ; sans tuiles (TensorRT, --no-span-tiles), rien.
  SpanTiles idle;
  idle.set_grid(kGrid);
  CHECK(idle.summary() == "SPAN par tuiles : 0 images sur 0 calculées (0.0 %) ; images entières : aucune");

  // Dans la reprise : une image non tenue mais lisible par tuiles rend Tiles et le plan ; --no-span-tiles,
  // Compute.
  SpanReuse r;
  r.tiles.set_grid(kGrid);
  ReuseFacts facts{.same_epoch = true,
                   .has_output = true,
                   .pair_held = false,
                   .tiles_stale = TileStale::None,
                   .can_shift = true,
                   .scene = &scene};
  CHECK(r.decide(facts) == ReuseKind::Compute);  // la première suit l'état
  tile(scene, 1, 1).changed = true;
  CHECK(r.decide(facts) == ReuseKind::Tiles);
  CHECK(r.tile_plan.tiled);
  facts.tiles_stale = TileStale::Overwritten;
  CHECK(r.decide(facts) == ReuseKind::Compute);
  CHECK_FALSE(r.tile_plan.tiled);
  // Un saut l'emporte sur la sortie d'avant au bilan ; une chaîne sans tuiles, sur le saut.
  facts.tiles_stale = TileStale::Graded;
  facts.same_epoch = false;
  CHECK(r.decide(facts) == ReuseKind::Compute);
  facts.tiles_stale = TileStale::Chain;
  CHECK(r.decide(facts) == ReuseKind::Compute);
  CHECK(
      r.tiles.summary().ends_with("1 sortie d'avant écrasée (image créée, source sautée), 1 chaîne sans "
                                  "tuiles (export, YUV, TemporalFix, Smooth, plaque), 2 début, saut ou image "
                                  "abîmée"));
  facts.same_epoch = true;
  SpanReuse off;
  off.tiles.enabled = false;
  off.tiles.set_grid(kGrid);
  facts.tiles_stale = TileStale::None;
  CHECK(off.decide(facts) == ReuseKind::Compute);
  CHECK(off.decide(facts) == ReuseKind::Compute);
  CHECK(off.tiles.summary().empty());
  // Un moteur sans tuiles (TensorRT) : jamais actif.
  SpanTiles none;
  none.set_grid(kernels::SpanTileGrid{});
  CHECK_FALSE(none.active());
  CHECK(none.summary().empty());
}

TEST_CASE("Tuiles qui bougent : l'écart des blocs voit ce que la carte laisse passer", "[app][tiles]") {
  // Blocs 16 × 16 de la mesure à la taille de travail (64 × 32), du grain partout (0,78 niveau), et une
  // petite bouche (15 niveaux) dans le bloc (10, 13), pixels 160 à 176 et 208 à 224 : la tuile (2, 13) du
  // moteur, sous une carte qui ne dit rien.
  std::vector<std::uint32_t> deviation(64 * 32, 200);
  deviation[13 * 64 + 10] = 15 * 256;
  kernels::SceneMotion scene = still_scene();
  SpanTiles t;
  t.set_grid(kGrid);
  const TileFacts fresh{.fresh = true,
                        .can_shift = true,
                        .scene = &scene,
                        .blocks = PairBlocks{.deviation = deviation, .cols = 64, .side_px = 16.0F}};
  CHECK_FALSE(t.plan(fresh).tiled);  // la première suit l'état
  TilePlan p = t.plan(fresh);
  REQUIRE(p.tiled);
  CHECK(core_at(p, 2, 13) == 1);
  CHECK(kernels::count_span_tiles(p.core) == 27);  // 3 × 9 : une colonne et quatre lignes autour
  CHECK(t.summary().find("a vu 1 tuiles que la carte ne disait pas changées, sur 1 images") !=
        std::string::npos);
  // La mesure sur la copie réduite de RIFE (720p d'un travail 1080p) : un bloc couvre 24 pixels, le même
  // bloc est en 240 à 264 et 312 à 336, sur les tuiles (3, 19) à (4, 20).
  TileFacts small = fresh;
  small.blocks.side_px = 24.0F;
  p = t.plan(small);
  REQUIRE(p.tiled);
  CHECK(core_at(p, 2, 13) == 0);
  CHECK(core_at(p, 4, 20) == 1);
  CHECK(kernels::count_span_tiles(p.core) == 4 * 10);
  // Sous le seuil (1,9 niveau) : rien à recalculer.
  deviation[13 * 64 + 10] = 486;
  p = t.plan(fresh);
  REQUIRE(p.tiled);
  CHECK(kernels::count_span_tiles(p.core) == 0);
  deviation[13 * 64 + 10] = 15 * 256;
  // Sous un décalage, la mesure ne suit pas le contenu : seuls les bords et ce qui les voit.
  kernels::SceneMotion pan = still_scene();
  pan.camera[0] = 4.0;
  pan.background_held = false;
  for (auto& s : pan.tiles) s.dx = 4.0F;
  TileFacts panned = fresh;
  panned.scene = &pan;
  p = t.plan(panned);
  REQUIRE(p.tiled);
  CHECK(p.shift_x == 4);
  CHECK(kernels::count_span_tiles(p.core) == 4 * kGrid.rows);
  // Dans la reprise : les blocs passent de ReuseFacts au plan.
  SpanReuse r;
  r.tiles.set_grid(kGrid);
  ReuseFacts facts{.same_epoch = true,
                   .has_output = true,
                   .tiles_ok = true,
                   .can_shift = true,
                   .scene = &scene,
                   .blocks = fresh.blocks};
  CHECK(r.decide(facts) == ReuseKind::Compute);
  CHECK(r.decide(facts) == ReuseKind::Tiles);
  CHECK(r.tile_plan.core[static_cast<std::size_t>(13 * kGrid.cols + 2)] == 1);
}

}  // namespace kirari::app::play
