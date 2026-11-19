// Les tuiles qui bougent (upscale_tiles.h, kernels/spec/span_tiles.h) : la dilatation et les masques du
// moteur, puis la décision sur des cartes synthétiques (sans carte, tuile qui change et ce qui la voit,
// débandage, dérive lente, lueur locale, recette de l'entrée, panoramique entier et ses deux bords, coupe,
// limite d'âge, refus, caméra au bruit près, lente, qui bouge partout, zoom) et la sorte Tiles de la reprise.
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
        "moteur, 1 sortie d'avant gradée sur place, 1 début, saut ou image abîmée ; caméra de la carte sur 3 "
        "images : 3 tenues (sous 0.25 pixel partout), 0 en partie (zoom, rotation), 0 qui bougent partout");
  // Rien de calculé : la ligne le dit aussi.
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
      r.tiles.summary().contains("1 sortie d'avant écrasée (image créée, source sautée), 1 chaîne sans "
                                 "tuiles (export, YUV, TemporalFix, Smooth, plaque), 2 début, saut ou image "
                                 "abîmée"));
  facts.same_epoch = true;
  SpanReuse off;
  off.tiles.enabled = false;
  off.tiles.set_grid(kGrid);
  facts.tiles_stale = TileStale::None;
  CHECK(off.decide(facts) == ReuseKind::Compute);
  CHECK(off.decide(facts) == ReuseKind::Compute);
  CHECK(off.tiles.summary() == "SPAN par tuiles : coupées (--no-span-tiles)");
  // Un moteur sans tuiles (TensorRT) : jamais actif, et la ligne de fin le dit ; sans SPAN, aucune ligne.
  SpanTiles none;
  none.set_grid(kernels::SpanTileGrid{});
  CHECK_FALSE(none.active());
  CHECK(none.summary() == "SPAN par tuiles : aucune, ce moteur ne calcule que l'image entière");
  CHECK(SpanTiles{}.summary().empty());
}

TEST_CASE("Tuiles qui bougent : caméra au bruit près, lente, qui bouge partout, zoom", "[app][tiles]") {
  SpanTiles t;
  t.set_grid(kGrid);
  kernels::SceneMotion scene = still_scene();
  // La carte ne dit pas la caméra tenue (son seuil se lit au coin haut gauche, partie linéaire comprise) :
  // jusqu'ici, chacune de ces images était entière (« caméra sans décalage entier »).
  scene.background_held = false;
  const TileFacts facts{.stale = TileStale::None, .can_shift = true, .scene = &scene};
  CHECK_FALSE(t.plan(facts).tiled);  // la première suit l'état

  // Dialogue fixe : l'ajustement hésite de ± 0,15 pixel d'une paire à l'autre. Signé, le bruit s'annule :
  // toutes les images par tuiles, rien à recalculer.
  for (int i = 0; i < 10; ++i) {
    scene.camera[0] = i % 2 == 0 ? 0.15 : -0.15;
    const TilePlan p = t.plan(facts);
    REQUIRE(p.tiled);
    CHECK(kernels::count_span_tiles(p.core) == 0);
  }

  // Caméra lente, 0,15 pixel par paire dans le même sens : à la deuxième, 0,3 pixel reçu, au-delà du quart de
  // pixel : tout est à refaire, la région dépasse kMaxShare, l'image est entière ; puis ça recommence.
  scene.camera[0] = 0.15;
  CHECK(t.plan(facts).tiled);
  CHECK_FALSE(t.plan(facts).tiled);
  CHECK(t.plan(facts).tiled);

  // Une caméra qui bouge partout d'au moins le quart de pixel, sans décalage entier : entière d'emblée.
  scene.camera[0] = 0.62;
  CHECK_FALSE(t.plan(facts).tiled);

  // Zoom lent autour du centre (0,05 % par paire) : les colonnes du bord bougent de plus d'un quart de pixel
  // et se recalculent, avec ce qui les voit ; le centre se reprend.
  constexpr double kZoom = 0.0005;
  scene.camera = {-kZoom * 512.0, 1.0 + kZoom, 0.0, -kZoom * 256.0, 0.0, 1.0 + kZoom};
  const TilePlan p = t.plan(facts);
  REQUIRE(p.tiled);
  CHECK(core_at(p, 0, 16) == 1);
  CHECK(core_at(p, 15, 0) == 1);
  CHECK(core_at(p, 8, 16) == 0);
  CHECK(core_at(p, 5, 3) == 0);

  // Le bilan : la région trop large et ce qui la remplissait, puis ce que faisait la caméra.
  const std::string line = t.summary();
  CHECK(line.contains("1 région trop large (100.0 % des tuiles en cause, 100.0 % de région, caméra 100 %)"));
  CHECK(line.contains("1 caméra sans décalage entier"));
  CHECK(
      line.ends_with(" ; caméra de la carte sur 16 images : 14 tenues (sous 0.25 pixel partout), 1 en partie "
                     "(zoom, rotation), 1 qui bougent partout (médiane au centre 0.625 pixel)"));
}

}  // namespace kirari::app::play
