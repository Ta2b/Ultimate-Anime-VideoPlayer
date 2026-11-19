// Les tuiles qui bougent (upscale_tiles.h, kernels/spec/span_tiles.h) : la dilatation et les masques du
// moteur, puis la décision sur l'écart des blocs de la paire (bloc qui change et ce qui le voit, débandage,
// dérive lente partout, lueur locale qui s'additionne, voisin d'un recalcul qui garde sa somme, recette de
// l'entrée, sans mesure, mesure sur la copie réduite, première image, limite d'âge, refus, bilan), puis dans
// la reprise du lot A (sorte Tiles, reprise entière gardée au seuil de chaque bloc, reprise imposée, tuiles
// coupées).
#include "app/play/upscale_tiles.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/upscale_reuse.h"
#include "kernels/spec/span_tiles.h"

namespace kirari::app::play {
namespace {

// Une image de travail de 1024 × 512 : tuiles du moteur de 64 × 16 (16 × 32), halo d'une colonne et de quatre
// lignes (SPAN à huit blocs : portée 26 pixels de demi-résolution) ; mesure en blocs de 16 (64 × 32).
constexpr kernels::SpanTileGrid kGrid{.tile_w = 64,
                                      .tile_h = 16,
                                      .cols = 16,
                                      .rows = 32,
                                      .reach_cols = kernels::span_reach_tiles(26, 32),
                                      .reach_rows = kernels::span_reach_tiles(26, 8),
                                      .shift_step = 2,
                                      .scale = 2};

// L'écart de chaque bloc d'une paire, en 1/256 de niveau ; rien ne bouge au départ.
struct Blocks {
  std::vector<std::uint32_t> deviation = std::vector<std::uint32_t>(64 * 32, 0);
  float side_px = 16.0F;

  std::uint32_t& at(int c, int r) { return deviation[static_cast<std::size_t>(r * 64 + c)]; }
  PairBlocks pair() const { return PairBlocks{.deviation = deviation, .cols = 64, .side_px = side_px}; }
  TileFacts facts() const { return TileFacts{.stale = TileStale::None, .blocks = pair()}; }
};

std::uint8_t core_at(const TilePlan& p, int c, int r) {
  return p.core[static_cast<std::size_t>(r * kGrid.cols + c)];
}

// Le plan d'une image qui doit passer par tuiles, et son nombre de tuiles recalculées.
int tiled_core(SpanTiles& t, const TileFacts& facts, TilePlan* out = nullptr) {
  TilePlan p = t.plan(facts);
  REQUIRE(p.tiled);
  CHECK(p.shift_x == 0);
  CHECK(p.shift_y == 0);
  const int count = kernels::count_span_tiles(p.core);
  if (out != nullptr) *out = std::move(p);
  return count;
}

}  // namespace

TEST_CASE("Tuiles qui bougent : masques du moteur", "[app][tiles]") {
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
}

TEST_CASE("Tuiles qui bougent : décision sur l'écart des blocs", "[app][tiles]") {
  SpanTiles t;
  t.set_grid(kGrid);
  REQUIRE(t.active());
  Blocks blocks;

  // La première image se calcule entière (l'état ne suit encore rien) ; rien ne bouge ensuite : par tuiles,
  // aucune à recalculer.
  CHECK_FALSE(t.plan(blocks.facts()).tiled);
  CHECK(tiled_core(t, blocks.facts()) == 0);

  // Une bouche qui bouge (15 niveaux) dans le bloc (10, 13), pixels 160 à 176 et 208 à 224 : la tuile (2, 13)
  // du moteur, et toute sortie qui la voit, à une colonne et quatre lignes : 3 × 9 tuiles.
  blocks.at(10, 13) = 15 * 256;
  TilePlan p;
  CHECK(tiled_core(t, blocks.facts(), &p) == 27);
  CHECK(core_at(p, 2, 13) == 1);
  CHECK(core_at(p, 1, 9) == 1);
  CHECK(core_at(p, 3, 17) == 1);
  CHECK(core_at(p, 0, 13) == 0);
  CHECK(core_at(p, 2, 18) == 0);
  // Débandée, l'entrée voit la source jusqu'à 16 pixels de plus : une colonne et une ligne de plus.
  TileFacts debanded = blocks.facts();
  debanded.spread_px = 16;
  CHECK(tiled_core(t, debanded) == 5 * 11);
  // Le bloc s'arrête : ses tuiles et tout ce qui les voit viennent d'être recalculés, il repart de zéro.
  blocks.at(10, 13) = 0;
  CHECK(tiled_core(t, blocks.facts()) == 0);

  // Une image saine mais pas fraîche (saut, image abîmée derrière) : entière.
  TileFacts restart = blocks.facts();
  restart.stale = TileStale::Restart;
  CHECK_FALSE(t.plan(restart).tiled);

  // Dérive lente partout (nuages, fondu très lent) : 0,30 niveau par paire, sous le seuil ; à la troisième,
  // la somme passe 0,70 niveau partout : la région dépasse kMaxShare, l'image est entière.
  for (std::uint32_t& d : blocks.deviation) d = 77;
  CHECK(tiled_core(t, blocks.facts()) == 0);
  CHECK(tiled_core(t, blocks.facts()) == 0);
  CHECK_FALSE(t.plan(blocks.facts()).tiled);
  for (std::uint32_t& d : blocks.deviation) d = 0;

  // Lueur locale : 0,5 niveau par paire sur un bloc, sous le seuil ; à la deuxième, la somme le passe.
  blocks.at(10, 13) = 128;
  CHECK(tiled_core(t, blocks.facts()) == 0);
  CHECK(tiled_core(t, blocks.facts(), &p) == 27);
  CHECK(core_at(p, 2, 13) == 1);
  blocks.at(10, 13) = 0;

  // Un voisin dans le cœur d'un autre garde sa somme : le bloc (10, 17), sur la tuile (2, 17), reçoit 0,4
  // niveau par paire. La première fois, la bouche du bloc (10, 13) recalcule sa tuile, mais pas la tuile
  // (2, 21), qui le voit à quatre lignes et garde sa sortie de son entrée d'avant. La deuxième, la somme
  // passe le seuil : tout ce qui voit le bloc se recalcule, jusqu'à la tuile (2, 21).
  blocks.at(10, 13) = 15 * 256;
  blocks.at(10, 17) = 102;
  CHECK(tiled_core(t, blocks.facts(), &p) == 27);
  CHECK(core_at(p, 2, 17) == 1);
  CHECK(core_at(p, 2, 21) == 0);
  blocks.at(10, 13) = 0;
  CHECK(tiled_core(t, blocks.facts(), &p) == 27);
  CHECK(core_at(p, 2, 21) == 1);
  CHECK(core_at(p, 2, 12) == 0);
  blocks.at(10, 17) = 0;

  // Une autre recette de l'entrée (débandage allumé en cours de lecture) : entière, puis par tuiles ; le
  // retour à la première aussi.
  TileFacts other = blocks.facts();
  other.input_key = 7;
  CHECK_FALSE(t.plan(other).tiled);
  CHECK(t.plan(other).tiled);
  CHECK_FALSE(t.plan(blocks.facts()).tiled);

  // Une paire sans mesure : entière ; la suivante, mesurée à la même géométrie, par tuiles, avec un peu de
  // grain (une somme non nulle).
  CHECK_FALSE(t.plan(TileFacts{.stale = TileStale::None}).tiled);
  blocks.at(0, 0) = 50;
  CHECK(tiled_core(t, blocks.facts()) == 0);
  blocks.at(0, 0) = 0;

  // La mesure sur la copie réduite de RIFE (720p d'un travail 1080p) : un bloc couvre 24 pixels. Une autre
  // géométrie sous une somme non nulle : entière, puis le même bloc en 240 à 264 et 312 à 336, sur les tuiles
  // (3, 19) à (4, 20).
  Blocks small;
  small.side_px = 24.0F;
  CHECK_FALSE(t.plan(small.facts()).tiled);
  small.at(10, 13) = 15 * 256;
  CHECK(tiled_core(t, small.facts(), &p) == 4 * 10);
  CHECK(core_at(p, 2, 13) == 0);
  CHECK(core_at(p, 4, 20) == 1);
  CHECK(core_at(p, 3, 19) == 1);
}

TEST_CASE("Tuiles qui bougent : limite d'âge, refus et bilan", "[app][tiles]") {
  Blocks blocks;
  // La première image n'a pas de paire : sans mesure, entière ; la suivante prend la géométrie de la mesure
  // (toutes les sommes sont nulles) et passe par tuiles.
  SpanTiles first;
  first.set_grid(kGrid);
  CHECK_FALSE(first.plan(TileFacts{.stale = TileStale::None}).tiled);
  CHECK(first.plan(blocks.facts()).tiled);

  // Limite d'âge : une image fixe se reprend au plus kMaxAge fois par tuile, puis tout se recalcule.
  SpanTiles t;
  t.set_grid(kGrid);
  CHECK_FALSE(t.plan(blocks.facts()).tiled);
  int tiled = 0;
  for (int i = 0; i < SpanTiles::kMaxAge + 1; ++i) tiled += t.plan(blocks.facts()).tiled ? 1 : 0;
  CHECK(tiled == SpanTiles::kMaxAge);
  CHECK(
      t.summary().contains("1 région trop large (100.0 % des tuiles en cause, 100.0 % de région, blocs 0 %, "
                           "âge 100 %)"));
  // La reprise entière d'une sortie faite par tuiles vieillit aussi chaque tuile : elle s'arrête avant
  // qu'une tuile passe kMaxAge. Après l'image par tuiles, les tuiles hors du cœur ont une image d'âge.
  SpanTiles aged;
  aged.set_grid(kGrid);
  CHECK_FALSE(aged.plan(blocks.facts()).tiled);
  blocks.at(10, 13) = 15 * 256;
  CHECK(tiled_core(aged, blocks.facts()) == 27);
  blocks.at(10, 13) = 0;
  int kept = 0;
  while (kept < 100 && aged.holds(blocks.pair())) {
    aged.held(blocks.pair());
    ++kept;
  }
  CHECK(kept == SpanTiles::kMaxAge - 1);

  // Un refus du moteur : l'image redevient entière, et ne compte plus parmi celles par tuiles. La ligne du
  // bilan sort quand même, avec la raison de chaque image entière.
  SpanTiles once;
  once.set_grid(kGrid);
  CHECK_FALSE(once.plan(blocks.facts()).tiled);
  CHECK(once.plan(blocks.facts()).tiled);
  once.refused();
  TileFacts graded = blocks.facts();
  graded.stale = TileStale::Graded;
  CHECK_FALSE(once.plan(graded).tiled);
  CHECK_FALSE(once.plan(TileFacts{.stale = TileStale::None}).tiled);
  CHECK(once.summary() ==
        "SPAN par tuiles : 0 images sur 4 calculées (0.0 %) ; images entières : 1 sans mesure des blocs, 1 "
        "refus du moteur, 1 sortie d'avant gradée sur place, 1 début, saut ou image abîmée");
  // Rien de calculé : la ligne le dit aussi.
  SpanTiles idle;
  idle.set_grid(kGrid);
  CHECK(idle.summary() == "SPAN par tuiles : 0 images sur 0 calculées (0.0 %) ; images entières : aucune");
  // Un moteur sans tuiles (TensorRT) : jamais actif, et la ligne de fin le dit ; sans SPAN, aucune ligne.
  SpanTiles none;
  none.set_grid(kernels::SpanTileGrid{});
  CHECK_FALSE(none.active());
  CHECK(none.summary() == "SPAN par tuiles : aucune, ce moteur ne calcule que l'image entière");
  CHECK(SpanTiles{}.summary().empty());
}

TEST_CASE("Tuiles qui bougent : dans la reprise du lot A", "[app][tiles]") {
  Blocks blocks;
  SpanReuse r;
  r.tiles.set_grid(kGrid);
  ReuseFacts facts{.same_epoch = true,
                   .has_output = true,
                   .pair_held = false,
                   .tiles_stale = TileStale::None,
                   .blocks = blocks.pair()};
  CHECK(r.decide(facts) == ReuseKind::Compute);  // la première suit l'état
  // Une image non tenue mais lisible par tuiles rend Tiles et le plan.
  blocks.at(10, 13) = 15 * 256;
  CHECK(r.decide(facts) == ReuseKind::Tiles);
  CHECK(r.tile_plan.tiled);
  CHECK(r.tile_plan.core[static_cast<std::size_t>(13 * kGrid.cols + 2)] == 1);
  blocks.at(10, 13) = 0;

  // La sortie mêle maintenant des tuiles d'images différentes. Une paire tenue (0,5 niveau sur un bloc) se
  // reprend entière tant que chaque bloc reste au seuil ; la suivante, l'ancre tenue aussi, porterait ce bloc
  // à 1 niveau depuis le calcul de ses tuiles : elle se calcule, par tuiles.
  blocks.at(40, 20) = 128;
  facts.pair_held = true;
  CHECK(r.decide(facts) == ReuseKind::Whole);
  facts.anchor = AnchorGap{.worst_block = 150};
  CHECK(r.decide(facts) == ReuseKind::Tiles);
  CHECK(r.tile_plan.core[static_cast<std::size_t>(20 * kGrid.cols + 10)] == 1);
  CHECK(r.summary().contains(", arrêts aux tuiles 1"));
  blocks.at(40, 20) = 0;
  facts.anchor.reset();

  // Une reprise imposée (copie d'une tenue fondue, Smooth) ne dit pas l'écart des blocs : l'état ne suit
  // plus, la prochaine image calculée est entière.
  r.force();
  facts.pair_held = false;
  blocks.at(10, 13) = 15 * 256;
  CHECK(r.decide(facts) == ReuseKind::Compute);
  CHECK(r.decide(facts) == ReuseKind::Tiles);

  // Une sortie d'avant écrasée : entière. Un saut l'emporte sur la sortie d'avant au bilan ; une chaîne sans
  // tuiles, sur le saut.
  facts.tiles_stale = TileStale::Overwritten;
  CHECK(r.decide(facts) == ReuseKind::Compute);
  CHECK_FALSE(r.tile_plan.tiled);
  facts.tiles_stale = TileStale::Graded;
  facts.same_epoch = false;
  CHECK(r.decide(facts) == ReuseKind::Compute);
  facts.tiles_stale = TileStale::Chain;
  CHECK(r.decide(facts) == ReuseKind::Compute);
  const std::string line = r.tiles.summary();
  CHECK(line.contains("images entières : 1 sans mesure des blocs, "));  // la reprise imposée
  CHECK(
      line.contains("1 sortie d'avant écrasée (image créée, source sautée), 1 chaîne sans tuiles (export, "
                    "YUV, TemporalFix, Smooth, plaque), 2 début, saut ou image abîmée"));

  // Les tuiles coupées (--no-span-tiles) : chaque image se calcule entière, et le journal de fin ne dit rien
  // des tuiles.
  SpanReuse off;
  off.tiles.enabled = false;
  off.tiles.set_grid(kGrid);
  facts.same_epoch = true;
  facts.tiles_stale = TileStale::None;
  CHECK(off.decide(facts) == ReuseKind::Compute);
  CHECK(off.decide(facts) == ReuseKind::Compute);
  CHECK(off.tiles.summary().empty());
}

}  // namespace kirari::app::play
