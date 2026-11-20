// Éclats HDR sur les seuls effets, côté lecteur (app/play/effect_book.h) : la carte de la scène réduite pour
// le masque, les notes par paire, et le masque que suivent l'écran et le grade avant le warp.
#include "app/play/effect_book.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <catch2/catch_test_macros.hpp>

#include "core/frame.h"

namespace kirari::app::play {
namespace {

kernels::SceneMotion scene(int cols, int rows, kernels::SceneClass kind, float confidence) {
  kernels::SceneMotion s;
  s.tile_px = 32;
  s.cols = cols;
  s.rows = rows;
  s.tiles.assign(static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows),
                 kernels::SceneTile{.kind = kind, .confidence = confidence});
  return s;
}

EffectNote note(std::int64_t from_us, std::int64_t to_us, float value, bool cut = false) {
  render::EffectTiles t;
  t.cols = 4;
  t.rows = 2;
  t.effect.assign(8, value);
  t.cut = cut;
  t.span_us = to_us - from_us;
  return EffectNote{.epoch = 1,
                    .from_us = from_us,
                    .to_us = to_us,
                    .tiles = std::make_shared<const render::EffectTiles>(t)};
}

}  // namespace

TEST_CASE("Éclats des effets : la carte de la scène devient une valeur d'effet par tuile", "[app]") {
  kernels::SceneMotion s = scene(60, 34, kernels::SceneClass::Drawing, 0.8F);
  s.tiles[0].kind = kernels::SceneClass::Effect;
  s.tiles[0].confidence = 0.1F;
  s.tiles[1].kind = kernels::SceneClass::Uncertain;  // aplat qui ne change pas : un ciel, une feuille
  s.tiles[2].kind = kernels::SceneClass::Uncertain;
  s.tiles[2].changed = true;                          // aplat qui change : pas de verdict
  s.tiles[4].kind = kernels::SceneClass::Background;  // décor sûr : une bougie immobile en est un
  s.tiles[4].confidence = 1.0F;
  const render::EffectTiles t = effect_tiles_of(s, 1920, 1080, 41'708, false);
  REQUIRE(t.cols == 60);
  REQUIRE(t.rows == 34);
  // La marge entière partout, dessin sûr compris : une bougie immobile sortait en décor, puis en dessin (son
  // contour net sur le noir), et perdait 60 % de sa montée.
  for (std::size_t i = 0; i < 5; ++i) CHECK(t.effect[i] == 1.0F);
  // La couche des effets sûrs : la seule tuile d'effet, quelle que soit sa confiance.
  REQUIRE(t.sure.size() == t.effect.size());
  CHECK(t.sure[0] == 1.0F);
  CHECK(t.sure[1] == 0.0F);
  CHECK(t.sure[2] == 0.0F);
  CHECK(t.sure[3] == 0.0F);
  CHECK(t.cover_x == 1.0);
  CHECK(t.cover_y > 1.0);  // 34 × 32 = 1 088 lignes pour 1 080
  CHECK_FALSE(t.cut);
  CHECK_FALSE(t.hold);
  s.drawing_held = true;
  CHECK(effect_tiles_of(s, 1920, 1080, 41'708, false).hold);
  s.drawing_held = false;
  s.fade = true;
  CHECK(effect_tiles_of(s, 1920, 1080, 41'708, false).hold);
  // Une coupe ne donne aucune tuile : la paire enjambe deux plans (NVOF y voyait des effets dans un fond
  // sombre), le masque repart du neutre.
  s.cut_likely = true;
  const render::EffectTiles likely = effect_tiles_of(s, 1920, 1080, 41'708, false);
  CHECK(likely.cut);
  CHECK(likely.effect.empty());
  CHECK(likely.sure.empty());
  const render::EffectTiles cut =
      effect_tiles_of(scene(4, 2, kernels::SceneClass::Effect, 1.0F), 1920, 1080, 0, true);
  CHECK(cut.cut);
  CHECK(cut.effect.empty());
  // Une carte mal formée ne donne rien (le masque revient alors au neutre).
  kernels::SceneMotion bad = scene(4, 2, kernels::SceneClass::Effect, 1.0F);
  bad.tiles.pop_back();
  CHECK(effect_tiles_of(bad, 1920, 1080, 0, false).effect.empty());
}

TEST_CASE("Éclats des effets : la note de la paire de l'image montrée", "[app]") {
  EffectBook book;
  CHECK_FALSE(book.at(1, 100));
  book.add(note(0, 100, 1.0F));
  book.add(note(100, 200, 0.0F));
  REQUIRE(book.at(1, 150));
  CHECK(book.at(1, 150)->from_us == 100);
  CHECK(book.at(1, 100)->from_us == 0);  // b de la première paire
  CHECK_FALSE(book.at(1, 0));
  CHECK_FALSE(book.at(2, 150));  // autre époque
  book.clear();
  CHECK_FALSE(book.at(1, 150));
}

TEST_CASE("Éclats des effets : l'écran suit une fois par paire, puis revient au neutre sans carte", "[app]") {
  EffectFollower follower;
  CHECK_FALSE(follower.show(std::nullopt, 0));  // neutre, et le reste
  CHECK(follower.mask().neutral());
  const std::optional<EffectNote> first = note(0, 41'708, 0.0F, true);
  CHECK(follower.show(first, 41'708));
  CHECK_FALSE(follower.mask().neutral());
  // Les images interpolées et représentées de la même paire ne la relisent pas.
  CHECK_FALSE(follower.show(first, 41'708));
  CHECK_FALSE(follower.show(first, 20'000));
  // Sans carte : un pas par image montrée, au temps écoulé ; une représentation ne compte pas.
  CHECK(follower.show(std::nullopt, 83'416));
  CHECK_FALSE(follower.show(std::nullopt, 83'416));
  for (std::int64_t pts = 125'124; !follower.mask().neutral() && pts < 60'000'000; pts += 41'708)
    follower.show(std::nullopt, pts);
  CHECK(follower.mask().neutral());
}

TEST_CASE("Éclats des effets : une coupe sans tuiles signale la fin du masque après un saut", "[app]") {
  EffectFollower follower;
  REQUIRE(follower.show(note(0, 41'708, 0.0F, true), 41'708));
  REQUIRE_FALSE(follower.mask().neutral());
  render::EffectTiles tiles;
  tiles.cut = true;
  const EffectNote next{.epoch = 2,
                        .from_us = 1'000'000,
                        .to_us = 1'041'708,
                        .tiles = std::make_shared<const render::EffectTiles>(tiles)};
  CHECK(follower.show(next, next.to_us));
  CHECK(follower.mask().neutral());
  CHECK_FALSE(follower.show(next, next.to_us));
}

TEST_CASE("Éclats des effets : après un saut, une image sans carte ne garde pas l'ancien masque", "[app]") {
  const auto old = note(0, 41'708, 0.0F, true);
  EffectFollower follower;
  REQUIRE(follower.show(old, old.to_us, 1));
  REQUIRE_FALSE(follower.mask().neutral());
  CHECK(follower.show(std::nullopt, 25'000, 2));  // PTS réutilisé, mais nouvelle époque
  CHECK(follower.mask().neutral());
  CHECK_FALSE(follower.show(old, 25'000, 2));  // une note de l'ancienne époque ne le réactive pas
  CHECK(follower.mask().neutral());
  auto current = old;
  current.epoch = 2;
  CHECK(follower.show(current, 30'000, 2));
  CHECK_FALSE(follower.mask().neutral());
}

TEST_CASE("Éclats des effets : la source gradée prend la note la plus récente de son époque", "[app]") {
  EffectBook book;
  CHECK_FALSE(book.for_source(1, 100));
  book.add(note(0, 100, 1.0F));
  book.add(note(100, 200, 0.0F));
  REQUIRE(book.for_source(1, 200));
  CHECK(book.for_source(1, 200)->to_us == 200);
  // Agrandie en avance, la source suivante prend encore la paire en cours.
  CHECK(book.for_source(1, 241)->to_us == 200);
  CHECK(book.for_source(1, kNoPts)->to_us == 200);
  CHECK_FALSE(book.for_source(2, 200));  // autre époque : rien
  // La carte s'est tue : rien, le masque revient au neutre.
  CHECK_FALSE(book.for_source(1, 200 + kEffectStaleUs + 1));
}

TEST_CASE("Éclats des effets : le grade avant le warp suit une fois par paire, une époque à la fois",
          "[app]") {
  EffectFollower follower;
  CHECK_FALSE(follower.source(1, std::nullopt));
  const std::optional<EffectNote> first = note(0, 41'708, 0.0F, true);
  CHECK(follower.source(1, first));
  CHECK_FALSE(follower.mask().neutral());
  CHECK_FALSE(follower.source(1, first));  // la même paire pour la source agrandie en avance
  // Sans note : un pas d'une image vers le neutre par source.
  CHECK(follower.source(1, std::nullopt));
  for (int i = 0; i < 400 && !follower.mask().neutral(); ++i) follower.source(1, std::nullopt);
  CHECK(follower.mask().neutral());
  // Une autre époque (saut) efface le masque d'avant, même sans note.
  CHECK(follower.source(1, first));
  CHECK(follower.source(2, std::nullopt));
  CHECK(follower.mask().neutral());
  // La note d'une autre époque que la source ne vaut pas.
  CHECK_FALSE(follower.source(2, first));
  CHECK(follower.mask().neutral());
}

TEST_CASE("Éclats des effets : le journal de fin compte les images et garde les moments marqués", "[app]") {
  EffectTally tally;
  render::EffectMask none;
  tally.count(false, false, none, 0);  // sans carte
  CHECK(tally.images() == 1);
  CHECK(tally.noted() == 0);
  // Un masque posé d'un coup : moitié effet sûr (1), moitié décor sûr (0), dilaté puis flouté.
  const auto mask_of = [](float left, float right) {
    render::EffectTiles t;
    t.cols = 8;
    t.rows = 2;
    t.cut = true;
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 8; ++c) t.effect.push_back(c < 4 ? left : right);
    t.sure = t.effect;
    render::EffectMask m;
    m.update(t);
    return m;
  };
  const render::EffectMask half = mask_of(1.0F, 0.0F);
  const render::EffectMask still = mask_of(0.0F, 0.0F);  // aucun effet : jamais un moment marqué
  tally.count(true, true, half, 5'000'000);
  tally.count(true, true, still, 60'000'000);
  tally.count(true, true, half, 8'000'000);  // à moins de 10 s du premier, pas plus fort : ignoré
  tally.count(true, true, half, 30'000'000);
  tally.count(true, false, half, 40'000'000);  // masque non posé (GPU refusé) : pas compté
  CHECK(tally.images() == 6);
  CHECK(tally.noted() == 5);
  CHECK(tally.masked() == 4);
  CHECK(tally.held() > 0.2);
  CHECK(tally.held() < 1.0);
  CHECK(tally.moments()[0].pts_us == 5'000'000);
  CHECK(tally.moments()[0].effect > 0.4);  // la moitié gauche et sa dilatation
  CHECK(tally.moments()[0].held > 0.0);
  CHECK(tally.moments()[1].pts_us == 30'000'000);
  CHECK(tally.moments()[2].effect == 0.0);  // la troisième place reste vide
  // Le résumé d'une image : ce que le journal d'un instantané dit.
  const EffectSummary seen = effect_summary(half, true);
  CHECK(seen.on);
  CHECK(seen.cols == 8);
  CHECK(seen.rows == 2);
  CHECK(seen.sure > 0.4);
  CHECK(seen.sure < 0.8);
  CHECK(seen.held > 0.2);
  CHECK_FALSE(effect_summary(half, false).on);
  CHECK_FALSE(effect_summary(none, true).on);
}

}  // namespace kirari::app::play
