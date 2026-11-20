// Le masque des effets (render/effect_mask.h), sans GPU : lissage dans le temps, dilatation, paires tenues,
// coupes, retour au neutre sans carte.
#include "render/effect_mask.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace kirari::render {
namespace {

constexpr std::int64_t kPair = 41'708;  // une image à 23,976 i/s

EffectTiles grid(int cols, int rows, float value) {
  EffectTiles t;
  t.cols = cols;
  t.rows = rows;
  t.effect.assign(static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows), value);
  t.span_us = kPair;
  return t;
}

float centre(const EffectMask& mask) {
  return mask.values()[static_cast<std::size_t>(mask.rows() / 2) * static_cast<std::size_t>(mask.cols()) +
                       static_cast<std::size_t>(mask.cols() / 2)];
}

}  // namespace

TEST_CASE("Masque des effets : neutre au départ, le décor sûr ne l'éteint que lentement", "[render]") {
  EffectMask mask;
  CHECK(mask.neutral());
  CHECK(mask.values().empty());
  mask.update(grid(9, 5, 0.0F));
  CHECK_FALSE(mask.neutral());
  // Une paire : à peine 6 % de moins (constante de 0,7 s), pas de saut visible.
  CHECK(centre(mask) == Catch::Approx(1.0F - effect_step(kPair, kEffectFallUs)).margin(1e-4));
  CHECK(centre(mask) > 0.9F);
  for (int i = 0; i < 72; ++i) mask.update(grid(9, 5, 0.0F));  // trois secondes
  CHECK(centre(mask) < 0.05F);
}

TEST_CASE("Masque des effets : un effet monte en deux paires", "[render]") {
  EffectMask mask;
  for (int i = 0; i < 72; ++i) mask.update(grid(9, 5, 0.0F));
  mask.update(grid(9, 5, 1.0F));
  mask.update(grid(9, 5, 1.0F));
  CHECK(centre(mask) > 0.85F);
}

TEST_CASE("Masque des effets : un effet en deux ne retombe pas une paire sur deux", "[render]") {
  EffectMask mask;
  for (int i = 0; i < 72; ++i) mask.update(grid(9, 5, 0.0F));
  // Un effet animé en deux : la paire qui bouge le voit, la paire tenue n'apprend rien.
  float lowest = 1.0F;
  for (int i = 0; i < 48; ++i) {
    EffectTiles t = grid(9, 5, 1.0F);
    t.hold = i % 2 == 1;
    if (t.hold) t.effect.assign(t.effect.size(), 0.0F);
    const std::uint64_t serial = mask.serial();
    mask.update(t);
    if (t.hold) CHECK(mask.serial() == serial);
    if (i > 10) lowest = std::min(lowest, centre(mask));
  }
  CHECK(lowest > 0.99F);
  lowest = 1.0F;
  // Et un effet qui hésite d'une paire à l'autre (verdict effet, puis décor) reste haut.
  for (int i = 0; i < 48; ++i) {
    mask.update(grid(9, 5, i % 2 == 0 ? 1.0F : 0.0F));
    if (i > 4) lowest = std::min(lowest, centre(mask));
  }
  CHECK(lowest > 0.88F);  // l'équilibre : 0,91
}

TEST_CASE("Masque des effets : une tenue sur une nouvelle grille repart du neutre", "[render]") {
  EffectMask mask;
  EffectTiles initial = grid(9, 5, 1.0F);
  initial.sure = initial.effect;
  mask.update(initial);
  REQUIRE_FALSE(mask.neutral());
  const auto before = mask.serial();
  EffectTiles held = grid(5, 9, 0.0F);  // même nombre de tuiles, autre grille
  held.hold = true;
  held.sure.assign(held.effect.size(), 1.0F);
  mask.update(held);
  CHECK(mask.cols() == 5);
  CHECK(mask.rows() == 9);
  CHECK(mask.values().size() == held.effect.size());
  CHECK(mask.sure().size() == held.sure.size());
  CHECK(mask.neutral());
  CHECK(mask.serial() != before);
  const auto resized = mask.serial();
  held.cover_x = 1.125;
  mask.update(held);
  CHECK(mask.cover_x() == held.cover_x);
  CHECK(mask.serial() != resized);
  const auto moved = mask.serial();
  mask.update(held);
  CHECK(mask.serial() == moved);
}

TEST_CASE("Masque des effets : le halo d'un effet déborde d'une tuile, pas plus", "[render]") {
  EffectMask mask;
  for (int i = 0; i < 72; ++i) {
    EffectTiles t = grid(9, 1, 0.0F);
    t.effect[4] = 1.0F;
    mask.update(t);
  }
  const std::span<const float> v = mask.values();
  CHECK(v[4] > 0.95F);
  CHECK(v[3] > 0.7F);  // la voisine, dilatée puis floutée
  CHECK(v[5] > 0.7F);
  CHECK(v[1] < 0.1F);
  CHECK(v[7] < 0.1F);
}

TEST_CASE("Masque des effets : une tuile sans verdict ne fait que suivre un effet voisin", "[render]") {
  EffectMask mask;
  for (int i = 0; i < 72; ++i) mask.update(grid(3, 1, 0.0F));
  const float before = mask.values()[1];
  EffectTiles unsure = grid(3, 1, kEffectUnsure);
  mask.update(unsure);
  CHECK(mask.values()[1] == Catch::Approx(before));
  unsure.effect[0] = 1.0F;
  mask.update(unsure);
  mask.update(unsure);
  CHECK(mask.values()[1] > 0.5F);
}

TEST_CASE("Masque des effets : une coupe pose le masque d'un coup", "[render]") {
  EffectMask mask;
  EffectTiles cut = grid(9, 5, 0.0F);
  cut.cut = true;
  mask.update(cut);
  CHECK(centre(mask) == 0.0F);
  // Une coupe vers un plan sans verdict repart du neutre.
  EffectTiles unsure = grid(9, 5, kEffectUnsure);
  unsure.cut = true;
  mask.update(unsure);
  CHECK(mask.neutral());
}

TEST_CASE("Masque des effets : une coupe sans carte du nouveau plan remet le masque au neutre", "[render]") {
  EffectMask mask;
  EffectTiles cut = grid(9, 5, 0.0F);
  cut.cut = true;
  mask.update(cut);
  REQUIRE_FALSE(mask.neutral());
  const std::uint64_t before = mask.serial();
  EffectTiles blind;  // pas de tuiles : la carte de la paire n'est pas là
  blind.cut = true;
  blind.span_us = kPair;
  mask.update(blind);
  CHECK(mask.neutral());
  CHECK(mask.values().empty());
  CHECK(mask.serial() != before);
  // Sans coupe, la même note vide ne fait que relâcher.
  mask.update(cut);
  blind.cut = false;
  mask.update(blind);
  CHECK_FALSE(mask.neutral());
  CHECK(centre(mask) == Catch::Approx(effect_step(kPair, kEffectFallUs)).margin(1e-4));
}

TEST_CASE("Masque des effets : sans carte, retour lent au neutre, puis plus rien", "[render]") {
  EffectMask mask;
  EffectTiles cut = grid(9, 5, 0.0F);
  cut.cut = true;
  mask.update(cut);
  mask.relax(kPair);
  CHECK(centre(mask) == Catch::Approx(effect_step(kPair, kEffectFallUs)).margin(1e-4));
  for (int i = 0; i < 400 && !mask.neutral(); ++i) mask.relax(kPair);
  CHECK(mask.neutral());
  CHECK(mask.values().empty());
  // Une grille vide ou mal formée vaut une paire sans carte.
  EffectTiles bad = grid(9, 5, 0.0F);
  bad.effect.pop_back();
  mask.update(bad);
  CHECK(mask.neutral());
}

TEST_CASE("Masque des effets : la couche des effets sûrs monte vite, retombe lentement, ne garde rien seule",
          "[render]") {
  EffectMask mask;
  EffectTiles t = grid(9, 5, 1.0F);
  t.sure.assign(t.effect.size(), 0.0F);
  t.sure[(2 * 9) + 4] = 1.0F;  // un effet sûr au centre
  mask.update(t);
  // Pas une coupe : en deux paires, l'effet sûr est presque plein au centre ; la marge reste entière.
  mask.update(t);
  REQUIRE(mask.sure().size() == mask.values().size());
  CHECK(mask.sure()[(2 * 9) + 4] > 0.6F);
  CHECK(mask.sure()[0] == 0.0F);
  CHECK(centre(mask) == Catch::Approx(1.0F));
  CHECK_FALSE(mask.neutral());  // tout gardé, mais un effet sûr : la passe reste
  // L'effet parti : lentement vers 0, puis le masque redevient neutre.
  t.sure.assign(t.effect.size(), 0.0F);
  mask.update(t);
  CHECK(mask.sure()[(2 * 9) + 4] > 0.5F);
  for (int i = 0; i < 400 && !mask.neutral(); ++i) mask.update(t);
  CHECK(mask.neutral());
  // Une coupe pose la couche d'un coup ; sans carte, elle retombe comme le reste revient à 1.
  t.sure[(2 * 9) + 4] = 1.0F;
  t.cut = true;
  mask.update(t);
  CHECK(mask.sure()[(2 * 9) + 4] == Catch::Approx(1.0F));
  mask.relax(kPair);
  CHECK(mask.sure()[(2 * 9) + 4] == Catch::Approx(1.0F - effect_step(kPair, kEffectFallUs)).margin(1e-4));
  // Une carte sans couche (avant elle) : aucun effet sûr.
  EffectMask bare;
  bare.update(grid(9, 5, 0.0F));
  CHECK(std::ranges::all_of(bare.sure(), [](float v) { return v == 0.0F; }));
}

}  // namespace kirari::render
