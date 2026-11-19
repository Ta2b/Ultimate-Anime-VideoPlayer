// Éclats HDR sur les seuls effets (render/effect_mask.h) sur le GPU : la passe du masque devant la passe
// anime. Sans masque, ou là où il vaut 1, l'image est celle d'aujourd'hui ; là où il vaut 0, une lumière ne
// garde que fx_rest de sa montée, et les surfaces ne bougent pas.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "grade_gpu_support.h"
#include "render/effect_mask.h"

namespace kirari::render {

using namespace gpu_test;

namespace {

// Un halo clair de 48 colonnes (195 nits) sur un fond sombre (3 nits), comme grade_gpu_anime_test.cpp.
std::vector<std::uint16_t> halo() {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (std::size_t y = 0; y < static_cast<std::size_t>(kSide); ++y) {
    for (std::size_t x = 0; x < static_cast<std::size_t>(kSide); ++x) {
      const double nits = x >= 104 && x < 152 ? 195.0 : 3.0;
      const auto code =
          static_cast<std::uint16_t>(std::lround(std::clamp(nits_to_pq(nits), 0.0, 1.0) * 65535.0));
      std::uint16_t* px = &pixels[(y * static_cast<std::size_t>(kSide) + x) * 4];
      px[0] = px[1] = px[2] = code;
      px[3] = 65535;
    }
  }
  return pixels;
}

GradeParams params() {
  GradeParams p;
  p.peak = kPeak;
  p.shadows = 0.0;
  return p;
}

// Un masque de 8 × 2 tuiles posé d'un coup (coupe), une valeur par colonne de tuiles.
EffectMask mask_of(const std::vector<float>& columns) {
  EffectTiles tiles;
  tiles.cols = static_cast<int>(columns.size());
  tiles.rows = 2;
  tiles.cut = true;
  for (int row = 0; row < tiles.rows; ++row)
    tiles.effect.insert(tiles.effect.end(), columns.begin(), columns.end());
  EffectMask mask;
  mask.update(tiles);
  return mask;
}

double luma_at(const std::vector<std::uint16_t>& image, std::size_t x) {
  const std::size_t at = (static_cast<std::size_t>(kSide / 2) * kSide + x) * 4;
  const double r = pq_to_nits(image[at] / 65535.0);
  const double g = pq_to_nits(image[at + 1] / 65535.0);
  const double b = pq_to_nits(image[at + 2] / 65535.0);
  return 0.2627 * r + 0.6780 * g + 0.0593 * b;
}

// Le plus grand écart de code 16 bits entre deux images (le lissage temporel de la passe de zone garde la
// valeur + 1 en float : une relecture peut bouger d'un code).
int max_gap(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
  int gap = 0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    gap = std::max(gap, std::abs(int{a[i]} - int{b[i]}));
  return a.size() == b.size() ? gap : 65535;
}

// Le halo, et une lueur sous le blanc (90 nits, colonnes 224 à 255, la tuile 7) : une lumière seulement
// sur un effet sûr.
std::vector<std::uint16_t> halo_and_glow() {
  std::vector<std::uint16_t> pixels = halo();
  const auto code = static_cast<std::uint16_t>(std::lround(nits_to_pq(90.0) * 65535.0));
  for (std::size_t y = 0; y < static_cast<std::size_t>(kSide); ++y) {
    for (std::size_t x = 224; x < static_cast<std::size_t>(kSide); ++x) {
      std::uint16_t* px = &pixels[(y * static_cast<std::size_t>(kSide) + x) * 4];
      px[0] = px[1] = px[2] = code;
    }
  }
  return pixels;
}

// Toute la marge gardée (1 partout), l'effet sûr sur les colonnes de tuiles marquées.
EffectMask sure_on(const std::vector<float>& columns) {
  EffectTiles tiles;
  tiles.cols = static_cast<int>(columns.size());
  tiles.rows = 2;
  tiles.cut = true;
  tiles.effect.assign(columns.size() * 2, 1.0F);
  for (int row = 0; row < tiles.rows; ++row)
    tiles.sure.insert(tiles.sure.end(), columns.begin(), columns.end());
  EffectMask mask;
  mask.update(tiles);
  return mask;
}

}  // namespace

TEST_CASE("Éclats des effets : sans masque ou sur un effet, le grade d'aujourd'hui", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = halo();
  const std::vector<std::uint16_t> today = graded_image(gpu, source, params());
  // Un masque neutre n'ajoute pas de passe.
  const EffectMask neutral;
  CHECK_FALSE(gpu.grade->set_effect_mask(&neutral));
  CHECK(gpu.grade->hooks().size() == 3);
  CHECK(max_gap(graded_image(gpu, source, params()), today) <= 2);
  // Effet à partir de la quatrième colonne de tuiles (la troisième suit par dilatation, la deuxième en
  // partie par le flou) : le halo (colonnes 104 à 151, tuiles 3 et 4) garde exactement sa montée au centre
  // des tuiles 3 et 4.
  const EffectMask effect = mask_of({0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F});
  REQUIRE(gpu.grade->set_effect_mask(&effect));
  CHECK(gpu.grade->hooks().size() == 4);
  const std::vector<std::uint16_t> masked = graded_image(gpu, source, params());
  for (const std::size_t x : {112U, 144U, 16U, 240U}) {
    INFO("colonne " << x);
    CHECK(luma_at(masked, x) == Catch::Approx(luma_at(today, x)).epsilon(1e-3));
  }
  // Le masque ôté : l'image d'aujourd'hui (au code près du lissage temporel).
  CHECK_FALSE(gpu.grade->set_effect_mask(nullptr));
  CHECK(gpu.grade->hooks().size() == 3);
  CHECK(max_gap(graded_image(gpu, source, params()), today) <= 2);
}

TEST_CASE("Éclats des effets : hors effet, la lumière garde moins de montée, la surface rien ne change",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = halo();
  const std::vector<std::uint16_t> today = graded_image(gpu, source, params());
  const EffectMask still = mask_of(std::vector<float>(8, 0.0F));
  REQUIRE(gpu.grade->set_effect_mask(&still));
  const std::vector<std::uint16_t> masked = graded_image(gpu, source, params());
  const double before = luma_at(today, 128);
  const double after = luma_at(masked, 128);
  INFO("halo : " << before << " nits sans masque, " << after << " sur un décor sûr");
  CHECK(before > 1.8 * 195.0);
  CHECK(after < 0.85 * before);
  CHECK(after > 1.15 * 195.0);  // jamais sous le relèvement des surfaces
  CHECK(luma_at(masked, 16) == Catch::Approx(luma_at(today, 16)).epsilon(1e-3));
  gpu.grade->set_effect_mask(nullptr);
}

TEST_CASE("Éclats des effets : sur un effet sûr, la lumière monte plus haut, rien ne change ailleurs",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = halo_and_glow();
  const std::vector<std::uint16_t> today = graded_image(gpu, source, params());
  // L'effet sûr sur le halo (tuiles 3 et 4) : le halo monte plus haut ; la lueur (tuile 7) et le fond ne
  // bougent pas.
  const EffectMask halo_only = sure_on({0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F});
  REQUIRE(gpu.grade->set_effect_mask(&halo_only));
  const std::vector<std::uint16_t> boosted = graded_image(gpu, source, params());
  INFO("halo : " << luma_at(today, 128) << " nits, " << luma_at(boosted, 128) << " sur un effet sûr");
  CHECK(luma_at(boosted, 128) > 1.05 * luma_at(today, 128));
  CHECK(luma_at(boosted, 240) == Catch::Approx(luma_at(today, 240)).epsilon(1e-3));
  CHECK(luma_at(boosted, 16) == Catch::Approx(luma_at(today, 16)).epsilon(1e-3));
  // L'effet sûr partout : la lueur, sous la porte d'aujourd'hui, devient une lumière.
  const EffectMask everywhere = sure_on(std::vector<float>(8, 1.0F));
  REQUIRE(gpu.grade->set_effect_mask(&everywhere));
  const std::vector<std::uint16_t> glowing = graded_image(gpu, source, params());
  INFO("lueur : " << luma_at(today, 240) << " nits, " << luma_at(glowing, 240) << " sur un effet sûr");
  CHECK(luma_at(today, 240) < 1.2 * 90.0);  // aujourd'hui, une surface (relèvement 1,15)
  CHECK(luma_at(glowing, 240) > 1.2 * luma_at(today, 240));
  CHECK(luma_at(glowing, 16) == Catch::Approx(luma_at(today, 16)).epsilon(1e-3));
  gpu.grade->set_effect_mask(nullptr);
}

TEST_CASE("Éclats des effets : la vue du masque, lumières rouges, effets sûrs verts, marge retenue bleue",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = halo();
  GradeParams p = params();
  p.fx_view = 1.0;
  // Sans masque : le halo, une lumière, en rouge ; ni vert ni bleu.
  const std::vector<Rgb> bare = graded_rgb(gpu, source, p);
  CHECK(bare[128][0] > 2.0 * bare[128][1]);
  CHECK(bare[128][2] == Catch::Approx(bare[128][1]).epsilon(0.05));
  CHECK(bare[16][2] == Catch::Approx(bare[16][1]).epsilon(0.05));
  // Tout retenu (décor sûr partout) : le halo en magenta, le fond en bleu.
  const EffectMask still = mask_of(std::vector<float>(8, 0.0F));
  REQUIRE(gpu.grade->set_effect_mask(&still));
  const std::vector<Rgb> held = graded_rgb(gpu, source, p);
  CHECK(held[128][0] > 2.0 * held[128][1]);
  CHECK(held[128][2] > 2.0 * held[128][1]);
  CHECK(held[16][2] > 2.0 * held[16][0]);
  // Un effet sûr partout : le fond en vert, le halo en jaune (rouge et vert).
  const EffectMask effect = sure_on(std::vector<float>(8, 1.0F));
  REQUIRE(gpu.grade->set_effect_mask(&effect));
  const std::vector<Rgb> sure = graded_rgb(gpu, source, p);
  CHECK(sure[16][1] > 2.0 * sure[16][2]);
  CHECK(sure[128][0] > 2.0 * sure[128][2]);
  CHECK(sure[128][1] > 2.0 * sure[128][2]);
  gpu.grade->set_effect_mask(nullptr);
}

}  // namespace kirari::render
