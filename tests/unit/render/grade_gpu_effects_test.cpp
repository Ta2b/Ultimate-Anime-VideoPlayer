// Éclats HDR sur les seuls effets (render/effect_mask.h) sur le GPU : la passe du masque devant la passe
// anime. Sans masque, ou là où il vaut 1, l'image est celle d'aujourd'hui ; là où il vaut 0, une lumière ne
// garde que fx_rest de sa montée, et les surfaces ne bougent pas.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "grade_chain_support.h"
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

// La vue du masque rendue sur une sortie de side × side (le cadre crop, la mise à l'échelle filter) : le
// pixel de la ligne du milieu à 6 % du cadre, en nits.
Rgb view_pixel(Gpu& gpu, const std::vector<std::uint16_t>& source, int side, const pl_rect2df& crop,
               const pl_filter_config* filter) {
  const pl_gpu g = gpu.vk->gpu;
  pl_tex_params in_params{};
  in_params.w = kSide;
  in_params.h = kSide;
  in_params.format = pl_find_named_fmt(g, "rgba16");
  in_params.sampleable = true;
  in_params.host_writable = true;
  in_params.initial_data = source.data();
  pl_tex in = pl_tex_create(g, &in_params);
  pl_tex_params out_params{};
  out_params.w = side;
  out_params.h = side;
  out_params.format = in_params.format;
  out_params.renderable = true;
  out_params.host_readable = true;
  pl_tex out = pl_tex_create(g, &out_params);
  REQUIRE(in != nullptr);
  REQUIRE(out != nullptr);
  pl_render_params render = pl_render_default_params;
  render.dither_params = nullptr;
  render.upscaler = filter;
  render.downscaler = filter;
  const std::span<const pl_hook* const> hooks = gpu.grade->hooks();
  render.hooks = hooks.data();
  render.num_hooks = static_cast<int>(hooks.size());
  const pl_frame image = frame_of(in, false);
  pl_frame target = frame_of(out, false);
  target.crop = crop;
  REQUIRE(pl_render_image(gpu.renderer, &image, &target, &render));
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(side) * static_cast<std::size_t>(side) * 4);
  pl_tex_transfer_params download{};
  download.tex = out;
  download.ptr = pixels.data();
  REQUIRE(pl_tex_download(g, &download));
  pl_tex_destroy(g, &in);
  pl_tex_destroy(g, &out);
  const auto x = static_cast<std::size_t>(crop.x0 + (0.06F * (crop.x1 - crop.x0)));
  const std::size_t at = ((static_cast<std::size_t>(side / 2) * static_cast<std::size_t>(side)) + x) * 4;
  return {pq_to_nits(pixels[at] / 65535.0), pq_to_nits(pixels[at + 1] / 65535.0),
          pq_to_nits(pixels[at + 2] / 65535.0)};
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

TEST_CASE("Lueurs guidées : le masque précédent garde sa valeur ailleurs et revient au retrait du guide",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const auto source = halo_and_glow();
  GradeParams p = params();
  p.fx_view = 1.0;
  const EffectMask legacy = sure_on({0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F});
  REQUIRE(gpu.grade->set_effect_mask(&legacy));
  const auto before = graded_rgb(gpu, source, p);
  std::array<float, 16> scores{};
  const render::EffectGuide guide{.cols = 8, .rows = 2, .scores = scores};
  scores[7] = scores[15] = 0.35F;
  REQUIRE(gpu.grade->set_effect_mask(&legacy, guide));
  const auto together = graded_rgb(gpu, source, p);
  CHECK(together[240][1] > before[240][1] + 0.01);
  CHECK(together[128][1] == Catch::Approx(before[128][1]).epsilon(1e-3));
  CHECK(together[16][1] == Catch::Approx(before[16][1]).epsilon(1e-3));
  REQUIRE(gpu.grade->set_effect_mask(&legacy));
  const auto after = graded_rgb(gpu, source, p);
  CHECK(after[240][1] == Catch::Approx(before[240][1]).epsilon(1e-3));
  // L'autre case du menu est éteinte par défaut : la grille doit agir seule, avec l'alpha neutre de base.
  REQUIRE_FALSE(gpu.grade->set_effect_mask(nullptr));
  const auto bare = graded_rgb(gpu, source, p);
  REQUIRE(gpu.grade->set_effect_mask(nullptr, guide));
  const auto guide_only = graded_rgb(gpu, source, p);
  CHECK(guide_only[240][1] > bare[240][1] + 0.01);
  CHECK(guide_only[128][1] == Catch::Approx(bare[128][1]).epsilon(1e-3));
  CHECK(guide_only[16][1] == Catch::Approx(bare[16][1]).epsilon(1e-3));
  REQUIRE_FALSE(gpu.grade->set_effect_mask(nullptr));
  const auto bare_again = graded_rgb(gpu, source, p);
  CHECK(bare_again[240][1] == Catch::Approx(bare[240][1]).epsilon(1e-3));
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

// Essai du PC (#943) : la vue sortait verte, sans trace du masque. libplacebo avait écarté la passe du masque
// après la première image (effect_pass.cpp) ; l'alpha restait à 1, lu « effet sûr partout ». Sous la pile du
// lecteur (la LUT de base, le contraste, les éclats, « skin fix ») et image après image, la vue se lit.
TEST_CASE("Éclats des effets : la vue du masque se lit sous la pile de LUT du lecteur, image après image",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const Pile pile;
  set_player_output(gpu, pile, kPeak);
  GradeParams p = player_params(kPeak);
  p.fx_view = 1.0;
  const std::vector<std::uint16_t> source = halo();
  // Sans masque : le fond gris (aucune teinte), le halo rouge.
  const std::vector<Rgb> bare = graded_rgb(gpu, source, p);
  CHECK(bare[16][1] == Catch::Approx(bare[16][0]).epsilon(0.05));
  CHECK(bare[16][2] == Catch::Approx(bare[16][0]).epsilon(0.05));
  CHECK(bare[128][0] > 2.0 * bare[128][1]);
  // Un effet sûr sur la moitié droite : le fond y sort vert, la moitié gauche reste grise.
  const EffectMask effect = sure_on({0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F});
  REQUIRE(gpu.grade->set_effect_mask(&effect));
  const std::vector<Rgb> half = graded_rgb(gpu, source, p);
  CHECK(half[16][1] == Catch::Approx(half[16][0]).epsilon(0.05));
  CHECK(half[240][1] > 2.0 * half[240][0]);
  CHECK(half[240][1] > 2.0 * half[240][2]);
  // La marge retenue partout (le cas du PC : 49 %) : le fond bleu, pas vert.
  const EffectMask still = mask_of(std::vector<float>(8, 0.0F));
  REQUIRE(gpu.grade->set_effect_mask(&still));
  for (int frame = 0; frame < 2; ++frame) {
    const std::vector<Rgb> held = graded_rgb(gpu, source, p);
    CHECK(held[16][2] > 2.0 * held[16][0]);
    CHECK(held[16][2] > 2.0 * held[16][1]);
  }
  gpu.grade->set_effect_mask(nullptr);
  // Hors de la vue, la pile reste : elle change le halo.
  p.fx_view = 0.0;
  const std::vector<Rgb> with_pile = graded_rgb(gpu, source, p);
  GradeParams flat = p;
  flat.look_mixes.fill(0.0);
  const std::vector<Rgb> without = graded_rgb(gpu, source, flat);
  CHECK(std::abs(with_pile[128][1] - without[128][1]) > 0.02 * without[128][1]);
}

// La passe du masque tient image après image sur chaque chemin de l'écran et du grade avant le warp : à
// taille égale, réduite, agrandie au bilinéaire, en bandes noires. libplacebo l'écartait après la première
// image partout où l'image arrivait sans passe de mise à l'échelle à part (seuls spline36 et EWA
// agrandissant y échappaient).
TEST_CASE("Éclats des effets : la passe du masque tient image après image, à toute échelle",
          "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const EffectMask still = mask_of(std::vector<float>(8, 0.0F));
  REQUIRE(gpu.grade->set_effect_mask(&still));
  GradeParams p = params();
  p.fx_view = 1.0;
  gpu.grade->set_params(p);
  const std::vector<std::uint16_t> source = halo();
  struct Output {
    int side;
    pl_rect2df crop;
    const pl_filter_config* filter;
  };
  constexpr float kSmall = kSide / 2.0F;
  constexpr float kLarge = kSide * 2.0F;
  const std::array outputs{Output{kSide, {0, 0, kSide, kSide}, &pl_filter_spline36},
                           Output{kSide / 2, {0, 0, kSmall, kSmall}, &pl_filter_mitchell},
                           Output{kSide * 2, {0, 0, kLarge, kLarge}, &pl_filter_bilinear},
                           Output{kSide * 2, {64, 64, kLarge - 64, kLarge - 64}, &pl_filter_bilinear}};
  for (const Output& out : outputs) {
    for (int frame = 0; frame < 2; ++frame) {
      // La marge retenue partout : le fond bleu, jamais vert (l'alpha à 1, « effet sûr partout »).
      const Rgb px = view_pixel(gpu, source, out.side, out.crop, out.filter);
      CHECK(px[2] > 2.0 * px[0]);
      CHECK(px[2] > 2.0 * px[1]);
      CHECK(pl_renderer_get_errors(gpu.renderer).errors == PL_RENDER_ERR_NONE);
    }
  }
  gpu.grade->set_effect_mask(nullptr);
}

// Les lueurs vues dans l'image (fx_glow), sans carte : la carte ne voit que ce qui bouge, une corne lumineuse
// ou un halo immobiles n'y sont jamais des effets (essai du PC, Re:Zero à 16:23 et 16:45). Un halo qui domine
// nettement sa zone monte comme un effet sûr ; le fond, et une lumière à peine plus claire que sa zone, ne
// bougent pas ; la vue du masque le peint en vert.
TEST_CASE("Éclats des effets : une lueur vue dans l'image monte sans la carte", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = halo();
  GradeParams p = params();
  const std::vector<Rgb> flat = graded_rgb(gpu, source, p);
  p.fx_glow = 1.0;
  const std::vector<Rgb> glow = graded_rgb(gpu, source, p);
  CHECK(glow[128][0] > 1.05 * flat[128][0]);
  CHECK(glow[16][0] == Catch::Approx(flat[16][0]).epsilon(0.001));
  // Un halo à peine plus clair que sa zone (fond à 120 nits) : pas une lueur, rien ne change.
  std::vector<std::uint16_t> bright = source;
  const auto code = static_cast<std::uint16_t>(std::lround(nits_to_pq(120.0) * 65535.0));
  for (std::size_t i = 0; i < bright.size(); i += 4)
    if (bright[i] < code) bright[i] = bright[i + 1] = bright[i + 2] = code;
  p.fx_glow = 0.0;
  const std::vector<Rgb> dull = graded_rgb(gpu, bright, p);
  p.fx_glow = 1.0;
  const std::vector<Rgb> dull_glow = graded_rgb(gpu, bright, p);
  CHECK(dull_glow[128][0] == Catch::Approx(dull[128][0]).epsilon(0.001));
  // La vue : le halo en jaune (lumière d'effet), le fond sans vert.
  p.fx_view = 1.0;
  const std::vector<Rgb> view = graded_rgb(gpu, source, p);
  CHECK(view[128][1] > 2.0 * view[128][2]);
  CHECK(view[16][1] == Catch::Approx(view[16][0]).epsilon(0.05));
}

}  // namespace kirari::render
