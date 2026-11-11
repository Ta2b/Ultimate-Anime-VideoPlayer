// Anime4K (render/anime4k.h) : les shaders embarqués lus par libplacebo et rendus sur une image synthétique.
// Lavapipe suffit ; sans appareil Vulkan, les tests [gpu] se sautent.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <libplacebo/renderer.h>

#include "grade_gpu_support.h"
#include "render/anime4k.h"

namespace ajp::render {
namespace {

using namespace gpu_test;

constexpr int kIn = 128;  // l'image de travail ; Anime4K la double

// Un disque sombre au trait net sur un fond clair, réduit par une moyenne 4 × 4 (une ligne d'anime réduite :
// bord anticrénelé, que l'agrandissement doit garder net).
std::vector<std::uint16_t> disc() {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kIn) * kIn * 4);
  for (int y = 0; y < kIn; ++y) {
    for (int x = 0; x < kIn; ++x) {
      double sum = 0.0;
      for (int sy = 0; sy < 4; ++sy) {
        for (int sx = 0; sx < 4; ++sx) {
          const double dx = x + (sx + 0.5) / 4.0 - kIn / 2.0;
          const double dy = y + (sy + 0.5) / 4.0 - kIn / 2.0;
          sum += std::hypot(dx, dy) < kIn / 3.0 ? 0.15 : 0.8;
        }
      }
      const auto code = static_cast<std::uint16_t>(sum / 16.0 * 65535.0 + 0.5);
      const std::size_t at = (static_cast<std::size_t>(y) * kIn + static_cast<std::size_t>(x)) * 4;
      pixels[at] = pixels[at + 1] = pixels[at + 2] = code;
      pixels[at + 3] = 65535;
    }
  }
  return pixels;
}

// L'image rendue sur une cible de side × side (canal rouge, de 0 à 1), avec ces hooks.
std::vector<double> rendered(Gpu& gpu, const std::vector<std::uint16_t>& source,
                             std::span<const pl_hook* const> hooks, int side) {
  const pl_gpu g = gpu.vk->gpu;
  pl_tex_params tex_params{};
  tex_params.w = kIn;
  tex_params.h = kIn;
  tex_params.format = pl_find_named_fmt(g, "rgba16");
  tex_params.sampleable = true;
  tex_params.host_writable = true;
  tex_params.initial_data = source.data();
  pl_tex in = pl_tex_create(g, &tex_params);
  tex_params.initial_data = nullptr;
  tex_params.sampleable = false;
  tex_params.host_writable = false;
  tex_params.renderable = true;
  tex_params.host_readable = true;
  tex_params.w = side;
  tex_params.h = side;
  pl_tex out = pl_tex_create(g, &tex_params);
  REQUIRE(in != nullptr);
  REQUIRE(out != nullptr);
  pl_render_params render = pl_render_default_params;
  render.dither_params = nullptr;
  render.hooks = hooks.data();
  render.num_hooks = static_cast<int>(hooks.size());
  pl_frame image = frame_of(in, true);
  image.crop = {0, 0, static_cast<float>(kIn), static_cast<float>(kIn)};
  pl_frame target = frame_of(out, true);
  target.crop = {0, 0, static_cast<float>(side), static_cast<float>(side)};
  REQUIRE(pl_render_image(gpu.renderer, &image, &target, &render));
  const auto width = static_cast<std::size_t>(side);
  std::vector<std::uint16_t> pixels(width * width * 4);
  pl_tex_transfer_params download{};
  download.tex = out;
  download.ptr = pixels.data();
  REQUIRE(pl_tex_download(g, &download));
  pl_tex_destroy(g, &in);
  pl_tex_destroy(g, &out);
  std::vector<double> red(width * width);
  for (std::size_t i = 0; i < red.size(); ++i) red[i] = pixels[i * 4] / 65535.0;
  return red;
}

// La pente la plus forte de la ligne du milieu : un bord plus net la rend plus raide.
double steepest(const std::vector<double>& image, int side) {
  const auto row = static_cast<std::size_t>(side / 2) * static_cast<std::size_t>(side);
  double best = 0.0;
  for (std::size_t x = 1; x < static_cast<std::size_t>(side); ++x)
    best = std::max(best, std::abs(image[row + x] - image[row + x - 1]));
  return best;
}

double mean_gap(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += std::abs(a[i] - b[i]);
  return sum / static_cast<double>(a.size());
}

constexpr std::array kChoices{
    Anime4kChoice{.size = Anime4kSize::Fast, .mode = Anime4kMode::A},
    Anime4kChoice{.size = Anime4kSize::Fast, .mode = Anime4kMode::B},
    Anime4kChoice{.size = Anime4kSize::Fast, .mode = Anime4kMode::C},
    Anime4kChoice{.size = Anime4kSize::Quality, .mode = Anime4kMode::A},
    Anime4kChoice{.size = Anime4kSize::Quality, .mode = Anime4kMode::B},
    Anime4kChoice{.size = Anime4kSize::Quality, .mode = Anime4kMode::C},
};

TEST_CASE("Anime4K : chaque jeu de shaders se lit et agrandit net", "[render][gpu][anime4k]") {
  Gpu gpu;
  require_device(gpu);
  const std::vector<std::uint16_t> source = disc();
  const std::vector<double> plain = rendered(gpu, source, {}, kIn * 2);
  for (const Anime4kChoice choice : kChoices) {
    CAPTURE(name(choice.size), name(choice.mode));
    auto shaders = Anime4k::create(gpu.vk->gpu, choice);
    REQUIRE(shaders);
    CHECK(shaders->choice() == choice);
    CHECK(shaders->hooks().size() == (choice.mode == Anime4kMode::C ? 2U : 3U));
    const std::vector<double> sharp = rendered(gpu, source, shaders->hooks(), kIn * 2);
    // Les shaders ont tourné : l'image n'est pas celle du filtre seul, et le trait est plus raide.
    CHECK(mean_gap(sharp, plain) > 1e-3);
    CHECK(steepest(sharp, kIn * 2) > steepest(plain, kIn * 2));
    // Le plafonnement des hautes lumières (Clamp_Highlights) : le côté clair du trait ne déborde pas plus
    // que celui du filtre seul (ses lobes négatifs). Le côté sombre, lui, se creuse : Restore et le ×2
    // d'Anime4K renforcent les traits sombres (sur ce bord, jusqu'au noir), c'est leur rôle.
    const double plain_high = std::ranges::max(plain);
    CHECK(std::ranges::max(sharp) <= plain_high + 0.004);
    // Les aplats, loin du trait, gardent leur valeur.
    const auto at = [&](int x, int y) { return sharp[static_cast<std::size_t>(y * kIn * 2 + x)]; };
    CHECK(std::abs(at(kIn, kIn) - 0.15) < 0.01);
    CHECK(std::abs(at(4, 4) - 0.8) < 0.01);
  }
}

TEST_CASE("Anime4K : sur une fenêtre à peine plus grande que l'image, pas de ×2", "[render][gpu][anime4k]") {
  // La condition //!WHEN des shaders ×2 : la fenêtre doit dépasser 1,2 fois l'image. En dessous, le ×2 serait
  // réduit aussitôt, Anime4K ne garde que sa restauration (A) ou rien (C).
  Gpu gpu;
  require_device(gpu);
  const std::vector<std::uint16_t> source = disc();
  const int side = kIn + kIn / 8;  // 1,125
  const std::vector<double> plain = rendered(gpu, source, {}, side);
  // C sans son ×2 : il ne reste que le plafond des hautes lumières, qui retire les lobes clairs du filtre,
  // sans raidir le trait.
  auto c = Anime4k::create(gpu.vk->gpu, {.size = Anime4kSize::Fast, .mode = Anime4kMode::C});
  REQUIRE(c);
  const std::vector<double> c_only = rendered(gpu, source, c->hooks(), side);
  CHECK(mean_gap(c_only, plain) < 5e-3);
  CHECK(steepest(c_only, side) <= steepest(plain, side) + 0.01);
  // A garde sa restauration (sans //!WHEN) : le trait se raidit même sans ×2.
  auto a = Anime4k::create(gpu.vk->gpu, {.size = Anime4kSize::Fast, .mode = Anime4kMode::A});
  REQUIRE(a);
  const std::vector<double> a_only = rendered(gpu, source, a->hooks(), side);
  CHECK(steepest(a_only, side) > steepest(c_only, side) + 0.02);
  // Et sur une fenêtre au double, C agrandit bien : ce n'est pas le ×2 qui manque à C plus haut.
  CHECK(mean_gap(rendered(gpu, source, c->hooks(), kIn * 2), rendered(gpu, source, {}, kIn * 2)) > 1e-3);
}

TEST_CASE("Anime4K : déplacé, le jeu de shaders reste valide", "[render][gpu][anime4k]") {
  Gpu gpu;
  require_device(gpu);
  auto made = Anime4k::create(gpu.vk->gpu, {.size = Anime4kSize::Fast, .mode = Anime4kMode::B});
  REQUIRE(made);
  Anime4k moved = std::move(*made);
  CHECK(made->hooks().empty());  // NOLINT(bugprone-use-after-move) : l'état laissé est vide, c'est le test
  CHECK(moved.hooks().size() == 3U);
  auto other = Anime4k::create(gpu.vk->gpu, {.size = Anime4kSize::Quality, .mode = Anime4kMode::C});
  REQUIRE(other);
  moved = std::move(*other);
  CHECK(moved.choice().size == Anime4kSize::Quality);
  CHECK(moved.hooks().size() == 2U);
}

}  // namespace
}  // namespace ajp::render
