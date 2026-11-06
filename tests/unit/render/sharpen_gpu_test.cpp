// La netteté sur le GPU (render/sharpen.h) : le hook embarqué rendu par libplacebo sur une marche
// synthétique, cible identique à la source. Lavapipe suffit ; sans appareil Vulkan, le test se saute.
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <libplacebo/renderer.h>

#include "grade_gpu_support.h"
#include "render/sharpen.h"

namespace ajp::render {
namespace {

using namespace gpu_test;

// Une marche verticale : à gauche `dark`, à droite `light` (codes 16 bits, gris).
std::vector<std::uint16_t> step(double dark, double light) {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (int y = 0; y < kSide; ++y) {
    for (int x = 0; x < kSide; ++x) {
      const auto code = static_cast<std::uint16_t>((x < kSide / 2 ? dark : light) * 65535.0 + 0.5);
      const std::size_t at = (static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4;
      pixels[at] = pixels[at + 1] = pixels[at + 2] = code;
      pixels[at + 3] = 65535;
    }
  }
  return pixels;
}

// La ligne du milieu, en valeurs de 0 à 1 (canal rouge), après le hook (ou sans hook).
std::vector<double> rendered(Gpu& gpu, const std::vector<std::uint16_t>& source, const pl_hook* hook) {
  const pl_gpu g = gpu.vk->gpu;
  pl_tex_params tex_params{};
  tex_params.w = kSide;
  tex_params.h = kSide;
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
  pl_tex out = pl_tex_create(g, &tex_params);
  REQUIRE(in != nullptr);
  REQUIRE(out != nullptr);
  pl_render_params render = pl_render_default_params;
  render.dither_params = nullptr;
  if (hook != nullptr) {
    render.hooks = &hook;
    render.num_hooks = 1;
  }
  const pl_frame image = frame_of(in, true);
  const pl_frame target = frame_of(out, true);
  REQUIRE(pl_render_image(gpu.renderer, &image, &target, &render));
  std::vector<std::uint16_t> pixels(source.size());
  pl_tex_transfer_params download{};
  download.tex = out;
  download.ptr = pixels.data();
  REQUIRE(pl_tex_download(g, &download));
  pl_tex_destroy(g, &in);
  pl_tex_destroy(g, &out);
  std::vector<double> row(kSide);
  for (int x = 0; x < kSide; ++x)
    row[static_cast<std::size_t>(x)] =
        pixels[(static_cast<std::size_t>(kSide / 2) * kSide + static_cast<std::size_t>(x)) * 4] / 65535.0;
  return row;
}

TEST_CASE("netteté : rien à 0, un bord plus franc sans halo, les aplats intacts", "[render][gpu][sharpen]") {
  Gpu gpu;
  require_device(gpu);
  auto sharpen = Sharpen::create(gpu.vk->gpu);
  REQUIRE(sharpen);
  CHECK(sharpen->hook() == nullptr);  // force 0 : aucun hook posé
  sharpen->set_strength(2.0);
  CHECK(sharpen->strength() == 1.0);  // bornée
  REQUIRE(sharpen->hook() != nullptr);

  const std::vector<std::uint16_t> source = step(0.3, 0.6);
  const std::vector<double> plain = rendered(gpu, source, nullptr);
  const std::vector<double> sharp = rendered(gpu, source, sharpen->hook());
  const auto edge = static_cast<std::size_t>(kSide / 2);
  // Le bord se creuse côté sombre et monte côté clair, au plus de 10 % de l'écart (pas de liseré).
  CHECK(sharp[edge - 1] < plain[edge - 1] - 0.005);
  CHECK(sharp[edge] > plain[edge] + 0.005);
  CHECK(sharp[edge - 1] >= 0.3 - 0.1 * 0.3 - 0.002);
  CHECK(sharp[edge] <= 0.6 + 0.1 * 0.3 + 0.002);
  // Loin du bord, l'aplat ne bouge pas, à la précision près de la texture intermédiaire du hook
  // (demi-flottant : un pas de 0,0005 vers 0,6).
  CHECK(sharp[10] == Catch::Approx(plain[10]).margin(1e-3));
  CHECK(sharp[kSide - 10] == Catch::Approx(plain[kSide - 10]).margin(1e-3));
}

}  // namespace
}  // namespace ajp::render
