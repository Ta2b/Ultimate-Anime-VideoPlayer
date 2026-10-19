// Le grade sur le GPU : les hooks embarqués rendus par libplacebo sur des rampes PQ BT.2020
// synthétiques, cible identique à la source (aucune conversion, aucun tramage). Sans appareil
// Vulkan, les tests se sautent.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include "render/grade.h"

namespace ajp::render {
namespace {

constexpr int kSide = 256;
constexpr double kPeak = 1000.0;

double nits_to_pq(double nits) {
  const double q = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), 0.1593017578125);
  return std::pow((0.8359375 + 18.8515625 * q) / (1.0 + 18.6875 * q), 78.84375);
}

double pq_to_nits(double e) {
  const double p = std::pow(std::clamp(e, 0.0, 1.0), 1.0 / 78.84375);
  return 10000.0 * std::pow(std::max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}

// Un GPU Vulkan sans fenêtre, son renderer et le grade.
struct Gpu {
  pl_log log = nullptr;
  pl_vulkan vk = nullptr;
  pl_renderer renderer = nullptr;
  std::optional<Grade> grade;
  Gpu() {
    log = pl_log_create(PL_API_VER, nullptr);
    vk = pl_vulkan_create(log, &pl_vulkan_default_params);
    if (vk == nullptr) return;
    renderer = pl_renderer_create(log, vk->gpu);
    if (auto created = Grade::create(vk->gpu)) grade.emplace(std::move(*created));
  }
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;
  Gpu(Gpu&&) = delete;
  Gpu& operator=(Gpu&&) = delete;
  ~Gpu() {
    grade.reset();
    pl_renderer_destroy(&renderer);
    pl_vulkan_destroy(&vk);
    pl_log_destroy(&log);
  }
};

pl_frame frame_of(pl_tex tex) {
  pl_frame frame{};
  frame.num_planes = 1;
  frame.planes[0].texture = tex;
  frame.planes[0].components = 4;
  for (int c = 0; c < 4; ++c) frame.planes[0].component_mapping[c] = c;
  frame.repr = pl_color_repr_rgb;
  frame.color = pl_color_space_hdr10;
  frame.color.hdr.max_luma = static_cast<float>(kPeak);
  frame.crop = {0, 0, kSide, kSide};
  return frame;
}

// Chaque ligne : une rampe de 0 à top nits sur le canal R, G à green × R et B à blue × R.
std::vector<std::uint16_t> ramp(double top, double green, double blue) {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (int y = 0; y < kSide; ++y) {
    for (int x = 0; x < kSide; ++x) {
      const double r = top * (x + 1) / kSide;
      const std::array<double, 4> rgba{nits_to_pq(r), nits_to_pq(r * green), nits_to_pq(r * blue), 1.0};
      for (std::size_t c = 0; c < 4; ++c)
        pixels[(static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4 + c] =
            static_cast<std::uint16_t>(std::lround(rgba[c] * 65535.0));
    }
  }
  return pixels;
}

// La rampe passée par le grade : les nits de R le long de la ligne du milieu.
std::vector<double> graded(Gpu& gpu, const std::vector<std::uint16_t>& source, const GradeParams& params) {
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
  gpu.grade->set_params(params);
  pl_render_params render = pl_render_default_params;
  render.dither_params = nullptr;
  const std::span<const pl_hook* const> hooks = gpu.grade->hooks();
  render.hooks = hooks.data();
  render.num_hooks = static_cast<int>(hooks.size());
  const pl_frame image = frame_of(in);
  const pl_frame target = frame_of(out);
  REQUIRE(pl_render_image(gpu.renderer, &image, &target, &render));
  std::vector<std::uint16_t> pixels(source.size());
  pl_tex_transfer_params download{};
  download.tex = out;
  download.ptr = pixels.data();
  REQUIRE(pl_tex_download(g, &download));
  pl_tex_destroy(g, &in);
  pl_tex_destroy(g, &out);
  std::vector<double> red(kSide);
  const std::size_t row = static_cast<std::size_t>(kSide / 2) * kSide;
  for (std::size_t x = 0; x < red.size(); ++x) red[x] = pq_to_nits(pixels[(row + x) * 4] / 65535.0);
  return red;
}

// Des éclats poussés fort, sans pied des noirs ni vibrance : seule l'épaule compte.
GradeParams strong(double guard) {
  GradeParams p;
  p.peak = kPeak;
  p.paper = 0.5 * kPeak;
  p.in_max = kPeak;
  p.shadows = 0.0;
  // Même gain pour les aplats et les éclats : la carte d'isolement (blocs 8x8) ne joue pas.
  p.area_gain = 3.0;
  p.spec_gain = 3.0;
  p.vibrance = 0.0;
  p.gamut_p3 = 0.0;
  p.guard = guard;
  return p;
}

}  // namespace

// Un orange vif (Y ≈ 0,67 R) entre dans la zone des éclats : son canal R dépasse le pic bien avant
// sa luminance.
TEST_CASE("garde-fou : un orange saturé garde son dégradé au lieu d'un plateau au pic", "[render][gpu]") {
  Gpu gpu;
  if (gpu.vk == nullptr) SKIP("pas d'appareil Vulkan");
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> orange = ramp(900.0, 0.6, 0.03);
  const std::vector<double> hard = graded(gpu, orange, strong(0.0));
  const std::vector<double> soft = graded(gpu, orange, strong(1.0));
  // Sans garde-fou : le haut de la rampe est écrêté au pic (canal R à 1000 nits).
  const auto at_peak = [](const std::vector<double>& v) {
    return std::ranges::count_if(v, [](double n) { return n > 0.995 * kPeak; });
  };
  CHECK(at_peak(hard) > kSide / 8);
  // Avec : sous le pic, et toujours croissant (le dégradé reste).
  CHECK(at_peak(soft) == 0);
  // Croissant au pas près des textures intermédiaires de libplacebo (demi-flottants : ~0,4 % en PQ).
  for (std::size_t x = kSide / 2; x < soft.size(); ++x) CHECK(soft[x] >= soft[x - 1] * 0.994);
  CHECK(soft.back() > soft[kSide / 2] + 50.0);
}

TEST_CASE("garde-fou : le gris et le blanc ne bougent pas", "[render][gpu]") {
  Gpu gpu;
  if (gpu.vk == nullptr) SKIP("pas d'appareil Vulkan");
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> gray = ramp(900.0, 1.0, 1.0);
  const std::vector<double> hard = graded(gpu, gray, strong(0.0));
  const std::vector<double> soft = graded(gpu, gray, strong(1.0));
  for (std::size_t x = 0; x < hard.size(); ++x) CHECK(std::abs(hard[x] - soft[x]) < 0.5);
}

}  // namespace ajp::render
