// HDR10+ par plan (8.9) : une rampe grise PQ jusqu'à 400 nits, montrée sur une cible de 766 nits par la
// spline de libplacebo, sans détection de pic. Un plan annoncé à 400 nits passe tel quel ; le même annoncé à
// 4 000 nits est compressé.
#include "render/scene_hdr.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <libplacebo/renderer.h>
#include <libplacebo/tone_mapping.h>

#include "grade_gpu_support.h"

namespace ajp::render {
namespace {

using gpu_test::kSide;

constexpr double kTop = 400.0;
constexpr float kTarget = 766.0F;

// La rampe passée de la source (plan annoncé à scene) à la cible : les nits de la ligne du milieu.
std::vector<double> shown(gpu_test::Gpu& gpu, const SceneHdr& scene) {
  const pl_gpu g = gpu.vk->gpu;
  const std::vector<std::uint16_t> source = gpu_test::ramp(kTop, 1.0, 1.0);
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
  pl_frame image = gpu_test::frame_of(in, false);
  image.color.hdr.max_luma = 4000.0F;  // la valeur statique de la source : elle ne compte plus
  set_scene_hdr(image.color, scene);
  pl_frame target = gpu_test::frame_of(out, false);
  target.color.hdr.max_luma = kTarget;
  const pl_color_map_params map = scene_color_map();
  pl_render_params params = pl_render_default_params;
  params.dither_params = nullptr;
  params.peak_detect_params = nullptr;
  params.color_map_params = &map;
  REQUIRE(pl_render_image(gpu.renderer, &image, &target, &params));
  std::vector<std::uint16_t> pixels(source.size());
  pl_tex_transfer_params download{};
  download.tex = out;
  download.ptr = pixels.data();
  REQUIRE(pl_tex_download(g, &download));
  pl_tex_destroy(g, &in);
  pl_tex_destroy(g, &out);
  std::vector<double> line(kSide);
  const std::size_t row = static_cast<std::size_t>(kSide / 2) * kSide;
  for (std::size_t x = 0; x < line.size(); ++x)
    line[x] = gpu_test::pq_to_nits(pixels[(row + x) * 4] / 65535.0);
  return line;
}

double input_at(std::size_t x) {
  return kTop * static_cast<double>(x + 1) / kSide;
}

TEST_CASE("scene_hdr : les métadonnées HDR10+ d'un plan", "[render]") {
  pl_color_space space = pl_color_space_hdr10;
  set_scene_hdr(space, {.max_nits = 400.0F, .avg_nits = 80.0F});
  CHECK(space.hdr.scene_max[0] == 400.0F);
  CHECK(space.hdr.scene_max[2] == 400.0F);
  CHECK(space.hdr.scene_avg == 80.0F);
  CHECK(pl_hdr_metadata_contains(&space.hdr, PL_HDR_METADATA_HDR10PLUS));
  pl_color_space empty = pl_color_space_hdr10;
  set_scene_hdr(empty, {.max_nits = 400.0F, .avg_nits = 0.0F});
  CHECK_FALSE(pl_hdr_metadata_contains(&empty.hdr, PL_HDR_METADATA_HDR10PLUS));
  const pl_color_map_params map = scene_color_map();
  CHECK(map.tone_mapping_function == &pl_tone_map_spline);
  CHECK(map.metadata == PL_HDR_METADATA_HDR10PLUS);
}

TEST_CASE("scene_hdr : un plan sous la cible passe, un plan au-dessus est compressé", "[render][gpu]") {
  gpu_test::Gpu gpu;
  gpu_test::require_device(gpu);
  const std::vector<double> kept = shown(gpu, {.max_nits = 400.0F, .avg_nits = 100.0F});
  const std::vector<double> squeezed = shown(gpu, {.max_nits = 4000.0F, .avg_nits = 100.0F});
  // Au-dessus de 20 nits : en dessous, l'écart relatif d'un code PQ 16 bits n'est plus parlant.
  for (std::size_t x = kSide / 16; x < kept.size(); ++x) {
    INFO("x = " << x << ", entrée " << input_at(x) << " nits, sortie " << kept[x]);
    CHECK(kept[x] == Catch::Approx(input_at(x)).epsilon(0.01));
  }
  const std::size_t bright = static_cast<std::size_t>(kSide) * 3 / 4 - 1;  // 300 nits
  INFO("à 300 nits : " << squeezed[bright]);
  CHECK(squeezed[bright] < 0.99 * input_at(bright));
  CHECK(squeezed.back() < 0.99 * kTop);
}

}  // namespace
}  // namespace ajp::render
