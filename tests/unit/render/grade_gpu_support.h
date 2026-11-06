// Les outils des tests du grade sur le GPU (grade_gpu_test.cpp, grade_gpu_local_test.cpp) : un Vulkan sans
// fenêtre, les images synthétiques en PQ BT.2020 ou en SDR, et leur passage par les hooks du grade.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include "render/grade.h"
#include "vulkan_validation.h"

namespace ajp::render::gpu_test {

inline constexpr int kSide = 256;
inline constexpr double kPeak = 1000.0;

inline double nits_to_pq(double nits) {
  const double q = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), 0.1593017578125);
  return std::pow((0.8359375 + 18.8515625 * q) / (1.0 + 18.6875 * q), 78.84375);
}

inline double pq_to_nits(double e) {
  const double p = std::pow(std::clamp(e, 0.0, 1.0), 1.0 / 78.84375);
  return 10000.0 * std::pow(std::max(p - 0.8359375, 0.0) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
}

// Les erreurs de la couche de validation, que libplacebo journalise en « vk … » (vk_dbg_utils_cb).
inline void on_placebo_log(void* /*priv*/, pl_log_level level, const char* message) {
  if (message == nullptr) return;
  const std::string_view text(message);
  if ((level == PL_LOG_ERR || level == PL_LOG_FATAL) && text.starts_with("vk ")) {
    ajp::test::note_validation_error(text);
  }
}

inline bool has_validation_layer(const pl_vk_inst_t& inst) {
  for (int i = 0; i < inst.num_layers; ++i) {
    if (std::strcmp(inst.layers[i], "VK_LAYER_KHRONOS_validation") == 0) return true;
  }
  return false;
}

// Un GPU Vulkan sans fenêtre, son renderer et le grade.
struct Gpu {
  pl_log log = nullptr;
  pl_vk_inst inst = nullptr;
  pl_vulkan vk = nullptr;
  pl_renderer renderer = nullptr;
  std::optional<Grade> grade;
  bool layer_missing = false;  // demandée (mode on ou require) et absente
  Gpu() {
    pl_log_params log_params{};
    log_params.log_cb = on_placebo_log;
    log_params.log_level = PL_LOG_WARN;
    log = pl_log_create(PL_API_VER, &log_params);
    const ajp::test::ValidationMode mode = ajp::test::validation_mode();
    pl_vk_inst_params inst_params{};
    inst_params.debug = mode != ajp::test::ValidationMode::Off;
    inst = pl_vk_inst_create(log, &inst_params);
    if (inst == nullptr) return;
    if (inst_params.debug && !has_validation_layer(*inst)) {
      layer_missing = true;
      if (mode != ajp::test::ValidationMode::Require)
        ajp::test::warn_without_layer("libplacebo ne la trouve pas");
    }
    pl_vulkan_params params = pl_vulkan_default_params;
    params.instance = inst->instance;
    params.get_proc_addr = inst->get_proc_addr;
    params.allow_software = true;
    vk = pl_vulkan_create(log, &params);
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
    pl_vk_inst_destroy(&inst);
    pl_log_destroy(&log);
  }
};

// Saute sans appareil ; échoue quand la couche de validation est exigée et absente.
inline void require_device(const Gpu& gpu) {
  if (gpu.layer_missing && ajp::test::validation_mode() == ajp::test::ValidationMode::Require) {
    FAIL("couche de validation Vulkan exigée (AJP_TEST_VULKAN_VALIDATION=require) et absente");
  }
  if (gpu.vk == nullptr) SKIP("pas d'appareil Vulkan");
}

// La sortie SDR des tests : BT.709, gamma 2,2 (celle d'un écran SDR sous libplacebo).
inline pl_color_space sdr_space() {
  pl_color_space space{};
  space.primaries = PL_COLOR_PRIM_BT_709;
  space.transfer = PL_COLOR_TRC_GAMMA22;
  return space;
}

inline pl_frame frame_of(pl_tex tex, bool sdr) {
  pl_frame frame{};
  frame.num_planes = 1;
  frame.planes[0].texture = tex;
  frame.planes[0].components = 4;
  for (int c = 0; c < 4; ++c) frame.planes[0].component_mapping[c] = c;
  frame.repr = pl_color_repr_rgb;
  frame.color = pl_color_space_hdr10;
  frame.color.hdr.max_luma = static_cast<float>(kPeak);
  if (sdr) frame.color = sdr_space();
  frame.crop = {0, 0, kSide, kSide};
  return frame;
}

// Chaque ligne : une rampe de 0 à top nits sur le canal R, G à green × R et B à blue × R.
inline std::vector<std::uint16_t> ramp(double top, double green, double blue) {
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

// La même en codes SDR : R de 0 à top, G à green × R et B à blue × R (codes, pas lumière).
inline std::vector<std::uint16_t> sdr_ramp(double top, double green, double blue) {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  for (int y = 0; y < kSide; ++y) {
    for (int x = 0; x < kSide; ++x) {
      const double r = top * (x + 1) / kSide;
      const std::array<double, 4> rgba{r, r * green, r * blue, 1.0};
      for (std::size_t c = 0; c < 4; ++c)
        pixels[(static_cast<std::size_t>(y) * kSide + static_cast<std::size_t>(x)) * 4 + c] =
            static_cast<std::uint16_t>(std::lround(rgba[c] * 65535.0));
    }
  }
  return pixels;
}

using Rgb = std::array<double, 3>;

// L'image entière passée par le grade : ses codes 16 bits RGBA.
inline std::vector<std::uint16_t> graded_image(Gpu& gpu, const std::vector<std::uint16_t>& source,
                                               const GradeParams& params) {
  const bool sdr = params.sdr_out > 0.5 || params.sdr_tm > 0.5;  // sdr_tm : cible SDR, grade en PQ
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
  const pl_frame image = frame_of(in, sdr);
  const pl_frame target = frame_of(out, sdr);
  REQUIRE(pl_render_image(gpu.renderer, &image, &target, &render));
  std::vector<std::uint16_t> pixels(source.size());
  pl_tex_transfer_params download{};
  download.tex = out;
  download.ptr = pixels.data();
  REQUIRE(pl_tex_download(g, &download));
  pl_tex_destroy(g, &in);
  pl_tex_destroy(g, &out);
  return pixels;
}

// La rampe passée par le grade : les codes de R, G et B le long de la ligne du milieu.
inline std::vector<Rgb> graded_codes(Gpu& gpu, const std::vector<std::uint16_t>& source,
                                     const GradeParams& params) {
  const std::vector<std::uint16_t> pixels = graded_image(gpu, source, params);
  std::vector<Rgb> line(kSide);
  const std::size_t row = static_cast<std::size_t>(kSide / 2) * kSide;
  for (std::size_t x = 0; x < line.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) line[x][c] = pixels[(row + x) * 4 + c] / 65535.0;
  return line;
}

// La rampe PQ passée par le grade : les nits de R, G et B le long de la ligne du milieu.
inline std::vector<Rgb> graded_rgb(Gpu& gpu, const std::vector<std::uint16_t>& source,
                                   const GradeParams& params) {
  std::vector<Rgb> line = graded_codes(gpu, source, params);
  for (Rgb& px : line)
    for (double& c : px) c = pq_to_nits(c);
  return line;
}

// Les nits de R seuls.
inline std::vector<double> graded(Gpu& gpu, const std::vector<std::uint16_t>& source,
                                  const GradeParams& params) {
  const std::vector<Rgb> line = graded_rgb(gpu, source, params);
  std::vector<double> red(line.size());
  std::ranges::transform(line, red.begin(), [](const Rgb& px) { return px[0]; });
  return red;
}

// Des éclats poussés fort, sans pied des noirs ni vibrance : seule l'épaule compte.
inline GradeParams strong(double guard) {
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

// Un grade neutre (gains à 1, ni pied ni vibrance), sauf les couleurs étendues.
inline GradeParams expanded(double expand) {
  GradeParams p;
  p.peak = kPeak;
  p.shadows = 0.0;
  p.area_gain = 1.0;
  p.spec_gain = 1.0;
  p.vibrance = 0.0;
  p.gamut_p3 = 0.0;
  p.expand = expand;
  return p;
}

// Une sortie SDR neutre : gains à 1, ni pied, ni vibrance, ni aspect TrueHDR.
inline GradeParams sdr_neutral() {
  GradeParams p;
  p.sdr_out = 1.0;
  p.sdr_gamma = 2.2;
  p.peak = 203.0;
  p.paper = 203.0;
  p.in_max = 203.0;
  p.shadows = 0.0;
  p.shadow_knee = 0.25 * 203.0;
  p.area_gain = 1.0;
  p.spec_gain = 1.0;
  p.vibrance = 0.0;
  p.gamut_p3 = 0.0;
  return p;
}

// Écart de saturation d'un pixel : (max − min) / max de ses codes.
inline double saturation(const Rgb& px) {
  const double hi = std::max({px[0], px[1], px[2]});
  return hi > 0.0 ? (hi - std::min({px[0], px[1], px[2]})) / hi : 0.0;
}

// Une image uniforme à `nits` (PQ), comme une scène pleine d'un seul niveau.
inline std::vector<std::uint16_t> uniform(double nits) {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  const auto code = static_cast<std::uint16_t>(std::lround(nits_to_pq(nits) * 65535.0));
  for (std::size_t i = 0; i < pixels.size(); i += 4) {
    pixels[i] = pixels[i + 1] = pixels[i + 2] = code;
    pixels[i + 3] = 65535;
  }
  return pixels;
}

// La route native avec l'aspect TrueHDR émulé, contraste 200 et gris moyen 51, sans autre retouche.
inline GradeParams native_look(double adapt) {
  GradeParams p;
  p.peak = kPeak;
  p.paper = 0.5 * kPeak;
  p.in_max = kPeak;
  p.shadows = 0.0;
  p.area_gain = 1.0;
  p.spec_gain = 1.0;
  p.vibrance = 0.0;
  p.gamut_p3 = 0.0;
  p.emulate = 1.0;
  p.adapt = adapt;
  p.sdr_white = 203.0;
  p.em_base = 204.0;
  p.em_pow = 6.0 * std::pow(50.0 / 51.0, 0.35);
  p.em_lo = 1.38;
  p.em_hi = 1.28;
  p.em_sat = 1.0;
  return p;
}

}  // namespace ajp::render::gpu_test
