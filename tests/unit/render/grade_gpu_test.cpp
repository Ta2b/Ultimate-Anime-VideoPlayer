// Le grade sur le GPU : les hooks embarqués rendus par libplacebo sur des rampes PQ BT.2020 (ou
// SDR BT.709 gamma 2,2) synthétiques, cible identique à la source (aucune conversion, aucun tramage). Un
// Vulkan logiciel (lavapipe) suffit, ce qui les fait tourner en CI ; sans aucun appareil Vulkan, ils se
// sautent. L'instance porte la couche de validation selon tests/support/vulkan_validation.h : ses
// erreurs, que libplacebo journalise en « vk … », sont notées et font échouer l'exécutable.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>
#include <libplacebo/vulkan.h>

#include "render/cube_lut.h"
#include "render/grade.h"
#include "vulkan_validation.h"

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

// Les erreurs de la couche de validation, que libplacebo journalise en « vk … » (vk_dbg_utils_cb).
void on_placebo_log(void* /*priv*/, pl_log_level level, const char* message) {
  if (message == nullptr) return;
  const std::string_view text(message);
  if ((level == PL_LOG_ERR || level == PL_LOG_FATAL) && text.starts_with("vk ")) {
    ajp::test::note_validation_error(text);
  }
}

bool has_validation_layer(const pl_vk_inst_t& inst) {
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
void require_device(const Gpu& gpu) {
  if (gpu.layer_missing && ajp::test::validation_mode() == ajp::test::ValidationMode::Require) {
    FAIL("couche de validation Vulkan exigée (AJP_TEST_VULKAN_VALIDATION=require) et absente");
  }
  if (gpu.vk == nullptr) SKIP("pas d'appareil Vulkan");
}

// La sortie SDR des tests : BT.709, gamma 2,2 (celle d'un écran SDR sous libplacebo).
pl_color_space sdr_space() {
  pl_color_space space{};
  space.primaries = PL_COLOR_PRIM_BT_709;
  space.transfer = PL_COLOR_TRC_GAMMA22;
  return space;
}

pl_frame frame_of(pl_tex tex, bool sdr) {
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

// La même en codes SDR : R de 0 à top, G à green × R et B à blue × R (codes, pas lumière).
std::vector<std::uint16_t> sdr_ramp(double top, double green, double blue) {
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

// La rampe passée par le grade : les codes de R, G et B le long de la ligne du milieu.
std::vector<Rgb> graded_codes(Gpu& gpu, const std::vector<std::uint16_t>& source, const GradeParams& params) {
  const bool sdr = params.sdr_out > 0.5;
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
  std::vector<Rgb> line(kSide);
  const std::size_t row = static_cast<std::size_t>(kSide / 2) * kSide;
  for (std::size_t x = 0; x < line.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) line[x][c] = pixels[(row + x) * 4 + c] / 65535.0;
  return line;
}

// La rampe PQ passée par le grade : les nits de R, G et B le long de la ligne du milieu.
std::vector<Rgb> graded_rgb(Gpu& gpu, const std::vector<std::uint16_t>& source, const GradeParams& params) {
  std::vector<Rgb> line = graded_codes(gpu, source, params);
  for (Rgb& px : line)
    for (double& c : px) c = pq_to_nits(c);
  return line;
}

// Les nits de R seuls.
std::vector<double> graded(Gpu& gpu, const std::vector<std::uint16_t>& source, const GradeParams& params) {
  const std::vector<Rgb> line = graded_rgb(gpu, source, params);
  std::vector<double> red(line.size());
  std::ranges::transform(line, red.begin(), [](const Rgb& px) { return px[0]; });
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

// Un grade neutre (gains à 1, ni pied ni vibrance), sauf les couleurs étendues.
GradeParams expanded(double expand) {
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
GradeParams sdr_neutral() {
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
double saturation(const Rgb& px) {
  const double hi = std::max({px[0], px[1], px[2]});
  return hi > 0.0 ? (hi - std::min({px[0], px[1], px[2]})) / hi : 0.0;
}

}  // namespace

// Un orange vif (Y ≈ 0,67 R) entre dans la zone des éclats : son canal R dépasse le pic bien avant
// sa luminance.
TEST_CASE("garde-fou : un orange saturé garde son dégradé au lieu d'un plateau au pic", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
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
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> gray = ramp(900.0, 1.0, 1.0);
  const std::vector<double> hard = graded(gpu, gray, strong(0.0));
  const std::vector<double> soft = graded(gpu, gray, strong(1.0));
  for (std::size_t x = 0; x < hard.size(); ++x) CHECK(std::abs(hard[x] - soft[x]) < 0.5);
}

// Le rouge pur BT.709 dans le conteneur BT.2020 : (0,6274 ; 0,0691 ; 0,0164) × R. Relu avec les
// primaires P3, il devient (0,7538 ; 0,0457 ; -0,0012) : G/R passe de 0,110 à 0,061.
TEST_CASE("couleurs étendues : un rouge BT.709 va vers le rouge P3, le gris ne bouge pas", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> red = ramp(400.0, 0.0691 / 0.6274, 0.0164 / 0.6274);
  const std::vector<Rgb> plain = graded_rgb(gpu, red, expanded(0.0));
  const std::vector<Rgb> wide = graded_rgb(gpu, red, expanded(1.0));
  const std::size_t mid = kSide / 2;
  const double before = plain[mid][1] / plain[mid][0];
  const double after = wide[mid][1] / wide[mid][0];
  CHECK(before > 0.10);
  CHECK(after < 0.075);
  CHECK(wide[mid][0] > plain[mid][0]);  // le canal rouge monte : la couleur est plus pure
  const std::vector<std::uint16_t> gray = ramp(400.0, 1.0, 1.0);
  const std::vector<Rgb> gray_plain = graded_rgb(gpu, gray, expanded(0.0));
  const std::vector<Rgb> gray_wide = graded_rgb(gpu, gray, expanded(1.0));
  for (std::size_t x = 0; x < gray_plain.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(gray_wide[x][c] - gray_plain[x][c]) < 0.5);
}

TEST_CASE("look : huit LUT s'enchaînent, la dernière compte comme la première", "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = sdr_ramp(1.0, 0.8, 0.6);
  const std::vector<Rgb> base = graded_codes(gpu, source, sdr_neutral());
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  std::array<const CubeLut*, kLookSlots> luts{};
  luts[3] = &inverted;
  luts[7] = &inverted;
  REQUIRE(gpu.grade->set_looks(luts));
  GradeParams params = sdr_neutral();
  params.look_mix4 = 1.0;
  params.look_mix8 = 1.0;
  const std::vector<Rgb> twice = graded_codes(gpu, source, params);
  for (std::size_t x = 8; x < twice.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(twice[x][c] - base[x][c]) < 8e-3);
  params.look_mix8 = 0.0;
  const std::vector<Rgb> once = graded_codes(gpu, source, params);
  CHECK(once[32][0] > 0.55);
}

}  // namespace ajp::render

namespace ajp::render {
namespace {

// Une image uniforme à `nits` (PQ), comme une scène pleine d'un seul niveau.
std::vector<std::uint16_t> uniform(double nits) {
  std::vector<std::uint16_t> pixels(static_cast<std::size_t>(kSide) * kSide * 4);
  const auto code = static_cast<std::uint16_t>(std::lround(nits_to_pq(nits) * 65535.0));
  for (std::size_t i = 0; i < pixels.size(); i += 4) {
    pixels[i] = pixels[i + 1] = pixels[i + 2] = code;
    pixels[i + 3] = 65535;
  }
  return pixels;
}

// La route native avec l'aspect TrueHDR émulé, contraste 200 et gris moyen 51, sans autre retouche.
GradeParams native_look(double adapt) {
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

}  // namespace

TEST_CASE("HDR adaptatif : la nuit garde ses ombres, une scène claire s'étend moins", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const auto center = [&](double nits, double adapt) {
    return graded(gpu, uniform(nits), native_look(adapt))[kSide / 2];
  };
  // Nuit (3 nits, 1,5 % du blanc SDR) : moitié du contraste sous le pivot, plus clair qu'en fixe.
  const double night_fixed = center(3.0, 0.0);
  const double night_adapt = center(3.0, 1.0);
  CHECK(night_adapt > 1.2 * night_fixed);
  // Scène claire (180 nits, 89 % du blanc, tout en aplat clair) : l'expansion est freinée.
  const double day_fixed = center(180.0, 0.0);
  const double day_adapt = center(180.0, 1.0);
  CHECK(day_fixed > 450.0);
  CHECK(day_adapt < 0.7 * day_fixed);
  CHECK(day_adapt > 180.0);
  // Scène ordinaire (60 nits) : la courbe fixe, à l'arrondi près.
  CHECK(std::abs(center(60.0, 1.0) - center(60.0, 0.0)) < 0.02 * center(60.0, 0.0));
}

TEST_CASE("sortie SDR : le grade neutre rend l'image telle quelle", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  for (const auto& [green, blue] : {std::pair{1.0, 1.0}, std::pair{0.6, 0.1}, std::pair{0.2, 0.9}}) {
    const std::vector<std::uint16_t> source = sdr_ramp(1.0, green, blue);
    const std::vector<Rgb> out = graded_codes(gpu, source, sdr_neutral());
    const std::size_t row = static_cast<std::size_t>(kSide / 2) * kSide;
    for (std::size_t x = 0; x < out.size(); ++x)
      for (std::size_t c = 0; c < 3; ++c)
        CHECK(std::abs(out[x][c] - source[(row + x) * 4 + c] / 65535.0) < 4e-3);
  }
}

TEST_CASE("sortie SDR : couleurs et contraste poussés, sans dépasser le blanc", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  GradeParams vivid = sdr_neutral();
  vivid.vibrance = 0.45;
  vivid.emulate = 1.0;
  vivid.sdr_white = 203.0;
  vivid.em_base = 200.0;
  vivid.em_lo = 1.19;
  vivid.em_hi = 1.14;
  vivid.em_sat = 1.1;
  // Un bleu ciel moyen : plus saturé, toujours dans le gamut BT.709.
  const std::vector<std::uint16_t> sky = sdr_ramp(0.8, 1.6, 2.0);  // codes écrêtés à 1 : B sature
  const std::vector<Rgb> plain = graded_codes(gpu, sdr_ramp(0.5, 1.4, 1.8), sdr_neutral());
  const std::vector<Rgb> pushed = graded_codes(gpu, sdr_ramp(0.5, 1.4, 1.8), vivid);
  const std::size_t mid = 3 * kSide / 4;
  CHECK(saturation(pushed[mid]) > saturation(plain[mid]) + 0.01);
  for (const Rgb& px : graded_codes(gpu, sky, vivid))
    for (const double c : px) CHECK((c >= 0.0 && c <= 1.0));
  // Le gris : le blanc reste blanc et neutre.
  const std::vector<Rgb> gray = graded_codes(gpu, sdr_ramp(1.0, 1.0, 1.0), vivid);
  CHECK(gray.back()[0] > 0.985);
  CHECK(std::abs(gray[kSide / 8][0] - gray[kSide / 8][2]) < 4e-3);
}

TEST_CASE("sortie SDR : le contraste maximal de RTX HDR n'écrase pas les sombres", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  // Contraste 200, gris moyen 51 (les réglages de RTX HDR de Tatoub), le reste neutre.
  GradeParams look = sdr_neutral();
  look.emulate = 1.0;
  look.sdr_white = 203.0;
  look.em_base = 204.0;
  look.em_pow = 6.0 * std::pow(50.0 / 51.0, 0.35);
  look.em_lo = 1.38;
  look.em_hi = 1.28;
  look.em_sat = 1.0;
  const std::vector<Rgb> out = graded_codes(gpu, sdr_ramp(1.0, 1.0, 1.0), look);
  // La médiane d'une scène de nuit de Re:Zero : code 0,145, 2,9 nits. Avec le seul contraste sous
  // le pivot, elle tombait à 1 nit ; l'aspect TrueHDR la garde au-dessus de 2 nits.
  const auto code = [](std::size_t x) { return static_cast<double>(x + 1) / kSide; };
  for (std::size_t x = 0; x < out.size(); ++x) {
    const double in = code(x);
    if (in < 0.1 || in > 0.3) continue;
    const double nits_in = 203.0 * std::pow(in, 2.2);
    const double nits_out = 203.0 * std::pow(out[x][1], 2.2);
    CHECK(nits_out > 0.7 * nits_in);
  }
}

TEST_CASE("sortie SDR : le vrai noir OLED met les quasi-noirs à 0", "[render][gpu]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  GradeParams black = sdr_neutral();
  // Codes 0,02 et 0,06 en gamma 2,2 à 203 nits : 0,04 et 0,4 nit environ.
  black.black_lo = 203.0 * std::pow(0.02, 2.2);
  black.black_hi = 203.0 * std::pow(0.06, 2.2);
  const std::vector<Rgb> out = graded_codes(gpu, sdr_ramp(0.25, 1.0, 1.0), black);
  // Les codes de la rampe : 0,25 × (x + 1) / 256.
  const auto code = [](std::size_t x) { return 0.25 * static_cast<double>(x + 1) / kSide; };
  for (std::size_t x = 0; x < out.size(); ++x) {
    if (code(x) < 0.015) CHECK(out[x][0] < 1e-3);
    if (code(x) > 0.09) CHECK(std::abs(out[x][0] - code(x)) < 4e-3);
  }
}

// Le look (LUT 3D) : ni la passe sautée (look_mix = 0), ni une identité ne changent l'image ; une LUT
// inversée l'inverse ; une LUT d'HDR lit les codes PQ.
TEST_CASE("look : identité et intensité nulle laissent l'image, une LUT inversée l'inverse",
          "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = sdr_ramp(1.0, 0.8, 0.6);
  const std::vector<Rgb> base = graded_codes(gpu, source, sdr_neutral());
  const CubeLut neutral = identity_lut(17);
  REQUIRE(gpu.grade->set_looks({&neutral, nullptr, nullptr}));
  GradeParams params = sdr_neutral();
  params.look_mix = 1.0;
  params.look_white = 203.0;
  const std::vector<Rgb> same = graded_codes(gpu, source, params);
  for (std::size_t x = 8; x < same.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(same[x][c] - base[x][c]) < 6e-3);
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  REQUIRE(gpu.grade->set_looks({&inverted, nullptr, nullptr}));
  const std::vector<Rgb> flipped = graded_codes(gpu, source, params);
  CHECK(flipped[32][0] > 0.55);   // sombre -> clair
  CHECK(flipped[240][0] < 0.35);  // clair -> sombre
  params.look_mix = 0.0;
  const std::vector<Rgb> skipped = graded_codes(gpu, source, params);
  for (std::size_t x = 0; x < skipped.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(skipped[x][c] == base[x][c]);  // octet pour octet
  REQUIRE(gpu.grade->set_looks({}));  // retour à l'identité 2x2x2 sans LUT
}

TEST_CASE("look : une LUT d'HDR (PQ BT.2020) lit les codes PQ, l'identité ne bouge rien",
          "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = ramp(600.0, 0.7, 0.5);
  GradeParams neutral;
  neutral.peak = kPeak;
  neutral.shadows = 0.0;
  neutral.area_gain = 1.0;
  neutral.spec_gain = 1.0;
  neutral.vibrance = 0.0;
  neutral.gamut_p3 = 0.0;
  const std::vector<Rgb> base = graded_codes(gpu, source, neutral);
  const CubeLut lut = identity_lut(33);
  REQUIRE(gpu.grade->set_looks({&lut, nullptr, nullptr}));
  GradeParams params = neutral;
  params.look_mix = 1.0;
  params.look_pq = 1.0;
  const std::vector<Rgb> out = graded_codes(gpu, source, params);
  for (std::size_t x = 4; x < out.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(out[x][c] - base[x][c]) < 6e-3);
  // Le chemin SDR-dans-HDR avec une LUT neutre garde l'image à quelques millièmes près en dessous du blanc.
  params.look_pq = 0.0;
  const std::vector<Rgb> via_sdr = graded_codes(gpu, source, params);
  for (std::size_t x = 4; x < via_sdr.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(std::abs(via_sdr[x][c] - base[x][c]) < 1.5e-2);
}

TEST_CASE("look : deux LUT s'enchaînent dans l'ordre, deux inversions se compensent", "[render][gpu][lut]") {
  Gpu gpu;
  require_device(gpu);
  REQUIRE(gpu.grade);
  const std::vector<std::uint16_t> source = sdr_ramp(1.0, 0.8, 0.6);
  const std::vector<Rgb> base = graded_codes(gpu, source, sdr_neutral());
  CubeLut inverted = identity_lut(17);
  for (float& value : inverted.rgb) value = 1.0F - value;
  REQUIRE(gpu.grade->set_looks({&inverted, &inverted, nullptr}));
  GradeParams params = sdr_neutral();
  params.look_mix = 1.0;
  params.look_mix2 = 1.0;
  const std::vector<Rgb> twice = graded_codes(gpu, source, params);
  for (std::size_t x = 8; x < twice.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c)
      CHECK(std::abs(twice[x][c] - base[x][c]) < 8e-3);  // inversion puis inversion
  params.look_mix2 = 0.0;                                // la deuxième sautée : une seule inversion
  const std::vector<Rgb> once = graded_codes(gpu, source, params);
  CHECK(once[32][0] > 0.55);
  params.look_mix = 0.0;
  params.look_mix2 = 1.0;  // seule la deuxième
  const std::vector<Rgb> second_only = graded_codes(gpu, source, params);
  CHECK(second_only[32][0] > 0.55);
  params.look_mix2 = 0.0;
  const std::vector<Rgb> none = graded_codes(gpu, source, params);
  for (std::size_t x = 0; x < none.size(); ++x)
    for (std::size_t c = 0; c < 3; ++c) CHECK(none[x][c] == base[x][c]);
}

}  // namespace ajp::render
