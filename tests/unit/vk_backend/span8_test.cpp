// Le paquet « .span8 » sous le moteur Vulkan (net_span8.h, vulkan_span_derive.cpp), sur l'appareil du système
// (lavapipe en CI, sans matrices FP8 : le noyau direct y calcule l'arrondi E4M3 en float) : construit comme
// un ONNX, il rend ce que l'interpréteur de l'IR rend de son graphe ; tensor_peaks relève les maxima d'une
// fenêtre ; la dérivation calibre, juge et chronomètre, et ses maxima par tuiles égalent ceux de l'image
// entière.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_rig.h"
#include "gpu/ir/import.h"
#include "gpu/ir/interpreter.h"
#include "gpu/span_fp8/derive.h"
#include "gpu/span_fp8/metrics.h"
#include "gpu/span_fp8/package.h"
#include "gpu/span_fp8/package_model.h"
#include "gpu/vulkan/backend/kernel_ops.h"
#include "gpu/vulkan/compute/buffer.h"
#include "kernels/ref/half.h"
#include "net_rig.h"

using namespace kirari;
using namespace kirari::gpu;
using namespace kirari::gpu::vk::test;

namespace {

namespace sp = span_fp8;

sp::Conv noise_conv(int outs, int taps, int inputs, std::uint32_t& seed, float amplitude) {
  sp::Conv conv{.weights = std::vector<std::uint16_t>(static_cast<std::size_t>(outs * taps * inputs)),
                .bias = std::vector<float>(static_cast<std::size_t>(outs))};
  const auto next = [&seed] {
    seed = seed * 1664525U + 1013904223U;
    return static_cast<float>(seed >> 8U) / 16777216.0F - 0.5F;
  };
  for (std::uint16_t& w : conv.weights) w = kernels::ref::float_to_half(amplitude * next());
  for (float& b : conv.bias) b = kernels::ref::round_to_half(0.05F * next());
  return conv;
}

// Un SPAN de poids pseudo-aléatoires, 48 canaux réels (rangés en 64, comme un b5f48), échelles à 1.
sp::Package noise_package(int blocks) {
  std::uint32_t seed = 4242U;
  sp::Package p;
  p.real_channels = 48;
  p.first = noise_conv(sp::kChannels, 9, sp::kUnshuffledPadded, seed, 0.3F);
  p.blocks.assign(static_cast<std::size_t>(blocks), sp::Block{});
  for (sp::Block& b : p.blocks) {
    for (sp::Conv& conv : b.conv) conv = noise_conv(sp::kChannels, 9, sp::kChannels, seed, 0.12F);
    b.attention = noise_conv(sp::kChannels, 1, sp::kChannels, seed, 0.3F);
  }
  p.concat = noise_conv(sp::kChannels, 1, sp::kConcatChannels, seed, 0.15F);
  p.pixel = noise_conv(sp::kPixelChannels, 9, sp::kChannels, seed, 0.12F);
  p.scales.fill(1.0F);
  return p;
}

std::vector<float> pattern(int height, int width) {
  std::vector<float> rgb(std::size_t{3} * static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
  for (std::size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = kernels::ref::round_to_half(
        0.5F + 0.45F * std::sin(0.37F * static_cast<float>(i % 97) + 0.011F * static_cast<float>(i)));
  return rgb;
}

std::vector<ir::NamedValue> interpret(const sp::Package& p, const sp::ModelOptions& options,
                                      const std::vector<float>& rgb, int height, int width) {
  auto model = sp::package_model(p, options);
  REQUIRE(model);
  auto graph = ir::import_model(*model);
  REQUIRE(graph);
  const ir::Value input{.type = ir::DataType::F16, .dims = {1, 3, height, width}, .f = rgb, .i = {}};
  auto ran = ir::run(*graph, std::vector<ir::NamedValue>{{.name = sp::kModelInput, .value = input}});
  REQUIRE(ran);
  return std::move(*ran);
}

// Les échelles que la référence FP16 tire de cette image, comme la dérivation.
sp::Package calibrated(sp::Package p, const std::vector<float>& rgb, int height, int width) {
  const std::vector<ir::NamedValue> tapped =
      interpret(p, {.quantize = false, .peaks = true}, rgb, height, width);
  std::array<float, sp::kScaleCount> peaks{};
  for (std::size_t k = 0; k < sp::scale_count(p.blocks.size()); ++k)
    for (const float v : tapped[1 + k].value.f) peaks[k] = std::max(peaks[k], std::abs(v));
  auto scales = sp::scales_from_peaks(peaks, p.blocks.size());
  REQUIRE(scales);
  p.scales = *scales;
  return p;
}

ModelSpec write_package(const std::filesystem::path& dir, const std::string& name, const sp::Package& p) {
  const std::filesystem::path path = dir / (name + ".span8");
  const std::vector<std::byte> bytes = sp::serialize(p);
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(out.good());
  return ModelSpec{
      .id = name, .onnx_path = path, .sha256 = std::string(64, '0'), .precision = TensorType::F16};
}

ShapeSet shapes(std::int64_t h, std::int64_t w) {
  const Shape s{{1, 3, h, w}};
  return {.min = s, .opt = s, .max = s};
}

std::vector<float> run_engine(vk::VulkanBackend& backend, Engine& engine, const std::vector<float>& rgb,
                              int height, int width) {
  kernels::RgbImage image(width, height);
  const std::size_t plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  for (std::size_t c = 0; c < 3; ++c)
    std::copy_n(rgb.begin() + static_cast<std::ptrdiff_t>(c * plane), plane, image.planes[c].data.begin());
  auto in = require_value(backend.alloc_tensor(TensorType::F16, Shape{{1, 3, height, width}}));
  auto out = require_value(backend.alloc_tensor(TensorType::F16, Shape{{1, 3, 2 * height, 2 * width}}));
  write_tensor(backend, in->tensor(), rgb_bytes(image, TensorType::F16));
  std::array inputs{in->tensor()};
  std::array outputs{out->tensor()};
  require_ok(backend.enqueue(engine, inputs, outputs));
  const kernels::RgbImage got = rgb_from(read_tensor(backend, out->tensor(), plane * 4 * 3 * 2), 2 * width,
                                         2 * height, TensorType::F16);
  std::vector<float> flat;
  for (const auto& p : got.planes) flat.insert(flat.end(), p.data.begin(), p.data.end());
  return flat;
}

SpanStill still(int width, int height, std::uint32_t seed) {
  const kernels::RgbImage image = synthetic_rgb(width, height, seed);
  SpanStill s{.width = width, .height = height, .rgb = {}};
  for (const auto& p : image.planes)
    for (const float v : p.data)
      s.rgb.push_back(static_cast<std::uint8_t>(std::clamp(std::lround(v * 255.0F), 0L, 255L)));
  return s;
}

}  // namespace

TEST_CASE("span8 : le paquet construit sous Vulkan rend ce que l'interpréteur rend de son graphe",
          "[vk_backend][span8]") {
  TempDir dir("span8");
  auto backend = make_backend(dir.path() / "cache", {.autotune = false});
  if (!backend->profile().target.f16_storage) SKIP("appareil sans stockage 16 bits");
  constexpr int kH = 16, kW = 24;
  const std::vector<float> rgb = pattern(kH, kW);
  const sp::Package p = calibrated(noise_package(2), rgb, kH, kW);
  auto engine = require_value(backend->load(require_value([&] {
    Progress progress;
    return backend->build(write_package(dir.path(), "span_noise", p), shapes(kH, kW), progress);
  }())));
  const std::vector<float> want = interpret(p, {}, rgb, kH, kW)[0].value.f;
  const std::vector<float> got = run_engine(*backend, *engine, rgb, kH, kW);
  REQUIRE(got.size() == want.size());
  const double db = sp::psnr(want, got);
  WARN("PSNR du moteur Vulkan contre l'interpréteur : " << db << " dB");
  CHECK(db > 45.0);
}

TEST_CASE("span8 : tensor_peaks garde le plus grand |valeur| de la fenêtre, sans remise à zéro",
          "[vk_backend][span8]") {
  TempDir dir("peaks");
  auto backend = make_backend(dir.path() / "cache");
  if (!backend->profile().target.f16_storage) SKIP("appareil sans stockage 16 bits");
  constexpr int kC = 3, kH = 6, kW = 8;
  const auto tensor_with = [&](float inside, float outside) {
    std::vector<kernels::PlaneF> planes(kC, kernels::PlaneF(kW, kH));
    for (int c = 0; c < kC; ++c)
      for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
          const bool in = x >= 2 && x < 6 && y >= 1 && y < 5;
          float v = 0.01F * static_cast<float>((x + y + c) % 5);
          if (in && x == 3 && y == 2 + c % 2)
            v = -inside * static_cast<float>(c + 1);  // négatif : |v| compte
          if (!in && x == 0 && y == 0) v = outside;
          planes[static_cast<std::size_t>(c)].data[static_cast<std::size_t>(y * kW + x)] = v;
        }
    auto t = require_value(backend->alloc_tensor(TensorType::F16, Shape{{1, kC, kH, kW}}));
    write_tensor(*backend, t->tensor(), planes_bytes(planes, TensorType::F16));
    return t;
  };
  auto a = tensor_with(2.0F, 100.0F);
  auto b = tensor_with(0.5F, 1000.0F);
  auto result = require_value(vk::Buffer::create(backend->context(), backend->memory(),
                                                 4 * sizeof(std::uint32_t), vk::MemoryUse::Readback));
  std::memset(result.mapped(), 0, 4 * sizeof(std::uint32_t));
  auto* ops = static_cast<vk::VulkanKernelOps*>(backend->kernels());
  const kernels::Window area{.x = 2, .y = 1, .width = 4, .height = 4};
  const std::array first{a->tensor(), b->tensor()};
  require_ok(ops->tensor_peaks(first, area, result.view()));
  // Le second tenseur relevé à la place du premier : un maximum plus petit ne remplace rien.
  const std::array second{b->tensor()};
  require_ok(ops->tensor_peaks(second, area, result.view()));
  // Une lecture attendue sur l'hôte, après tout ce que la file a reçu (wait() n'attend que côté GPU).
  (void)read_tensor(*backend, a->tensor(), 2);
  std::array<float, 4> got{};
  std::memcpy(got.data(), result.mapped(), sizeof(got));
  CHECK(got[0] == 6.0F);  // le plus grand des trois canaux de a : 2 × 3
  CHECK(got[1] == 1.5F);  // celui de b : 0,5 × 3
  CHECK(got[2] == 0.0F);
  CHECK(got[3] == 0.0F);
  // Une fenêtre qui sort du tenseur est refusée.
  const kernels::Window outside{.x = 6, .y = 0, .width = 4, .height = 2};
  CHECK_FALSE(ops->tensor_peaks(second, outside, result.view()));
}

TEST_CASE("span8 : la dérivation calibre, juge et chronomètre ; ses maxima par tuiles sont ceux de l'image",
          "[vk_backend][span8]") {
  TempDir dir("derive");
  constexpr int kW = 96, kH = 80;
  SpanFp8Request request{.package = sp::serialize(noise_package(2)),
                         .calibration = {still(kW, kH, 1), still(kW, kH, 2)},
                         .validation = {still(kW, kH, 3)},
                         .hard = {still(kW, kH, 4)}};
  const auto derive = [&](int tile, bool with_fp16) {
    auto backend = make_backend(dir.path() / "cache",
                                {.autotune = false, .span_fp8_emulated = true, .span_fp8_tile = tile});
    if (!backend->profile().target.f16_storage) SKIP("appareil sans stockage 16 bits");
    SpanFp8Request r = request;
    if (with_fp16) {
      sp::Package fp16 = noise_package(2);
      fp16.fp16 = true;
      Progress progress;
      r.fp16 = require_value(
          backend->build(write_package(dir.path(), "span_fp16", fp16), shapes(kH, kW), progress));
    }
    std::vector<std::string> stages;
    Progress progress{.report = [&](float, const std::string& stage) { stages.push_back(stage); }};
    auto derived = require_value(backend->derive_span_fp8(r, progress));
    CHECK(std::ranges::find(stages, "calibration") != stages.end());
    CHECK(std::ranges::find(stages, "mesure du temps") != stages.end());
    return derived;
  };
  const SpanFp8Derivation whole = derive(0, true);
  const SpanFp8Derivation tiled = derive(40, false);  // demi-tuiles de 20 points, halo de 8 : 4 × 4 tuiles
  const std::size_t served = sp::scale_count(2);
  for (std::size_t k = 0; k < served; ++k) {
    INFO("tenseur " << k);
    CHECK(whole.peaks[k] > 0.0F);
    CHECK(std::abs(tiled.peaks[k] - whole.peaks[k]) <= 1e-3F * whole.peaks[k]);
    // La validation s'ajoute à la calibration.
    CHECK(whole.validation_peaks[k] >= whole.peaks[k]);
  }
  for (std::size_t k = served; k < sp::kScaleCount; ++k) CHECK(whole.peaks[k] == 0.0F);
  const auto scales = sp::scales_from_peaks(whole.peaks, 2);
  REQUIRE(scales);
  CHECK(whole.scales == *scales);
  REQUIRE(whole.scores.size() == 1);
  REQUIRE(whole.hard_scores.size() == 1);
  CHECK(whole.scores_acc16.empty());
  WARN("PSNR FP8 contre FP16 : " << whole.scores[0].psnr_db << " dB, SSIM " << whole.scores[0].ssim);
  CHECK(whole.scores[0].psnr_db > 35.0);
  CHECK(whole.scores[0].ssim > 0.98);
  CHECK(whole.kernels_ms > 0.0);
  CHECK(whole.trt_fp16_ms > 0.0);
  CHECK(tiled.trt_fp16_ms == 0.0);
  CHECK(whole.timed_size == "96x80");
  CHECK(whole.reference == "vulkan-fp16");
  CHECK(whole.gpu_arch.starts_with("vulkan:"));
  CHECK(whole.kernels_version > 0);
}

TEST_CASE("span8 : sans matrices FP8, la dérivation est refusée", "[vk_backend][span8]") {
  TempDir dir("refuse");
  auto backend = make_backend(dir.path() / "cache", {.autotune = false});
  if (backend->profile().target.matrix_f8) SKIP("appareil à matrices FP8");
  SpanFp8Request request{.package = sp::serialize(noise_package(2)), .validation = {still(16, 16, 1)}};
  Progress progress;
  require_refusal(backend->derive_span_fp8(request, progress), RefusalCode::FormatUnsupported);
  CHECK(backend->capabilities().span_fp8_cutlass == false);
  CHECK(backend->capabilities().fp8 == false);
}
