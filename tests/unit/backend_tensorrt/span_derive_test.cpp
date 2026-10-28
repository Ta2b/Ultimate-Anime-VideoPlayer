// La dérivation native du paquet « .span8 » (Backend::derive_span_fp8, docs/span-fp8-cutlass.md) sur de
// vraies images d'anime, contre le SPAN FP16 de TensorRT. Test caché [.span_derive] : il demande le SPAN
// d'origine (AJP_SPAN_FP8_MODEL) et des images PPM P6 de 2560×1440 au moins (AJP_SPAN_DERIVE_IMAGES, séparées
// par des virgules), passés à CMake, jamais copiés dans le dépôt ; sans eux il se SKIP.
//
//   cmake -DAJP_SPAN_FP8_MODEL=.../2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx
//         -DAJP_SPAN_DERIVE_IMAGES=a/s5.ppm,a/s9.ppm,b/s14.ppm ...
//   ajp-gpu ajp_backend_tensorrt_tests "[.span_derive]"
//
// Douze découpes de 1920×1088 à des endroits différents (huit calibrent, quatre sont mises de côté) ; le reçu
// doit passer ses garde-fous (PSNR moyen d'au moins 42 dB), et le réseau FP8 gagner au moins 25 % de temps
// sur le moteur TensorRT FP16 du même modèle, mesurés l'un après l'autre en trois tours.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "backend_fixture.h"
#include "gpu/span_fp8/derive.h"
#include "gpu/span_fp8/onnx_weights.h"
#include "gpu/span_fp8/package.h"
#include "prepare/sha256.h"

namespace ajp::test {
namespace {

namespace fs = std::filesystem;

struct Ppm {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;  // entrelacé
};

std::optional<Ppm> read_ppm(const fs::path& path) {
  std::ifstream file(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  Ppm image;
  file >> magic >> image.width >> image.height >> maxval;
  if (!file || magic != "P6" || maxval != 255) return std::nullopt;
  file.get();
  image.rgb.resize(static_cast<std::size_t>(3 * image.width * image.height));
  file.read(reinterpret_cast<char*>(image.rgb.data()), static_cast<std::streamsize>(image.rgb.size()));
  if (!file) return std::nullopt;
  return image;
}

// Une découpe w×h en plans R, G, B.
gpu::SpanStill crop(const Ppm& image, int x0, int y0, int width, int height) {
  gpu::SpanStill still{.width = width, .height = height};
  const auto plane = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  still.rgb.resize(3 * plane);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c)
        still.rgb[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y * width + x)] =
            image.rgb[static_cast<std::size_t>(((y0 + y) * image.width + (x0 + x)) * 3 + c)];
  return still;
}

std::vector<std::string> split(const std::string& list) {
  std::vector<std::string> out;
  std::istringstream in(list);
  for (std::string item; std::getline(in, item, ',');)
    if (!item.empty()) out.push_back(item);
  return out;
}

gpu::ShapeSet fixed(int h, int w) {
  const gpu::Shape shape{.dims = {1, 3, h, w}};
  return {.min = shape, .opt = shape, .max = shape};
}

double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Millisecondes par passage d'un moteur : 20 passages de chauffe, puis 100 chronométrés.
double per_run_ms(gpu::Backend& backend, gpu::Engine& engine, const gpu::Tensor& in, const gpu::Tensor& out) {
  const std::array inputs{in};
  std::array outputs{out};
  for (int i = 0; i < 20; ++i) REQUIRE(backend.enqueue(engine, inputs, outputs));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const double begin = now_ms();
  for (int i = 0; i < 100; ++i) REQUIRE(backend.enqueue(engine, inputs, outputs));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  return (now_ms() - begin) / 100.0;
}

}  // namespace

TEST_CASE("Backend TensorRT : la dérivation native du SPAN FP8 sur douze images, contre TensorRT FP16",
          "[.span_derive][backend_tensorrt][cuda][span]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const fs::path model(AJP_SPAN_FP8_MODEL);
  const std::vector<std::string> paths = split(AJP_SPAN_DERIVE_IMAGES);
  if (!fs::is_regular_file(model) || paths.empty())
    SKIP("AJP_SPAN_FP8_MODEL et AJP_SPAN_DERIVE_IMAGES (CMake) : le SPAN d'origine et des images PPM");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  if (!loaded.backend->capabilities().span_fp8_cutlass) SKIP("appareil sans les noyaux FP8 CUTLASS (sm 8.9)");

  constexpr int kWidth = 1920, kHeight = 1088;
  constexpr std::array<std::array<int, 2>, 5> kOffsets{{{320, 176}, {0, 0}, {640, 352}, {320, 0}, {0, 352}}};
  std::vector<Ppm> images;
  for (const std::string& path : paths) {
    auto image = read_ppm(path);
    REQUIRE(image.has_value());
    REQUIRE(image->width >= kWidth);
    REQUIRE(image->height >= kHeight);
    images.push_back(std::move(*image));
  }
  gpu::SpanFp8Request request;
  auto weights = gpu::span_fp8::package_from_onnx_file(model);
  REQUIRE(weights);
  request.package = gpu::span_fp8::serialize(*weights);
  for (std::size_t k = 0; k < 12; ++k) {
    const Ppm& image = images[k % images.size()];
    const auto& at = kOffsets[(k / images.size() + k) % kOffsets.size()];
    const int x0 = std::min(at[0], image.width - kWidth), y0 = std::min(at[1], image.height - kHeight);
    (k % 3 == 2 ? request.validation : request.calibration).push_back(crop(image, x0, y0, kWidth, kHeight));
  }
  REQUIRE(request.calibration.size() == 8);
  REQUIRE(request.validation.size() == 4);

  gpu::Progress progress;
  const double begin = now_ms();
  auto derived = loaded.backend->derive_span_fp8(request, progress);
  const double derive_ms = now_ms() - begin;
  INFO((derived ? std::string() : derived.error().detail));
  REQUIRE(derived);
  std::vector<gpu::span_fp8::ImageScore> scores;
  for (const gpu::SpanFp8ImageScore& score : derived->scores) {
    scores.push_back({.psnr_db = score.psnr_db, .ssim = score.ssim});
    WARN("validation : PSNR " << score.psnr_db << " dB, SSIM " << score.ssim);
  }
  const gpu::span_fp8::Verdict verdict = gpu::span_fp8::judge(scores);
  WARN("dérivation en " << derive_ms << " ms ; PSNR moyen " << verdict.psnr_mean_db << " dB, pire "
                        << verdict.psnr_worst_db << " dB, pire SSIM " << verdict.ssim_worst
                        << " ; réseau FP8 " << derived->kernels_ms << " ms ; référence "
                        << derived->reference);
  CHECK(derived->reference == "cutlass-fp16");
  CHECK(verdict.passed);
  CHECK(verdict.psnr_mean_db >= 42.0);

  // Le temps : le paquet dérivé, chargé comme le lecteur le charge, contre le moteur TensorRT FP16.
  const fs::path dir = fs::temp_directory_path() / "ajp-span-derive-backend-test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  gpu::span_fp8::Package package = *weights;
  package.scales = derived->scales;
  const fs::path file = dir / "derive.span8";
  {
    const std::vector<std::byte> bytes = gpu::span_fp8::serialize(package);
    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  gpu::Progress build_progress;
  auto artifact = loaded.backend->build(gpu::ModelSpec{.id = "derive", .onnx_path = file, .sha256 = "0"},
                                        fixed(kHeight, kWidth), build_progress);
  REQUIRE(artifact);
  auto fp8 = loaded.backend->load(*artifact);
  REQUIRE(fp8);
  const auto sha = prepare::sha256_file(model);
  REQUIRE(sha);
  auto reference =
      loaded.backend->build(gpu::ModelSpec{.id = model.stem().string(), .onnx_path = model, .sha256 = *sha},
                            fixed(kHeight, kWidth), build_progress);
  REQUIRE(reference);
  auto fp16 = loaded.backend->load(*reference);
  REQUIRE(fp16);
  const gpu::Shape in_shape{.dims = {1, 3, kHeight, kWidth}},
      out_shape{.dims = {1, 3, 2 * kHeight, 2 * kWidth}};
  auto input = loaded.backend->alloc_tensor(gpu::TensorType::F16, in_shape);
  auto out16 = loaded.backend->alloc_tensor(gpu::TensorType::F16, out_shape);
  auto out8 = loaded.backend->alloc_tensor(gpu::TensorType::F16, out_shape);
  REQUIRE(input);
  REQUIRE(out16);
  REQUIRE(out8);
  REQUIRE(cudaMemset(reinterpret_cast<void*>((*input)->tensor().handle), 0x2C,
                     static_cast<std::size_t>(3 * kHeight * kWidth) * 2) == cudaSuccess);
  std::vector<double> ms16, ms8;
  for (int round = 0; round < 3; ++round) {  // ABAB : un PC qui chauffe ne favorise personne
    ms16.push_back(per_run_ms(*loaded.backend, **fp16, (*input)->tensor(), (*out16)->tensor()));
    ms8.push_back(per_run_ms(*loaded.backend, **fp8, (*input)->tensor(), (*out8)->tensor()));
  }
  std::ranges::sort(ms16);
  std::ranges::sort(ms8);
  const double m16 = ms16[1], m8 = ms8[1];
  WARN("TensorRT FP16 " << m16 << " ms, noyaux FP8 " << m8 << " ms (" << 100.0 * (m8 / m16 - 1.0) << " %)");
  CHECK(m8 <= 0.75 * m16);
  fs::remove_all(dir);
}

}  // namespace ajp::test
