// Export par puits CUDA du backend TensorRT : write_yuv contre ref::rgb_to_yuv, aux formats, tramages
// et sitings de l'export, sur une image dont la taille n'est pas un multiple des blocs.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "backend_fixture.h"
#include "kernels/ref/color.h"
#include "kernels/ref/half.h"
#include "kernels/spec.h"

namespace ajp::test {
namespace {

using kernels::Dither;
using kernels::Format;
using kernels::Siting;

// Des plans d'appareil au pas d'une image du pool, rendus à la fin du test.
struct DevicePlanes {
  void* luma = nullptr;
  void* chroma = nullptr;
  std::size_t luma_pitch = 0;
  std::size_t chroma_pitch = 0;
  DevicePlanes(int width, int height, std::size_t sample) {
    const std::size_t row = static_cast<std::size_t>(width) * sample;
    REQUIRE(cudaMallocPitch(&luma, &luma_pitch, row, static_cast<std::size_t>(height)) == cudaSuccess);
    REQUIRE(cudaMallocPitch(&chroma, &chroma_pitch, row, static_cast<std::size_t>(height / 2)) ==
            cudaSuccess);
  }
  ~DevicePlanes() {
    cudaFree(luma);
    cudaFree(chroma);
  }
  DevicePlanes(const DevicePlanes&) = delete;
  DevicePlanes& operator=(const DevicePlanes&) = delete;
};

// Les échantillons d'un plan relu, un par élément (8 ou 16 bits).
std::vector<std::uint16_t> read_plane(const void* plane, std::size_t pitch, int samples, int rows,
                                      std::size_t sample) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(samples) * sample *
                                  static_cast<std::size_t>(rows));
  REQUIRE(cudaMemcpy2D(bytes.data(), static_cast<std::size_t>(samples) * sample, plane, pitch,
                       static_cast<std::size_t>(samples) * sample, static_cast<std::size_t>(rows),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  std::vector<std::uint16_t> out(static_cast<std::size_t>(samples) * static_cast<std::size_t>(rows));
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = sample == 2 ? static_cast<std::uint16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8)) : bytes[i];
  }
  return out;
}

bool wait_done(const std::atomic<std::uint64_t>& done, std::uint64_t value) {
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (done.load(std::memory_order_acquire) < value) {
    if (std::chrono::steady_clock::now() > until) return false;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return true;
}

}  // namespace

TEST_CASE("Backend TensorRT : sortie YUV de l'export contre la référence",
          "[backend_tensorrt][cuda][export]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  const Format format = GENERATE(Format::Nv12, Format::P010);
  const Dither dither = GENERATE(Dither::None, Dither::Bayer8);
  const Siting siting = GENERATE(Siting::Left, Siting::TopLeft);
  const kernels::Range range = GENERATE(kernels::Range::Limited, kernels::Range::Full);
  constexpr int w = 70;  // ni 32 ni 64 : des blocs incomplets
  constexpr int h = 38;
  // Dégradés, aplats et un peu hors gamut, arrondis en FP16 comme le tenseur.
  kernels::RgbImage image(w, h);
  std::vector<float> planar(static_cast<std::size_t>(3 * w * h));
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const float raw = c == 0   ? static_cast<float>(x) / (w - 1)
                          : c == 1 ? static_cast<float>((x * 7 + y * 13) % 41) / 38.0F - 0.03F
                                   : (y < h / 2 ? 0.25F : static_cast<float>(y) / h);
        const float value = kernels::ref::half_to_float(kernels::ref::float_to_half(raw));
        image.planes[static_cast<std::size_t>(c)].at(x, y) = value;
        planar[static_cast<std::size_t>((c * h + y) * w + x)] = value;
      }
    }
  }
  const kernels::ref::ColorSpec color{.matrix = kernels::Matrix::Bt709, .range = range, .siting = siting};
  const auto expected = kernels::ref::rgb_to_yuv(image, format, color, kernels::kChainChromaFilter, dither);
  REQUIRE(expected);

  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  CHECK(loaded.backend->capabilities().yuv_targets);
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, h, w}});
  REQUIRE(rgb);
  upload((*rgb)->tensor(), planar);
  const std::size_t sample = format == Format::P010 ? 2 : 1;
  const DevicePlanes planes(w, h, sample);
  std::atomic<std::uint64_t> done{0};
  const gpu::YuvTarget target{.luma = reinterpret_cast<std::uintptr_t>(planes.luma),
                              .chroma = reinterpret_cast<std::uintptr_t>(planes.chroma),
                              .luma_pitch = planes.luma_pitch,
                              .chroma_pitch = planes.chroma_pitch,
                              .width = w,
                              .height = h,
                              .format = format,
                              .matrix = color.matrix,
                              .range = range,
                              .siting = siting,
                              .dither = dither,
                              .done = &done,
                              .done_value = 7};
  REQUIRE(loaded.backend->kernels()->write_yuv((*rgb)->tensor(), target));
  REQUIRE(wait_done(done, 7));

  const int unit = static_cast<int>(kernels::quantizer(format).qdiv);
  const auto luma = read_plane(planes.luma, planes.luma_pitch, w, h, sample);
  const auto chroma = read_plane(planes.chroma, planes.chroma_pitch, w, h / 2, sample);
  int worst = 0;
  std::size_t differ = 0;
  const auto compare = [&](std::uint16_t have, std::uint16_t want) {
    CHECK(have % unit == 0);
    const int gap = std::abs(int{have} - int{want}) / unit;
    worst = std::max(worst, gap);
    if (gap != 0) ++differ;
  };
  for (std::size_t i = 0; i < luma.size(); ++i) compare(luma[i], expected->y.data[i]);
  for (int y = 0; y < h / 2; ++y) {
    for (int x = 0; x < w / 2; ++x) {
      const std::size_t at = static_cast<std::size_t>(y * w + 2 * x);
      compare(chroma[at], expected->u.at(x, y));
      compare(chroma[at + 1], expected->v.at(x, y));
    }
  }
  CHECK(worst <= kernels::tolerance::kCode);
  CHECK(differ * 100 <= luma.size());  // les écarts restent des égalités basculées par la FMA
}

TEST_CASE("Backend TensorRT : write_yuv refuse une cible hors contrat", "[backend_tensorrt][cuda][export]") {
  if (!has_cuda()) SKIP("pas de GPU NVIDIA");
  Loaded loaded;
  REQUIRE(loaded.backend != nullptr);
  auto rgb = loaded.backend->alloc_tensor(gpu::TensorType::F16, {.dims = {1, 3, 8, 8}});
  REQUIRE(rgb);
  const DevicePlanes planes(8, 8, 2);
  std::atomic<std::uint64_t> done{0};
  const gpu::YuvTarget good{.luma = reinterpret_cast<std::uintptr_t>(planes.luma),
                            .chroma = reinterpret_cast<std::uintptr_t>(planes.chroma),
                            .luma_pitch = planes.luma_pitch,
                            .chroma_pitch = planes.chroma_pitch,
                            .width = 8,
                            .height = 8,
                            .done = &done,
                            .done_value = 1};
  gpu::KernelOps& ops = *loaded.backend->kernels();
  gpu::YuvTarget p016 = good;
  p016.format = Format::P016;
  CHECK(ops.write_yuv((*rgb)->tensor(), p016).error().code == RefusalCode::FormatUnsupported);
  gpu::YuvTarget wide = good;
  wide.width = 16;
  CHECK(ops.write_yuv((*rgb)->tensor(), wide).error().code == RefusalCode::ShapeInvalid);
  gpu::YuvTarget silent = good;
  silent.done = nullptr;
  CHECK(ops.write_yuv((*rgb)->tensor(), silent).error().code == RefusalCode::FieldOutOfRange);
  gpu::YuvTarget narrow = good;
  narrow.luma_pitch = 8;  // 8 échantillons de 16 bits n'y tiennent pas
  CHECK(ops.write_yuv((*rgb)->tensor(), narrow).error().code == RefusalCode::ShapeInvalid);
  REQUIRE(ops.write_yuv((*rgb)->tensor(), good));
  CHECK(wait_done(done, 1));
}

}  // namespace ajp::test
