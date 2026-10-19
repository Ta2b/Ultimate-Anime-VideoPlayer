#include "real_sdk.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <system_error>

#include <catch2/catch_test_macros.hpp>

#include "kernels/spec/color.h"

namespace ajp::vendor_hdr::test {
namespace {

namespace fs = std::filesystem;

// Hachage entier de (x, y, canal), uniforme sur [0, 1) : un tramage reproductible.
float uniform(std::uint32_t x, std::uint32_t y, std::uint32_t salt) {
  std::uint32_t h = (x * 0x8DA6B343U) ^ (y * 0xD8163841U) ^ (salt * 0xCB1AB31FU);
  h ^= h >> 16U;
  h *= 0x7FEB352DU;
  h ^= h >> 15U;
  h *= 0x846CA68BU;
  h ^= h >> 16U;
  return static_cast<float>(h >> 8U) / 16777216.0F;
}

std::uint32_t quantize8(float value, float noise) {
  const float code = std::round((value * 255.0F) + noise);
  return static_cast<std::uint32_t>(std::clamp(code, 0.0F, 255.0F));
}

struct Ycc10 {
  int y = 64;
  int cb = 512;
  int cr = 512;
};

Ycc10 source_pixel(int x, int y, int width, int height) {
  const double across = static_cast<double>(x) / std::max(1, width - 1);
  if (y < height / 4) return {.y = 64 + static_cast<int>(std::lround(876.0 * across))};
  if (y < height / 2) return {.y = 400, .cb = 64 + static_cast<int>(std::lround(896.0 * across)), .cr = 600};
  return {.y = 64 + static_cast<int>(std::lround(64.0 * across))};  // 0 à 16 niveaux 8 bits
}

}  // namespace

fs::path require_maxine_sdk() {
  const fs::path root(AJP_MAXINE_SDK_DIR);
  std::error_code error;
  if (!fs::exists(root / "include/nvVideoEffects.h", error)) {
    SKIP("SDK Maxine VFX absent de " << root.string() << " (AJP_TEST_MAXINE_SDK)");
  }
  return root;
}

std::unique_ptr<Cuda> require_cuda() {
  std::string why;
  std::unique_ptr<Cuda> cuda = Cuda::open(why);
  if (!cuda) SKIP(why);
  return cuda;
}

TrueHdrConfig sdr_config(int width, int height, std::uintptr_t cuda_context) {
  TrueHdrConfig config{.width = width, .height = height, .cuda_context = cuda_context};
  config.source.primaries = Primaries::Bt709;
  config.source.transfer = Transfer::Bt1886;
  config.source.matrix = ColorMatrix::Bt709;
  config.source.range = ColorRange::Limited;
  return config;
}

std::vector<std::uint32_t> source_rgba8(int width, int height, bool dither) {
  using namespace ajp::kernels;
  const YuvScale scale = yuv_scale(Format::Yuv420p10, Range::Limited);
  const Coefficients bt709 = coefficients(Matrix::Bt709);
  std::vector<std::uint32_t> pixels;
  pixels.reserve(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const Ycc10 ycc = source_pixel(x, y, width, height);
      const Rgb rgb = to_rgb(
          normalize(static_cast<float>(ycc.y), static_cast<float>(ycc.cb), static_cast<float>(ycc.cr), scale),
          bt709);
      const auto ux = static_cast<std::uint32_t>(x);
      const auto uy = static_cast<std::uint32_t>(y);
      std::uint32_t packed = 255U << 24U;
      std::uint32_t channel = 0;
      for (const float value : {rgb.r, rgb.g, rgb.b}) {
        // Triangulaire : somme de deux uniformes, de -1 à +1 code.
        const float noise = dither ? uniform(ux, uy, channel) + uniform(ux, uy, channel + 3) - 1.0F : 0.0F;
        packed |= quantize8(value, noise) << (8U * channel);
        ++channel;
      }
      pixels.push_back(packed);
    }
  }
  return pixels;
}

int near_black_row(int height) {
  return height / 2;
}

std::vector<std::uint32_t> flat_rgba8(int width, int height, std::uint32_t level) {
  return std::vector<std::uint32_t>(static_cast<std::size_t>(width) * static_cast<std::size_t>(height),
                                    level | (level << 8U) | (level << 16U) | (255U << 24U));
}

}  // namespace ajp::vendor_hdr::test
