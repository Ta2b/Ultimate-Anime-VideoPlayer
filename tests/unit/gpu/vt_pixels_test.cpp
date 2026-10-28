// Les images de l'interpolation d'Apple (gpu/coreml/vt_pixels.h) : chaque format relu redonne les
// plans écrits, à sa précision près ; l'ordre des canaux, les niveaux YCbCr et le choix du format.
#include "gpu/coreml/vt_pixels.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "kernels/ref/half.h"

namespace ajp::gpu::coreml_rt::vt {
namespace {

constexpr int kWidth = 6;
constexpr int kHeight = 4;
constexpr std::size_t kPixels = static_cast<std::size_t>(kWidth) * kHeight;

// Des plans lisses (la chroma 4:2:0 ne garde pas un détail d'un pixel), en F16.
std::vector<std::byte> smooth_rgb() {
  std::vector<std::byte> rgb(3 * kPixels * 2);
  for (std::size_t c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < kPixels; ++i) {
      const float x = static_cast<float>(i % kWidth);
      const float y = static_cast<float>(i / kWidth);
      const float v = 0.2F + (0.05F * x) + (0.04F * y) + (0.1F * static_cast<float>(c));
      const std::uint16_t h = kernels::ref::float_to_half(v);
      std::memcpy(rgb.data() + (((c * kPixels) + i) * 2), &h, 2);
    }
  }
  return rgb;
}

float at(const std::vector<std::byte>& rgb, std::size_t index) {
  std::uint16_t h = 0;
  std::memcpy(&h, rgb.data() + (index * 2), 2);
  return kernels::ref::half_to_float(h);
}

struct Buffer {
  std::vector<std::byte> luma;
  std::vector<std::byte> chroma;
  Planes planes;
};

Buffer buffer_for(Pixels format) {
  Buffer b;
  if (format == Pixels::RgbaHalf || format == Pixels::Bgra8) {
    const std::size_t stride = static_cast<std::size_t>(kWidth) * (format == Pixels::RgbaHalf ? 8 : 4) + 16;
    b.luma.resize(stride * kHeight);
    b.planes.stride[0] = stride;
  } else {
    const std::size_t sample = format == Pixels::Yuv420Full8 || format == Pixels::Yuv420Video8 ? 1 : 2;
    b.planes.stride[0] = (kWidth * sample) + 8;
    b.planes.stride[1] = (kWidth * sample) + 8;  // Cb et Cr entrelacés : width échantillons
    b.luma.resize(b.planes.stride[0] * kHeight);
    b.chroma.resize(b.planes.stride[1] * (kHeight / 2));
  }
  b.planes.data[0] = b.luma.data();
  b.planes.data[1] = b.chroma.empty() ? nullptr : b.chroma.data();
  return b;
}

}  // namespace

TEST_CASE("vt_pixels : chaque format relu redonne les plans RGB", "[gpu][coreml]") {
  const Pixels format = GENERATE(Pixels::RgbaHalf, Pixels::Bgra8, Pixels::Yuv420Full10, Pixels::Yuv420Video10,
                                 Pixels::Yuv420Full8, Pixels::Yuv420Video8);
  CAPTURE(name(format));
  const std::vector<std::byte> rgb = smooth_rgb();
  Buffer buffer = buffer_for(format);
  // En deux bandes de rangées, comme le découpage du fil.
  pack(format, rgb.data(), TensorType::F16, kWidth, kHeight, buffer.planes, {.first = 0, .last = 2});
  pack(format, rgb.data(), TensorType::F16, kWidth, kHeight, buffer.planes, {.first = 2, .last = kHeight});
  std::vector<float> back(3 * kPixels);
  unpack(format, buffer.planes, reinterpret_cast<std::byte*>(back.data()), TensorType::F32, kWidth, kHeight,
         {.first = 0, .last = kHeight});
  const float tolerance = format == Pixels::RgbaHalf ? 0.0F : is_yuv(format) ? 0.03F : 0.003F;
  for (std::size_t i = 0; i < back.size(); ++i) {
    CAPTURE(i);
    CHECK(std::abs(back[i] - at(rgb, i)) <= tolerance);
  }
}

TEST_CASE("vt_pixels : BGRA dans l'ordre de Core Video, luma et chroma aux niveaux du format",
          "[gpu][coreml]") {
  std::vector<std::byte> red(3 * kPixels * 2);
  const std::uint16_t one = kernels::ref::float_to_half(1.0F);
  for (std::size_t i = 0; i < kPixels; ++i) std::memcpy(red.data() + (i * 2), &one, 2);
  Buffer bgra = buffer_for(Pixels::Bgra8);
  pack(Pixels::Bgra8, red.data(), TensorType::F16, kWidth, kHeight, bgra.planes,
       {.first = 0, .last = kHeight});
  CHECK(std::to_integer<int>(bgra.luma[0]) == 0);
  CHECK(std::to_integer<int>(bgra.luma[1]) == 0);
  CHECK(std::to_integer<int>(bgra.luma[2]) == 255);
  CHECK(std::to_integer<int>(bgra.luma[3]) == 255);

  // Rouge pur en BT.709, plage vidéo 8 bits : Y = 16 + 219 × 0,2126, Cr au plus haut (240).
  Buffer yuv = buffer_for(Pixels::Yuv420Video8);
  pack(Pixels::Yuv420Video8, red.data(), TensorType::F16, kWidth, kHeight, yuv.planes,
       {.first = 0, .last = kHeight});
  CHECK(std::to_integer<int>(yuv.luma[0]) == 63);
  CHECK(std::to_integer<int>(yuv.chroma[1]) == 240);
  // 10 bits : la valeur dans les bits de poids fort.
  Buffer wide = buffer_for(Pixels::Yuv420Full10);
  pack(Pixels::Yuv420Full10, red.data(), TensorType::F16, kWidth, kHeight, wide.planes,
       {.first = 0, .last = kHeight});
  std::uint16_t y10 = 0;
  std::memcpy(&y10, wide.luma.data(), 2);
  CHECK((y10 & 0x3FU) == 0);
  CHECK((y10 >> 6U) == 217);  // 1023 × 0,2126
}

TEST_CASE("vt_pixels : le format préféré parmi ceux qu'offre l'interpolateur", "[gpu][coreml]") {
  const std::vector<std::uint32_t> yuv_only{four_cc(Pixels::Yuv420Video8), four_cc(Pixels::Yuv420Full10)};
  CHECK(preferred(yuv_only) == Pixels::Yuv420Full10);
  const std::vector<std::uint32_t> all{four_cc(Pixels::Bgra8), four_cc(Pixels::RgbaHalf)};
  CHECK(preferred(all) == Pixels::RgbaHalf);
  CHECK_FALSE(preferred(std::vector<std::uint32_t>{0x79343136}).has_value());  // « y416 »
  CHECK(four_cc(Pixels::Yuv420Video8) == 0x34323076U);                         // « 420v »
  CHECK(four_cc(Pixels::RgbaHalf) == 0x52476841U);                             // « RGhA »
}

}  // namespace ajp::gpu::coreml_rt::vt
