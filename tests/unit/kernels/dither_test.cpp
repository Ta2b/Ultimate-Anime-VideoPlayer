// Tramage ordonné de la sortie YUV (kernels/spec/color.h : bayer8, quantize_dithered ; ref
// rgb_to_yuv avec Dither::Bayer8), celui de l'export par puits CUDA.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "kernels/ref/color.h"
#include "kernels/spec.h"

using namespace ajp::kernels;

namespace {

constexpr std::array<int, 8> kFirstRow{0, 32, 8, 40, 2, 34, 10, 42};

constexpr bool bayer8_is_a_permutation() {
  std::array<int, 64> seen{};
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      const int v = bayer8(x, y);
      if (v < 0 || v > 63) return false;
      ++seen[static_cast<std::size_t>(v)];
    }
  }
  return std::ranges::all_of(seen, [](int n) { return n == 1; });
}

static_assert(bayer8_is_a_permutation());
static_assert(bayer8(0, 0) == 0 && bayer8(1, 1) == 16 && bayer8(0, 1) == 48 && bayer8(7, 7) == 21);
static_assert(bayer8(8, 0) == bayer8(0, 0) && bayer8(3, 13) == bayer8(3, 5));

}  // namespace

TEST_CASE("tramage : matrice de Bayer d'ordre 8 et ses seuils", "[color][dither]") {
  for (int x = 0; x < 8; ++x) CHECK(bayer8(x, 0) == kFirstRow[static_cast<std::size_t>(x)]);
  float sum = 0.0F;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      const float t = bayer8_threshold(x, y);
      CHECK(t > 0.0F);
      CHECK(t < 1.0F);
      sum += t;
    }
  }
  CHECK(sum == 32.0F);  // moyenne 1/2
}

TEST_CASE("tramage : quantification au seuil, bornes et NaN", "[color][dither]") {
  // Seuil 1/2 : arrondi au plus proche, égalités vers le haut (quantize va au pair).
  CHECK(quantize_dithered(100.4F, Format::Nv12, 0.5F) == 100);
  CHECK(quantize_dithered(100.5F, Format::Nv12, 0.5F) == 101);
  CHECK(quantize(100.5F, Format::Nv12) == 100);
  CHECK(quantize_dithered(100.1F, Format::Nv12, 0.95F) == 101);
  CHECK(quantize_dithered(100.9F, Format::Nv12, 0.05F) == 100);
  CHECK(quantize_dithered(-3.0F, Format::Nv12, 0.99F) == 0);
  CHECK(quantize_dithered(254.7F, Format::Nv12, 0.5F) == 255);
  CHECK(quantize_dithered(1000.0F, Format::Nv12, 0.01F) == 255);
  CHECK(quantize_dithered(std::numeric_limits<float>::quiet_NaN(), Format::Nv12, 0.5F) == 0);
  CHECK(quantize_dithered(std::numeric_limits<float>::infinity(), Format::P010, 0.5F) ==
        max_code(Format::P010));
  // P010 : 10 bits vrais, puis le décalage de 6 bits.
  CHECK(quantize_dithered(64.0F * 500.3F, Format::P010, 0.6F) == 500 * 64);
  CHECK(quantize_dithered(64.0F * 500.3F, Format::P010, 0.75F) == 501 * 64);
}

TEST_CASE("tramage : un aplat garde son niveau moyen, codes voisins seulement", "[color][dither]") {
  const Format format = GENERATE(Format::Nv12, Format::P010);
  const float gray = GENERATE(0.1F, 0.3137F, 0.5F, 0.777F);
  RgbImage image(16, 16, gray);
  const auto plain = ref::rgb_to_yuv(image, format, {});
  const auto dithered = ref::rgb_to_yuv(image, format, {}, kChainChromaFilter, Dither::Bayer8);
  REQUIRE(plain);
  REQUIRE(dithered);
  const YuvScale scale = yuv_scale(format, Range::Limited);
  const auto unit = static_cast<float>(quantizer(format).qdiv);
  const float exact = (gray * scale.yscale + scale.yoff) / unit;  // luma en codes vrais
  double total = 0.0;
  for (std::size_t i = 0; i < dithered->y.data.size(); ++i) {
    const float code = static_cast<float>(dithered->y.data[i]) / unit;
    CHECK(code >= std::floor(exact));
    CHECK(code <= std::floor(exact) + 1.0F);
    total += code;
  }
  CHECK(std::abs(total / static_cast<double>(dithered->y.data.size()) - exact) <= 1.0 / 64.0 + 1e-4);
  // Chroma neutre : 128 niveaux exacts, le seuil n'y change rien sauf l'arrondi d'une égalité.
  for (std::size_t i = 0; i < dithered->u.data.size(); ++i) {
    CHECK(std::abs(int{dithered->u.data[i]} - int{plain->u.data[i]}) <= static_cast<int>(unit));
    CHECK(std::abs(int{dithered->v.data[i]} - int{plain->v.data[i]}) <= static_cast<int>(unit));
  }
}

TEST_CASE("tramage : sans tramage, la sortie reste celle de quantize", "[color][dither]") {
  RgbImage image(8, 4, 0.0F);
  for (int y = 0; y < 4; ++y) {
    for (int x = 0; x < 8; ++x) {
      image.planes[0].at(x, y) = static_cast<float>(x) / 7.0F;
      image.planes[1].at(x, y) = static_cast<float>(y) / 3.0F;
      image.planes[2].at(x, y) = 0.25F;
    }
  }
  const auto by_default = ref::rgb_to_yuv(image, Format::P010, {});
  const auto none = ref::rgb_to_yuv(image, Format::P010, {}, kChainChromaFilter, Dither::None);
  REQUIRE(by_default);
  REQUIRE(none);
  CHECK(by_default->y.data == none->y.data);
  CHECK(by_default->u.data == none->u.data);
  CHECK(by_default->v.data == none->v.data);
}
