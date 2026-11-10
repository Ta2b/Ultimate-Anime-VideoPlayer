// PSNR et SSIM de la dérivation : mêmes définitions que tools/span/quantize_span.py (sans GPU).
#include "gpu/span_fp8/metrics.h"

#include <algorithm>
#include <limits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ajp::gpu::span_fp8 {
namespace {

// Des aplats à contours francs, planaire 3×h×w.
std::vector<float> picture(int width, int height) {
  std::vector<float> out(static_cast<std::size_t>(3 * width * height));
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x)
        out[static_cast<std::size_t>((c * height + y) * width + x)] =
            static_cast<float>((x / 8 + y / 8 + c) % 4) / 3.0F;
  return out;
}

}  // namespace

TEST_CASE("PSNR : identiques 99 dB, écart constant connu", "[span_fp8][metrics]") {
  const std::vector<float> a(300, 0.5F);
  CHECK(psnr(a, a) == Catch::Approx(99.0));
  const std::vector<float> b(300, 0.6F);
  CHECK(psnr(a, b) == Catch::Approx(20.0).margin(1e-4));  // erreur de 0,1 : 10·log10(1/0,01)
  CHECK(psnr(a, std::vector<float>(10)) == 0.0);          // tailles différentes
}

TEST_CASE("SSIM : 1 pour deux images identiques, moins pour une image abîmée", "[span_fp8][metrics]") {
  constexpr int kW = 64, kH = 48;
  const std::vector<float> a = picture(kW, kH);
  CHECK(ssim(kW, kH, a, a) == Catch::Approx(1.0).margin(1e-12));
  std::vector<float> noisy = a;
  for (std::size_t i = 0; i < noisy.size(); ++i)
    noisy[i] = std::clamp(noisy[i] + static_cast<float>((i * 7919U) % 13U) / 60.0F - 0.1F, 0.0F, 1.0F);
  const double damaged = ssim(kW, kH, a, noisy);
  CHECK(damaged < 0.98);
  CHECK(damaged > 0.0);
  const std::vector<float> flat(a.size(), 0.5F);
  CHECK(ssim(kW, kH, a, flat) < damaged);
  CHECK(ssim(8, 8, std::vector<float>(192), std::vector<float>(192)) == 0.0);  // plus petite que la fenêtre
}

TEST_CASE("SSIM : la valeur de l'outil Python sur deux petites images", "[span_fp8][metrics]") {
  // quantize_span.py ssim() (numpy, float64) sur ces deux images 12×12 : a est un dégradé horizontal, b = a/2
  // + 1/4.
  constexpr int kW = 12, kH = 12;
  std::vector<float> a(3 * kW * kH), b(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(i % 12) / 12.0F;
    b[i] = a[i] * 0.5F + 0.25F;
  }
  CHECK(ssim(kW, kH, a, b) == Catch::Approx(0.8070049854).margin(1e-6));
}

TEST_CASE("le quantile de l'erreur absolue se prend au rang, sur tout ou sur le sombre (3.7a)",
          "[span_fp8][metrics]") {
  // 1000 échantillons : 999 à 0, un à 0,5. Le 99,9e centile (rang 998) vaut 0, le maximum 0,5.
  std::vector<float> a(1000, 0.0F), b(1000, 0.0F);
  b[17] = 0.5F;
  CHECK(abs_error_quantile(a, b, 0.999) == 0.0);
  CHECK(abs_error_quantile(a, b, 1.0) == Catch::Approx(0.5));
  b[18] = 0.25F;  // deux sur mille : le rang 998 tombe sur le second plus grand
  CHECK(abs_error_quantile(a, b, 0.999) == Catch::Approx(0.25));
  CHECK(abs_error_quantile(a, std::vector<float>(3, 0.0F), 0.5) == 0.0);  // tailles différentes

  // 2×1 pixels : un sombre (0,05), un clair (0,8) ; l'erreur du clair ne compte pas sur le sombre.
  const std::vector<float> ref{0.05F, 0.8F, 0.05F, 0.8F, 0.05F, 0.8F};
  const std::vector<float> got{0.06F, 0.5F, 0.07F, 0.5F, 0.05F, 0.5F};
  CHECK(dark_abs_error_quantile(2, 1, ref, got, 1.0) == Catch::Approx(0.02).margin(1e-6));
  CHECK(abs_error_quantile(ref, got, 1.0) == Catch::Approx(0.3).margin(1e-6));
  const std::vector<float> bright(6, 0.9F);
  CHECK(dark_abs_error_quantile(2, 1, bright, got, 1.0) == 0.0);  // aucun pixel sombre
}

TEST_CASE("un échantillon non fini donne la pire note à chaque mesure (A02-01)", "[span_fp8][metrics]") {
  const int w = 32, h = 32;
  const std::vector<float> a(static_cast<std::size_t>(3 * w * h), 0.5F);
  for (const float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()}) {
    std::vector<float> b = a;
    b[static_cast<std::size_t>(w * 16 + 16)] = bad;  // un seul échantillon, au milieu du plan rouge
    CHECK(psnr(a, b) == 0.0);                        // avant : std::min(99, NaN) rendait 99 dB
    CHECK(psnr(b, a) == 0.0);
    CHECK(ssim(w, h, a, b) == 0.0);
    CHECK(ssim(w, h, b, a) == 0.0);
    // Un seul échantillon sur 3072 : sous tout quantile, donc invisible sans la règle.
    CHECK(abs_error_quantile(a, b, 0.5) == kNotFiniteError);
    CHECK(abs_error_quantile(a, b, 0.999) == kNotFiniteError);
    // La référence est claire (0,5 au-dessus du seuil sombre) : le pixel non fini compte quand même.
    CHECK(dark_abs_error_quantile(w, h, a, b, 0.999) == kNotFiniteError);
    CHECK(dark_abs_error_quantile(w, h, b, a, 0.999) == kNotFiniteError);
  }
  // Des images finies gardent leurs notes.
  CHECK(psnr(a, a) == Catch::Approx(99.0));
  CHECK(abs_error_quantile(a, a, 0.999) == 0.0);
}

}  // namespace ajp::gpu::span_fp8
